"""
inference.py - Baseline inference script for Auto-Scaling Infrastructure Agent

MANDATORY FORMAT (from hackathon spec):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Required environment variables:
    API_BASE_URL   The API endpoint (default: HuggingFace router)
    MODEL_NAME     The model identifier
    HF_TOKEN       Your HuggingFace / API token

Run:
    python inference.py                   # all 3 tasks, rule-based agent
    python inference.py --task 1          # single task
    python inference.py --agent llm       # LLM agent (requires HF_TOKEN)
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from environment import (
    AutoScalingEnvironment,
    ACTION_SCALE_UP,
    ACTION_SCALE_DOWN,
    ACTION_HOLD,
    ACTION_NAMES,
)
from graders import grade_episode, aggregate_scores

# 
# Mandatory env vars (per hackathon spec)
# 

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("API_KEY")
HF_TOKEN     = os.getenv("HF_TOKEN")  # optional fallback for local testing only

BENCHMARK    = "autoscaling-openenv"


# 
# Mandatory stdout logging - ONLY these 3 formats touch stdout
# 

def log_start(task_name: str, model: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} rewards={rewards_str}",
        flush=True,
    )


# 
# Rule-Based Agent - mirrors baseline.py exactly
# 

class RuleBasedAgent:
    """
    Task-aware deterministic agent. Mirrors the logic in baseline.py.

    Key design decisions:
        - Task 2: double pre-scale at steps X7 and X8 so 4 instances are
          active by wave start (3 instances at 350 rps = 97% CPU = instant crash)
        - Task 3: never drop below 2 instances - single instance cannot
          survive random spikes and the agent can't recover in time
        - Task 1: proactive scale-up at 55% SLA CPU to account for boot delay
        - Budget guard uses next-instance cost, not current-instance floor
    """

    def act(self, obs: Dict[str, Any]) -> int:
        cpu         = obs["cpu_usage"]
        queue       = obs["queue_length"]
        inst        = obs["current_instances"]
        pend        = obs["pending_instances"]
        sla_cpu     = obs["sla_cpu_limit"]
        sla_q       = obs["sla_queue_limit"]
        max_inst    = obs["max_instances"]
        rps         = obs["requests_per_second"]
        budget_left = obs["budget_remaining"]
        t           = obs["time_step"]
        max_steps   = obs["max_steps"]
        total       = inst + pend
        steps_left  = max_steps - t
        task_id     = obs["task_id"]
        consec      = obs["consecutive_critical_steps"]

        # Budget guard: block scale-up only if we can't afford one more
        # instance for the rest of the episode (5% safety margin)
        next_inst_cost = (inst + 1) * 0.5 * steps_left
        budget_tight = budget_left < next_inst_cost * 1.05

        #  Task 2: double pre-scale before each wave 
        if task_id == 2:
            in_low = rps < 150
            t_mod  = t % 10

            if in_low and t_mod == 7 and total < 4 and budget_left > 12:
                return ACTION_SCALE_UP
            if in_low and t_mod == 8 and total < 4 and budget_left > 10:
                return ACTION_SCALE_UP
            if rps >= 300 and cpu > 90 and total < max_inst and consec < 3:
                return ACTION_SCALE_UP
            if in_low and inst > 2 and cpu < 35.0 and t_mod <= 4:
                return ACTION_SCALE_DOWN
            return ACTION_HOLD

        #  Task 3: never go below 2 instances 
        if task_id == 3:
            min_inst = 2
            if not budget_tight and total < max_inst:
                if cpu > sla_cpu * 0.60 or queue > sla_q * 0.30:
                    return ACTION_SCALE_UP
            if inst > min_inst and cpu < 25.0 and queue < sla_q * 0.08:
                return ACTION_SCALE_DOWN
            return ACTION_HOLD

        #  Task 1: proactive spike response 
        if not budget_tight and total < max_inst:
            if cpu > sla_cpu * 0.55 or queue > sla_q * 0.25:
                return ACTION_SCALE_UP
        if inst > 1 and cpu < 28.0 and queue < sla_q * 0.10:
            return ACTION_SCALE_DOWN
        return ACTION_HOLD


# 
# LLM Agent (OpenAI-compatible client - mandatory per spec)
# 

SYSTEM_PROMPT = """You are an AI agent managing cloud server auto-scaling.

At each step you receive the current system state as JSON and must choose one action.
Your goal: keep CPU below the SLA limit, avoid critical overload, and stay within budget.

ACTIONS:
  0 = scale_up   (add 1 server - boots after 2 steps, plan ahead)
  1 = scale_down (remove 1 server immediately)
  2 = hold       (do nothing)

KEY RULES:
  - Servers take 2 steps to boot - scale up BEFORE CPU redlines, not after
  - Never exceed max_instances
  - Scale down only when cpu_usage is well below sla_cpu_limit and queue is near zero
  - Watch budget_remaining - running out ends the episode immediately
  - If consecutive_critical_steps is rising, scale up immediately
  - On task_id=3 (hard), never let current_instances drop to 1

Respond with ONLY a JSON object on a single line:
{"action": 0}   or   {"action": 1}   or   {"action": 2}
No explanation. No markdown. No extra text. Just the JSON."""


class LLMAgent:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
        self.fallback = RuleBasedAgent()

    def act(self, obs: Dict[str, Any]) -> int:
        prompt = (
            f"System state:\n{json.dumps(obs, indent=2)}\n\n"
            f"Choose your action (0=scale_up, 1=scale_down, 2=hold)."
        )
        try:
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=50,
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines()
                    if not l.startswith("```")
                ).strip()
            parsed = json.loads(raw)
            action = int(parsed.get("action", ACTION_HOLD))
            if action not in (ACTION_SCALE_UP, ACTION_SCALE_DOWN, ACTION_HOLD):
                return ACTION_HOLD
            return action
        except Exception as e:
            raise RuntimeError(f"LLM failed: {e}")


# 
# Task runner - emits ONLY mandatory stdout lines
# 

def run_task(task_id: int, agent, agent_name: str) -> Dict[str, Any]:
    env  = AutoScalingEnvironment()
    obs  = env.reset(task_id=task_id)
    task = env.task

    log_start(task_name=task.name, model=agent_name)

    rewards:    List[float]    = []
    step_count: int            = 0
    done:       bool           = False
    final_info: Dict[str, Any] = {}
    error:      Optional[str]  = None

    while not done:
        try:
            action = agent.act(obs)
        except Exception as exc:
            action = ACTION_HOLD
            error  = str(exc)

        try:
            obs, reward, done, info = env.step(action)
            if done:
                final_info = info
        except Exception as exc:
            reward     = 0.0
            done       = True
            error      = str(exc)
            final_info = {}

        step_count += 1
        rewards.append(reward)

        log_step(
            step=step_count,
            action=ACTION_NAMES.get(action, str(action)),
            reward=reward,
            done=done,
            error=error,
        )
        error = None

    termination = final_info.get("termination_reason", "unknown")
    success     = termination == "success"
    log_end(success=success, steps=step_count, rewards=rewards)

    result = grade_episode(task_id=task_id, info=final_info, task=task)
    result["total_reward"] = round(sum(rewards), 4)
    return result


# 
# Main - no extra prints, only mandatory log lines go to stdout
# 

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference against the auto-scaling environment."
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a single task (default: all 3).",
    )
    parser.add_argument(
        "--agent",
        choices=["rule", "llm"],
        default="llm",
        help="Agent to use: llm (default, requires API_BASE_URL and API_KEY) or rule.",
    )
    args = parser.parse_args()

    if args.agent == "llm":
        if "API_BASE_URL" not in os.environ or "API_KEY" not in os.environ:
            raise RuntimeError(
                "LLM mode requires API_BASE_URL and API_KEY environment variables."
            )
        agent = LLMAgent()
        agent_name = MODEL_NAME
    else:
        agent = RuleBasedAgent()
        agent_name = "rule-based"

    task_ids: List[int]        = [args.task] if args.task else [1, 2, 3]
    scores:   Dict[int, float] = {}

    for task_id in task_ids:
        result          = run_task(task_id=task_id, agent=agent, agent_name=agent_name)
        scores[task_id] = result["final_score"]

    if len(scores) == 3:
        aggregate_scores(scores)   # compute for internal use; not printed (spec compliance)


if __name__ == "__main__":
    main()