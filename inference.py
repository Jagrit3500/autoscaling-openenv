"""
inference.py - Baseline inference script for Auto-Scaling Infrastructure Agent

MANDATORY FORMAT (from hackathon spec):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

Required environment variables:
    API_BASE_URL   The API endpoint for the LLM
    MODEL_NAME     The model identifier
    HF_TOKEN       Your HuggingFace / API key

Run:
    python inference.py                  # all 3 tasks, rule-based agent
    python inference.py --task 1         # single task
    python inference.py --agent llm      # LLM agent (requires HF_TOKEN)
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

# ─────────────────────────────────────────────
# Mandatory env vars (per hackathon spec)
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

BENCHMARK    = "autoscaling-openenv"


# ─────────────────────────────────────────────
# Mandatory stdout logging — ONLY these lines go to stdout
# ─────────────────────────────────────────────

def log_start(task_name: str, model: str) -> None:
    print(f"[START] task={task_name} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────
# Rule-Based Agent (default — no API key needed)
# ─────────────────────────────────────────────

class RuleBasedAgent:
    def act(self, obs: Dict[str, Any]) -> int:
        cpu         = obs["cpu_usage"]
        queue       = obs["queue_length"]
        instances   = obs["current_instances"]
        pending     = obs["pending_instances"]
        sla_cpu     = obs["sla_cpu_limit"]
        sla_q       = obs["sla_queue_limit"]
        max_inst    = obs["max_instances"]
        rps         = obs["requests_per_second"]
        budget_left = obs["budget_remaining"]
        time_step   = obs["time_step"]
        max_steps   = obs["max_steps"]
        total       = instances + pending

        # Budget guard — stop scaling up if we risk running out
        steps_left = max_steps - time_step
        min_cost_to_finish = instances * 0.5 * steps_left
        budget_tight = budget_left < min_cost_to_finish * 1.2

        # Safety: single instance with rising CPU
        if instances == 1 and pending == 0 and cpu > sla_cpu * 0.45:
            if total < max_inst and not budget_tight:
                return ACTION_SCALE_UP

        # Wave anticipation: pre-scale 2 steps before next traffic wave
        in_low_phase  = rps < 150
        wave_boundary = (time_step % 10) >= 8
        budget_ok     = budget_left > 8.0
        needs_more    = cpu > 30.0
        if in_low_phase and wave_boundary and budget_ok and needs_more:
            if total < max_inst:
                return ACTION_SCALE_UP

        # Standard proactive scale up — skip if budget tight
        if not budget_tight:
            if cpu > sla_cpu * 0.60 or queue > sla_q * 0.30:
                if total < max_inst:
                    return ACTION_SCALE_UP

        # Aggressive scale down to save budget
        near_wave = (time_step % 10) >= 7
        comfortable = cpu < 35.0 and queue < sla_q * 0.15
        if comfortable and instances > 1 and not near_wave:
            return ACTION_SCALE_DOWN

        # Force scale down if budget critical and steps remaining
        if budget_tight and instances > 1 and steps_left > 10:
            if cpu < sla_cpu * 0.70 and queue < sla_q * 0.40:
                return ACTION_SCALE_DOWN

        return ACTION_HOLD


# ─────────────────────────────────────────────
# LLM Agent (OpenAI-compatible client — mandatory per spec)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI agent managing cloud server auto-scaling.

At each step you see the current system state and must decide one action.
Your goal is to keep CPU usage below the SLA limit while minimizing cost.

ACTIONS:
  0 = scale_up   (add 1 server, boots after 2 steps)
  1 = scale_down (remove 1 server immediately)
  2 = hold       (do nothing)

RULES:
  - Never exceed max_instances
  - Scale up proactively — servers take 2 steps to boot
  - Scale down only when CPU is genuinely low and queue is empty
  - Watch budget_remaining — if it hits 0 the episode ends

Respond with ONLY a JSON object: {"action": 0} or {"action": 1} or {"action": 2}
No explanation. No markdown. Just the JSON."""


class LLMAgent:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=API_BASE_URL,
            api_key=HF_TOKEN or "no-key",
        )
        self.fallback = RuleBasedAgent()

    def act(self, obs: Dict[str, Any]) -> int:
        prompt = (
            f"Current system state:\n{json.dumps(obs, indent=2)}\n\n"
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
                raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
            parsed = json.loads(raw)
            action = int(parsed.get("action", ACTION_HOLD))
            if action not in (ACTION_SCALE_UP, ACTION_SCALE_DOWN, ACTION_HOLD):
                return ACTION_HOLD
            return action
        except Exception:
            return self.fallback.act(obs)


# ─────────────────────────────────────────────
# Task runner — emits ONLY mandatory stdout logs
# ─────────────────────────────────────────────

def run_task(task_id: int, agent, agent_name: str) -> Dict[str, Any]:
    env  = AutoScalingEnvironment()
    obs  = env.reset(task_id=task_id)
    task = env.task

    log_start(task_name=task.name, model=agent_name)

    rewards: List[float] = []
    step_count = 0
    done = False
    final_info: Dict[str, Any] = {}
    error: Optional[str] = None

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
            reward = 0.0
            done   = True
            error  = str(exc)
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
    success = termination == "success"
    log_end(success=success, steps=step_count, rewards=rewards)

    result = grade_episode(task_id=task_id, info=final_info, task=task)
    result["total_reward"] = round(sum(rewards), 4)
    return result


# ─────────────────────────────────────────────
# Main — no extra prints, only [START]/[STEP]/[END]
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",  type=int, choices=[1, 2, 3], default=None)
    parser.add_argument("--agent", choices=["rule", "llm"], default="rule")
    args = parser.parse_args()

    if args.agent == "llm" and HF_TOKEN:
        agent      = LLMAgent()
        agent_name = MODEL_NAME
    else:
        agent      = RuleBasedAgent()
        agent_name = "rule-based"

    task_ids = [args.task] if args.task else [1, 2, 3]
    scores: Dict[int, float] = {}

    for task_id in task_ids:
        result = run_task(task_id=task_id, agent=agent, agent_name=agent_name)
        scores[task_id] = result["final_score"]

    if len(scores) == 3:
        aggregate_scores(scores)  # compute but do not print


if __name__ == "__main__":
    main()