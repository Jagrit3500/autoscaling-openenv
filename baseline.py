"""
baseline.py - Rule-based baseline agent for Auto-Scaling Infrastructure Agent

Key improvements over naive thresholds:
    - Task-specific logic - each task has different constraints and patterns
    - Task 2: double pre-scale before each wave (both servers active by wave start)
    - Task 3: never scale below 2 instances (single instance dies to any spike)
    - Budget guard uses next-instance cost, not current-instance floor

Run all 3 tasks:
    python baseline.py

Run a single task:
    python baseline.py --task 1
"""

import argparse

from environment import (
    AutoScalingEnvironment,
    ACTION_SCALE_UP,
    ACTION_SCALE_DOWN,
    ACTION_HOLD,
)
from graders import grade_episode, grade_episode_details, aggregate_scores, print_grade


class RuleBasedAgent:
    """
    Task-aware deterministic agent for server auto-scaling.

    Design rationale:
        Each task has a fundamentally different constraint:
        - Task 1: react to a single spike in time (boot delay awareness)
        - Task 2: pre-scale TWO servers before each wave so 4 are active
                  at wave start (3 instances at 350 rps = 97% CPU = instant crash)
        - Task 3: never drop to 1 instance - single instance cannot survive
                  random spikes, and budget is too tight to recover from a crash

    Why task-specific logic outperforms generic thresholds:
        A generic CPU threshold treats Task 2 and Task 3 identically.
        Task 2 has a predictable wave pattern that rewards anticipation.
        Task 3 has random traffic that punishes over-aggressive scale-down.
        Separate logic paths capture both requirements cleanly.
    """

    def act(self, obs: dict) -> int:
        cpu        = obs["cpu_usage"]
        queue      = obs["queue_length"]
        inst       = obs["current_instances"]
        pend       = obs["pending_instances"]
        sla_cpu    = obs["sla_cpu_limit"]
        sla_q      = obs["sla_queue_limit"]
        max_inst   = obs["max_instances"]
        rps        = obs["requests_per_second"]
        budget_left = obs["budget_remaining"]
        t          = obs["time_step"]
        max_steps  = obs["max_steps"]
        total      = inst + pend
        steps_left = max_steps - t
        task_id    = obs["task_id"]
        consec     = obs["consecutive_critical_steps"]

        #  Budget guard 
        # Block scale-up only if we can't afford one more server
        # for the remainder of the episode (with 5% safety margin).
        next_inst_cost = (inst + 1) * 0.5 * steps_left
        budget_tight = budget_left < next_inst_cost * 1.05

        # 
        # TASK 2 - Traffic Wave Management
        # Waves alternate between 100 rps (low) and 350 rps (high).
        # 3 instances at 350 rps = 97.2% CPU - above critical threshold.
        # 4 instances at 350 rps = 72.9% CPU - safely below SLA.
        # Strategy: pre-scale TWO servers at steps 7 and 8 of each cycle,
        # so both are fully active (booted) by step 10 when the wave hits.
        # Scale back to 2 during troughs to stay within budget.
        # 
        if task_id == 2:
            in_low = rps < 150
            t_mod  = t % 10

            # Pre-scale server 1: fires at step X7 (3 steps before wave)
            if in_low and t_mod == 7 and total < 4 and budget_left > 12:
                return ACTION_SCALE_UP

            # Pre-scale server 2: fires at step X8 (2 steps before wave)
            if in_low and t_mod == 8 and total < 4 and budget_left > 10:
                return ACTION_SCALE_UP

            # Emergency: wave arrived and we're underpowered
            if rps >= 300 and cpu > 90 and total < max_inst and consec < 3:
                return ACTION_SCALE_UP

            # Scale down during trough - not near next wave boundary
            if in_low and inst > 2 and cpu < 35.0 and t_mod <= 4:
                return ACTION_SCALE_DOWN

            return ACTION_HOLD

        # 
        # TASK 3 - Adaptive Scaling Under Uncertainty
        # Random traffic 50-480 rps. Budget=40 is very tight.
        # Critical insight: never drop to 1 instance.
        # 1 instance handles only 100 rps. Any spike to 150+ rps
        # sends CPU critical and the agent can't recover fast enough.
        # 2 instances handles up to 200 rps safely - covers most steps.
        # 
        if task_id == 3:
            min_inst = 2  # absolute floor - never go below this

            # Scale up if load is rising and budget allows
            if not budget_tight and total < max_inst:
                if cpu > sla_cpu * 0.60 or queue > sla_q * 0.30:
                    return ACTION_SCALE_UP

            # Scale down only when genuinely quiet AND above floor
            if inst > min_inst and cpu < 25.0 and queue < sla_q * 0.08:
                return ACTION_SCALE_DOWN

            return ACTION_HOLD

        # 
        # TASK 1 - Single Spike Recovery
        # Standard proactive scaling with boot delay awareness.
        # Act at 55% of SLA CPU (not 85%) so the new server is active
        # before CPU redlines.
        # 
        if not budget_tight and total < max_inst:
            if cpu > sla_cpu * 0.55 or queue > sla_q * 0.25:
                return ACTION_SCALE_UP

        if inst > 1 and cpu < 28.0 and queue < sla_q * 0.10:
            return ACTION_SCALE_DOWN

        return ACTION_HOLD


def main():
    parser = argparse.ArgumentParser(
        description="Run the rule-based baseline agent."
    )
    parser.add_argument(
        "--task",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run a single task (default: all 3).",
    )
    args = parser.parse_args()

    agent    = RuleBasedAgent()
    task_ids = [args.task] if args.task else [1, 2, 3]
    scores   = {}

    for task_id in task_ids:
        env  = AutoScalingEnvironment()
        obs  = env.reset(task_id=task_id)
        done = False
        final_info = {}

        while not done:
            action = agent.act(obs)
            obs, _, done, info = env.step(action)
            if done:
                final_info = info

        result = grade_episode_details(task_id, final_info, env.task)
        print_grade(result)
        scores[task_id] = result["final_score"]

    if len(scores) == 3:
        final_score = aggregate_scores(scores)
        print("=" * 50)
        print(f"BASELINE FINAL SCORE: {final_score:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    main()