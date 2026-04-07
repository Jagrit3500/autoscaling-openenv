from environment import (
    AutoScalingEnvironment,
    ACTION_SCALE_UP,
    ACTION_SCALE_DOWN,
    ACTION_HOLD,
)
from graders import grade_episode, aggregate_scores, print_grade


class RuleBasedAgent:
    def act(self, obs):
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
        total       = instances + pending
 
        # ── Rule 1: Safety — single instance with rising CPU ──────────
        # If only one active server and CPU is climbing, scale up
        # immediately even before hitting the normal threshold.
        # Critical for Task 3 which starts with minimal capacity.
        if instances == 1 and pending == 0 and cpu > sla_cpu * 0.45:
            if total < max_inst:
                return ACTION_SCALE_UP
 
        # ── Rule 2: Wave anticipation (Task 2 specific) ───────────────
        # Traffic in Task 2 alternates every 10 steps.
        # Pre-scale 2 steps before the spike so the server is
        # ready exactly when high traffic arrives (boot_delay = 2).
        # Guards: budget must be healthy, not already over-provisioned.
        in_low_phase  = rps < 150           # currently in low traffic
        wave_boundary = (time_step % 10) >= 8  # 2 steps before next wave
        budget_ok     = budget_left > 5.0   # enough budget to justify
        needs_more    = cpu > 30.0          # not already over-provisioned
 
        if in_low_phase and wave_boundary and budget_ok and needs_more:
            if total < max_inst:
                return ACTION_SCALE_UP
 
        # ── Rule 3: Standard proactive scale up ───────────────────────
        # Trigger at 60% SLA (not 75%) to account for 2-step boot delay.
        # Queue trigger at 30% catches buildup before it becomes critical.
        if (cpu > sla_cpu * 0.60 or queue > sla_q * 0.30):
            if total < max_inst:
                return ACTION_SCALE_UP
 
        # ── Rule 4: Conservative scale down ──────────────────────────
        # Only scale down when CPU is genuinely low AND we are not near
        # a wave boundary (avoids removing a server right before spike).
        near_wave = (time_step % 10) >= 7
        if cpu < 35.0 and queue < sla_q * 0.15 and instances > 1 and not near_wave:
            return ACTION_SCALE_DOWN
 
        return ACTION_HOLD


def main():
    agent = RuleBasedAgent()
    scores = {}

    for task_id in [1, 2, 3]:
        env = AutoScalingEnvironment()
        obs = env.reset(task_id=task_id)

        done = False
        final_info = {}

        while not done:
            action = agent.act(obs)
            obs, _, done, info = env.step(action)

            if done:
                final_info = info

        result = grade_episode(task_id, final_info, env.task)
        print_grade(result)

        scores[task_id] = result["final_score"]

    final_score = aggregate_scores(scores)

    print("=" * 50)
    print(f"BASELINE FINAL SCORE: {final_score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()