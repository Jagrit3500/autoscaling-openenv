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
        max_steps   = obs["max_steps"]
        total       = instances + pending

        # ── Budget guard ──────────────────────────────────────────────
        # If budget is running low, stop scaling up entirely.
        # Remaining budget must cover at least current instances
        # for the remaining steps — otherwise we go bankrupt.
        steps_left = max_steps - time_step
        min_cost_to_finish = instances * 0.5 * steps_left
        budget_tight = budget_left < min_cost_to_finish * 1.2

        # ── Rule 1: Safety — single instance with rising CPU ──────────
        if instances == 1 and pending == 0 and cpu > sla_cpu * 0.45:
            if total < max_inst and not budget_tight:
                return ACTION_SCALE_UP

        # ── Rule 2: Wave anticipation ─────────────────────────────────
        in_low_phase  = rps < 150
        wave_boundary = (time_step % 10) >= 8
        budget_ok     = budget_left > 8.0      # tighter than before
        needs_more    = cpu > 30.0
        if in_low_phase and wave_boundary and budget_ok and needs_more:
            if total < max_inst:
                return ACTION_SCALE_UP

        # ── Rule 3: Standard proactive scale up ───────────────────────
        # Skip if budget is tight
        if not budget_tight:
            if cpu > sla_cpu * 0.60 or queue > sla_q * 0.30:
                if total < max_inst:
                    return ACTION_SCALE_UP

        # ── Rule 4: Aggressive scale down ────────────────────────────
        # Scale down more readily to save budget:
        # - not near a wave boundary
        # - CPU comfortably low
        # - queue nearly empty
        # - budget is tight OR system is very comfortable
        near_wave = (time_step % 10) >= 7
        comfortable = cpu < 35.0 and queue < sla_q * 0.15
        if comfortable and instances > 1 and not near_wave:
            return ACTION_SCALE_DOWN

        # ── Rule 5: Force scale down if budget critical ───────────────
        # Even if near wave boundary, scale down if we're about to
        # run out of budget with more than 10 steps left.
        if budget_tight and instances > 1 and steps_left > 10:
            if cpu < sla_cpu * 0.70 and queue < sla_q * 0.40:
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