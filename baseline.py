from environment import (
    AutoScalingEnvironment,
    ACTION_SCALE_UP,
    ACTION_SCALE_DOWN,
    ACTION_HOLD,
)
from graders import grade_episode, aggregate_scores, print_grade


class RuleBasedAgent:
    def act(self, obs):
        cpu = obs["cpu_usage"]
        queue = obs["queue_length"]
        instances = obs["current_instances"]
        pending = obs["pending_instances"]

        sla_cpu = obs["sla_cpu_limit"]
        sla_q = obs["sla_queue_limit"]
        max_inst = obs["max_instances"]

        total = instances + pending

        if (cpu > sla_cpu * 0.75 or queue > sla_q * 0.6):
            if total < max_inst:
                return ACTION_SCALE_UP

        if cpu < 40 and queue < sla_q * 0.2:
            if instances > 1:
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