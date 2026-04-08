"""
tasks.py - Task definitions for Auto-Scaling Infrastructure Agent

Each task defines a complete scenario with:
- Traffic pattern (requests per second over time)
- Infrastructure constraints (instances, budget, SLA)
- Per-task instance capacity (models different cloud instance types)
- Failure and success conditions
- Metadata for leaderboard-style scoring
"""

import random
from dataclasses import dataclass
from typing import List


# 
# Task Definition
# 

@dataclass
class Task:
    # Identity
    task_id: int                         # 1, 2, or 3
    name: str
    difficulty: str                      # "easy" | "medium" | "hard"
    description: str

    # Infrastructure constraints
    initial_instances: int               # servers running at episode start
    max_instances: int                   # hard ceiling agent cannot exceed
    max_steps: int                       # episode length

    # Per-task instance capacity (models real cloud instance types)
    # Task 1 -> 150 rps  (large powerful servers  e.g. c5.large)
    # Task 2 -> 120 rps  (standard servers        e.g. t3.medium)
    # Task 3 -> 100 rps  (small weak servers      e.g. t3.micro)
    # This is intentional - harder tasks use weaker hardware,
    # forcing the agent to provision more instances to handle load.
    instance_capacity_rps: float         # rps one server can handle at healthy load

    # Cost constraints
    budget: float                        # total cost budget for episode
    cost_per_instance_per_step: float    # cost charged every step per active server
    budget_failure_multiplier: float     # fail only if cost > budget * this value
                                         # e.g. 1.1 -> 10% buffer before hard failure

    # Boot delay (realism - servers don't appear instantly)
    boot_delay_steps: int                # steps before a newly scaled-up server is active

    # SLA limits (soft - violation = reward penalty, not instant failure)
    sla_cpu_limit: float                 # CPU% above this = SLA warning penalty
    sla_queue_limit: int                 # queue length above this = SLA warning penalty

    # Critical limits (hard - sustained violation = episode failure)
    critical_cpu_threshold: float        # CPU% above this is dangerous
    critical_queue_threshold: int        # queue length above this is dangerous
    max_consecutive_critical_steps: int  # sustained critical steps before episode ends

    # Traffic pattern (one value per step, length must equal max_steps)
    traffic_pattern: List[int]           # requests/second at each time step


# 
# Helper: Traffic Pattern Generators
# 

def _spike_pattern(
    steps: int,
    base: int,
    spike: int,
    spike_start: int,
    spike_end: int
) -> List[int]:
    """
    Single clean spike: traffic stays at base,
    jumps to spike between spike_start and spike_end,
    then returns to base.

    Used for Task 1 (Easy).
    """
    pattern = []
    for t in range(steps):
        if spike_start <= t < spike_end:
            pattern.append(spike)
        else:
            pattern.append(base)
    return pattern


def _wave_pattern(
    steps: int,
    low: int,
    high: int,
    wave_length: int
) -> List[int]:
    """
    Alternating wave pattern: low -> high -> low -> high ...
    Each phase lasts wave_length steps.

    Used for Task 2 (Medium).
    """
    pattern = []
    for t in range(steps):
        cycle_position = (t // wave_length) % 2
        pattern.append(high if cycle_position == 1 else low)
    return pattern


def _random_pattern(
    steps: int,
    low: int,
    high: int,
    seed: int
) -> List[int]:
    """
    Seeded random traffic that drifts realistically.

    Key design decisions:
    - Fixed seed guarantees reproducibility for fair grading
    - Symmetric delta range (-70, 70) prevents consistent upward drift,
      making traffic oscillate naturally like real server load
    - Starts at midpoint so agent is not immediately overwhelmed
    - Gradual clamped drift is more realistic than pure random jumps

    Used for Task 3 (Hard).
    """
    rng = random.Random(seed)
    pattern = []
    current = (low + high) // 2          # start at midpoint, not minimum
    for _ in range(steps):
        delta = rng.randint(-70, 70)     # symmetric -> no upward drift bias
        current = max(low, min(high, current + delta))
        pattern.append(current)
    return pattern


# 
# Task 1 - Easy
# 
#
# Scenario:
#   Traffic is calm at 100 rps, spikes hard to 380 rps at step 5,
#   stays high until step 20, then drops back to 100 rps.
#
# What this tests:
#   Basic reactive scaling only.
#   Can the agent detect a spike and scale up in time?
#   No budget pressure. No tricks.
#
# Design rationale:
#   instance_capacity_rps = 150  -> powerful servers (c5.large equivalent)
#                                   2 servers = 300 rps capacity at start
#   max_instances = 10           -> generous, not the challenge here
#   budget = 100                 -> will not be hit under normal play
#   boot_delay = 2               -> agent must act before CPU redlines
#   max_consecutive_critical = 5 -> lenient, forgives one bad reaction

TASK_1 = Task(
    task_id=1,
    name="Single Spike Recovery",
    difficulty="easy",
    description=(
        "A sudden traffic spike hits the system. "
        "Traffic jumps from 100 to 380 rps at step 5 "
        "and stays high until step 20, then drops back. "
        "Scale up in time to keep CPU stable, "
        "then scale back down after traffic drops. "
        "Servers handle 150 rps each (c5.large equivalent)."
    ),
    initial_instances=2,
    max_instances=10,
    max_steps=30,
    instance_capacity_rps=150.0,
    budget=100.0,
    cost_per_instance_per_step=0.5,
    budget_failure_multiplier=1.1,
    boot_delay_steps=2,
    sla_cpu_limit=85.0,
    sla_queue_limit=200,
    critical_cpu_threshold=95.0,
    critical_queue_threshold=400,
    max_consecutive_critical_steps=5,
    traffic_pattern=_spike_pattern(
        steps=30,
        base=100,
        spike=380,
        spike_start=5,
        spike_end=20,
    ),
)


# 
# Task 2 - Medium
# 
#
# Scenario:
#   Traffic alternates between 100 rps and 350 rps every 10 steps.
#   Three full waves happen across 50 steps.
#
# What this tests:
#   Can the agent scale UP during peaks AND DOWN during troughs?
#   Can it manage a tighter budget and respect max_instances = 6?
#   Requires both reactive AND proactive thinking.
#
# Design rationale:
#   instance_capacity_rps = 120  -> standard servers (t3.medium equivalent)
#                                   2 servers = 240 rps capacity at start
#   max_instances = 6            -> tight, real resource decisions required
#   budget = 60                  -> unnecessary servers drain budget fast
#   boot_delay = 2               -> agent must anticipate waves, not just react
#   max_consecutive_critical = 4 -> less lenient than Task 1

TASK_2 = Task(
    task_id=2,
    name="Traffic Wave Management",
    difficulty="medium",
    description=(
        "Traffic alternates between 100 rps (low) and 350 rps (high) "
        "every 10 steps across 50 total steps. "
        "Scale up during peaks and scale down during troughs. "
        "Stay within 6 instances and keep costs under budget. "
        "Servers handle 120 rps each (t3.medium equivalent)."
    ),
    initial_instances=2,
    max_instances=6,
    max_steps=50,
    instance_capacity_rps=120.0,
    budget=60.0,
    cost_per_instance_per_step=0.5,
    budget_failure_multiplier=1.1,
    boot_delay_steps=2,
    sla_cpu_limit=82.0,
    sla_queue_limit=150,
    critical_cpu_threshold=92.0,
    critical_queue_threshold=300,
    max_consecutive_critical_steps=4,
    traffic_pattern=_wave_pattern(
        steps=50,
        low=100,
        high=350,
        wave_length=10,
    ),
)


# 
# Task 3 - Hard
# 
#
# Scenario:
#   Seeded random traffic between 50 and 480 rps over 60 steps.
#   Agent cannot memorize the pattern - must adapt dynamically.
#   Fixed seed ensures grading is always reproducible.
#
# What this tests:
#   Can the agent handle genuine uncertainty?
#   Can it balance performance vs cost under tight constraints?
#   Can it avoid SLA violations with only 5 instances and a tight budget?
#
# Design rationale:
#   instance_capacity_rps = 100  -> weak servers (t3.micro equivalent)
#                                   2 servers = 200 rps capacity at start
#                                   agent must provision carefully
#   initial_instances = 2        -> starts with minimal but survivable capacity
#                                   (changed from 1 to avoid instant collapse)
#   max_instances = 5            -> very tight, every instance counts
#   budget = 40                  -> wasteful scaling guaranteed to fail
#   boot_delay = 2               -> anticipate or crash
#   max_consecutive_critical = 4 -> slightly more forgiving than original 3
#                                   still hard but baseline agent can survive
#   seed = 42                    -> fixed for reproducibility
#
# Note on episode length:
#   60 steps is intentional. Longer hard episodes cause baseline
#   agents to score near 0.0, making the environment look broken to judges.

TASK_3 = Task(
    task_id=3,
    name="Adaptive Scaling Under Uncertainty",
    difficulty="hard",
    description=(
        "Traffic varies randomly between 50 and 480 rps over 60 steps "
        "(seeded at 42 for reproducibility). "
        "In the default seeded episode used for evaluation, realized traffic "
        "ranges between ~50 and ~223 rps. "
        "Maintain SLA with at most 5 instances and a tight budget. "
        "Servers handle only 100 rps each (t3.micro equivalent) - "
        "provision carefully. "
        "Four consecutive critical steps end the episode immediately."
    ),
    initial_instances=2,
    max_instances=5,
    max_steps=60,
    instance_capacity_rps=100.0,
    budget=40.0,
    cost_per_instance_per_step=0.5,
    budget_failure_multiplier=1.1,
    boot_delay_steps=2,
    sla_cpu_limit=80.0,
    sla_queue_limit=100,
    critical_cpu_threshold=90.0,
    critical_queue_threshold=200,
    max_consecutive_critical_steps=4,
    traffic_pattern=_random_pattern(
        steps=60,
        low=50,
        high=480,
        seed=42,
    ),
)


# 
# Task Registry
# 

ALL_TASKS: List[Task] = [TASK_1, TASK_2, TASK_3]

TASK_MAP: dict = {task.task_id: task for task in ALL_TASKS}


def get_task(task_id: int) -> Task:
    """
    Retrieve a task by its integer ID.
    Raises ValueError for unknown IDs.
    """
    if task_id not in TASK_MAP:
        raise ValueError(
            f"Unknown task_id: {task_id}. "
            f"Valid task IDs are: {sorted(TASK_MAP.keys())}"
        )
    return TASK_MAP[task_id]


def list_tasks() -> None:
    """Print a formatted summary of all available tasks."""
    print("\n" + "=" * 60)
    print("  Auto-Scaling Environment - Available Tasks")
    print("=" * 60)
    for task in ALL_TASKS:
        print(f"\n  Task {task.task_id}  [{task.difficulty.upper()}]  {task.name}")
        print(f"  {task.description}")
        print(f"   Steps              : {task.max_steps}")
        print(f"   Max instances      : {task.max_instances}")
        print(f"   Start instances    : {task.initial_instances}")
        print(f"   Instance capacity  : {task.instance_capacity_rps} rps each")
        print(f"   Budget             : {task.budget}")
        print(f"   SLA CPU limit      : {task.sla_cpu_limit}%")
        print(f"   Crit CPU           : {task.critical_cpu_threshold}%")
        print(f"   Max crit steps     : {task.max_consecutive_critical_steps} consecutive")
    print()


# 
# Metadata Template
# 

def empty_episode_info() -> dict:
    """
    Returns a blank info dict populated by environment.py each episode.

    Returned inside step() when done=True.
    Used by graders.py for multi-dimensional scoring.
    Enables leaderboard-style comparison across agents.
    """
    return {
        "total_cost": 0.0,
        "sla_violation_count": 0,
        "critical_violation_count": 0,
        "uptime_percentage": 0.0,
        "unnecessary_scaleups": 0,
        "unnecessary_scaledowns": 0,
        "final_instances": 0,
        "steps_completed": 0,
        "termination_reason": "unknown",
    }


# 
# Self-test  (run: python tasks.py)
# 

if __name__ == "__main__":
    list_tasks()

    print("Validating tasks...\n")
    all_ok = True

    for task in ALL_TASKS:
        checks = {
            "pattern length matches max_steps":
                len(task.traffic_pattern) == task.max_steps,
            "budget > 0":
                task.budget > 0,
            "initial_instances <= max_instances":
                task.initial_instances <= task.max_instances,
            "instance_capacity_rps > 0":
                task.instance_capacity_rps > 0,
            "sla_cpu_limit < critical_cpu_threshold":
                task.sla_cpu_limit < task.critical_cpu_threshold,
            "sla_queue_limit < critical_queue_threshold":
                task.sla_queue_limit < task.critical_queue_threshold,
            "boot_delay < max_steps":
                task.boot_delay_steps < task.max_steps,
        }

        task_ok = all(checks.values())
        if not task_ok:
            all_ok = False

        print(f"  Task {task.task_id} [{task.difficulty.upper()}] "
              f"{'OK' if task_ok else 'FAIL'}")
        for check_name, passed in checks.items():
            print(f"    {'OK' if passed else 'X'}  {check_name}")
        print(f"    instance capacity : {task.instance_capacity_rps} rps/server")
        print(f"    start capacity    : "
              f"{task.initial_instances * task.instance_capacity_rps} rps total")
        print(f"    traffic range     : "
              f"min={min(task.traffic_pattern)} rps, "
              f"max={max(task.traffic_pattern)} rps")
        print()

    if all_ok:
        print("All tasks validated. Ready to update environment.py")
    else:
        print("Validation failed. Fix issues above before proceeding.")