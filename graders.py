"""
graders.py - Multi-dimensional scoring for Auto-Scaling Infrastructure Agent

Scoring philosophy:
    - 6 dimensions: completion, uptime, sla, cost, stability, scaling efficiency
    - Per-task weight profiles — each task rewards different skills
    - Flat crash penalty (−0.3) as explicit additive deduction
    - aggregate_scores() rolls up all 3 tasks into one leaderboard number
    - All sub-scores clamped [0.0, 1.0] before weighting
    - Final score clamped [0.0, 1.0]

Usage:
    from graders import grade_episode, aggregate_scores

    obs, reward, done, info = env.step(action)
    if done:
        result = grade_episode(task_id=1, info=info, task=env.task)
        print(result["final_score"])   # 0.0 – 1.0
        print(result["breakdown"])     # per-dimension scores
"""

from __future__ import annotations

from typing import Any, Dict

from tasks import Task, get_task


# ─────────────────────────────────────────────
# Weight Profiles (per task)
# ─────────────────────────────────────────────
#
# Task 1 (Easy — Single Spike):
#   Primary test: can the agent react to a spike at all?
#   Uptime + SLA are the main signal. Cost is trivially easy.
#   No scaling efficiency pressure yet.
#
# Task 2 (Medium — Wave Management):
#   Primary test: scale up AND back down across waves.
#   Cost and scaling efficiency now matter — unnecessary servers
#   drain the tighter budget. SLA stays important.
#
# Task 3 (Hard — Adaptive/Uncertainty):
#   Primary test: survive at all under tight constraints.
#   Completion gets the highest weight — finishing is the achievement.
#   Cost is brutally tight, scaling efficiency matters.
#   SLA weight reduced — staying alive matters more than clean SLA.

WEIGHT_PROFILES: Dict[int, Dict[str, float]] = {
    1: {
        "completion":          0.25,
        "uptime":              0.25,
        "sla":                 0.20,
        "cost":                0.10,
        "stability":           0.12,
        "scaling_efficiency":  0.08,
    },
    2: {
        "completion":          0.20,
        "uptime":              0.20,
        "sla":                 0.15,
        "cost":                0.20,
        "stability":           0.12,
        "scaling_efficiency":  0.13,
    },
    3: {
        "completion":          0.30,
        "uptime":              0.15,
        "sla":                 0.10,
        "cost":                0.20,
        "stability":           0.15,
        "scaling_efficiency":  0.10,
    },
}


# ─────────────────────────────────────────────
# Sub-Score Functions
# ─────────────────────────────────────────────

def _score_completion(info: Dict[str, Any], task: Task) -> float:
    """
    Fraction of steps completed before termination.

    Full 1.0 only on success. Partial credit via x^0.75 curve so
    early crashes score much lower than near-complete runs.

        success after 30/30  → 1.0
        crash at 15/30       → 0.50^0.75 = 0.59
        crash at  3/30       → 0.10^0.75 = 0.18
    """
    steps_completed = max(1, info.get("steps_completed", 1))
    fraction = min(steps_completed / task.max_steps, 1.0)

    if info.get("termination_reason") == "success":
        return 1.0

    return round(fraction ** 0.75, 4)


def _score_uptime(info: Dict[str, Any], task: Task) -> float:
    """
    SLA uptime quality — piecewise with steep falloff below 90%.

    uptime_percentage = (safe_steps / total_steps) * 100
    Clamped to [0, 100] defensively even though env guarantees this.

    >= 95%  → 1.0
    >= 90%  → 0.80 – 1.00  (linear)
    >= 70%  → 0.40 – 0.80  (linear)
    <  70%  → 0.00 – 0.40  (linear, steep)
    """
    uptime = max(0.0, min(100.0, info.get("uptime_percentage", 0.0)))

    if uptime >= 95.0:
        return 1.0
    elif uptime >= 90.0:
        return round(0.80 + (uptime - 90.0) / 10.0 * 0.20, 4)
    elif uptime >= 70.0:
        return round(0.40 + (uptime - 70.0) / 20.0 * 0.40, 4)
    else:
        return round(max(0.0, uptime / 70.0 * 0.40), 4)


def _score_sla(info: Dict[str, Any], task: Task) -> float:
    """
    SLA violation rate — kept separate from uptime.

    uptime_percentage measures lost safe steps (outcome).
    sla_score measures raw violation frequency (behaviour).
    An agent can have high uptime but still trigger many SLA
    warnings per step — both signals matter independently.

        0 violations              → 1.0
        violations / steps = 0.5  → 0.5
        violations / steps = 1.0  → 0.0
    """
    steps = max(1, info.get("steps_completed", 1))
    sla_violations = info.get("sla_violation_count", 0)
    return round(max(0.0, 1.0 - (sla_violations / steps)), 4)


def _score_cost(info: Dict[str, Any], task: Task) -> float:
    """
    Cost efficiency relative to budget.

    Within budget               → 1.0
    Up to budget_failure limit  → linear decay to 0.0
    At or beyond hard limit     → 0.0

    Uses budget_failure_multiplier so scoring boundary matches
    the exact termination boundary in environment.py.
    """
    total_cost = info.get("total_cost", 0.0)
    budget = task.budget
    hard_limit = budget * task.budget_failure_multiplier

    if total_cost <= budget:
        return 1.0
    if total_cost >= hard_limit:
        return 0.0

    overspend_fraction = (total_cost - budget) / (hard_limit - budget)
    return round(max(0.0, 1.0 - overspend_fraction), 4)


def _score_stability(info: Dict[str, Any], task: Task) -> float:
    """
    Freedom from critical violations (CPU or queue above critical threshold).

    Uses steps_completed so partial episodes are not penalised for
    steps they never reached.

        0 critical steps          → 1.0
        <= 20% of steps critical  → 0.50 – 1.00
        <= 50% of steps critical  → 0.10 – 0.50
        >  50% critical           → 0.00 – 0.10
    """
    steps = max(1, info.get("steps_completed", 1))
    critical = info.get("critical_violation_count", 0)
    ratio = critical / steps

    if ratio == 0.0:
        return 1.0
    elif ratio <= 0.20:
        return round(1.0 - (ratio / 0.20) * 0.50, 4)
    elif ratio <= 0.50:
        return round(0.50 - ((ratio - 0.20) / 0.30) * 0.40, 4)
    else:
        return round(max(0.0, 0.10 - (ratio - 0.50) * 0.20), 4)


def _score_scaling_efficiency(info: Dict[str, Any], task: Task) -> float:
    """
    Penalise unnecessary or wasteful scaling actions.

    unnecessary_scaleups   = scaled up when system was healthy (wasted cost)
    unnecessary_scaledowns = scaled down which caused overload within 2 steps

    Tolerance scales with episode length so long episodes are not
    unfairly penalised for occasional mistakes.

        tolerance = max_steps // 5

        0 unnecessary                → 1.0
        unnecessary == tolerance     → 0.50
        unnecessary == 2x tolerance  → 0.00
    """
    tolerance = max(1, task.max_steps // 5)
    unnecessary = (
        info.get("unnecessary_scaleups", 0) +
        info.get("unnecessary_scaledowns", 0)
    )

    if unnecessary == 0:
        return 1.0

    ratio = unnecessary / tolerance
    if ratio <= 1.0:
        return round(1.0 - ratio * 0.50, 4)
    else:
        return round(max(0.0, 0.50 - (ratio - 1.0) * 0.50), 4)


# ─────────────────────────────────────────────
# Main Grader
# ─────────────────────────────────────────────

def grade_episode(
    task_id: int,
    info: Dict[str, Any],
    task: Task | None = None,
) -> Dict[str, Any]:
    """
    Score a completed episode from 0.0 to 1.0.

    Args:
        task_id : integer task ID (1, 2, or 3)
        info    : the info dict returned by env.step() on done=True
        task    : optional Task object (fetched from registry if None)

    Returns:
        {
            "task_id"       : int,
            "final_score"   : float (0.0 – 1.0),
            "breakdown"     : { dimension: raw_score, ... },
            "weighted"      : { dimension: raw x weight, ... },
            "weights"       : { dimension: weight, ... },
            "crash_penalty" : float,
            "termination"   : str,
            "steps"         : int,
            "budget_used"   : str,   e.g. "12.50 / 100.00"
            "uptime_pct"    : float,
        }
    """
    if task is None:
        task = get_task(task_id)

    weights = WEIGHT_PROFILES[task_id]

    # Raw sub-scores
    breakdown = {
        "completion":         _score_completion(info, task),
        "uptime":             _score_uptime(info, task),
        "sla":                _score_sla(info, task),
        "cost":               _score_cost(info, task),
        "stability":          _score_stability(info, task),
        "scaling_efficiency": _score_scaling_efficiency(info, task),
    }

    # Weighted contributions
    weighted = {
        dim: round(score * weights[dim], 6)
        for dim, score in breakdown.items()
    }

    # Flat crash penalty — explicit deduction, easy for judges to audit
    crash_penalty = 0.3 if info.get("termination_reason") != "success" else 0.0

    raw_total = sum(weighted.values()) - crash_penalty
    final_score = round(min(1.0, max(0.0, raw_total)), 4)

    return {
        "task_id":       task_id,
        "final_score":   final_score,
        "breakdown":     breakdown,
        "weighted":      weighted,
        "weights":       weights,
        "crash_penalty": crash_penalty,
        "termination":   info.get("termination_reason", "unknown"),
        "steps":         info.get("steps_completed", 0),
        "budget_used":   f"{info.get('total_cost', 0.0):.2f} / {task.budget:.2f}",
        "uptime_pct":    info.get("uptime_percentage", 0.0),
    }


# ─────────────────────────────────────────────
# Leaderboard Roll-up
# ─────────────────────────────────────────────

def aggregate_scores(scores: Dict[int, float]) -> float:
    """
    Aggregate per-task scores into one leaderboard number.

    Weights:
        Task 1 (easy)   → 20%
        Task 2 (medium) → 35%
        Task 3 (hard)   → 45%

    A submission that only solves easy tasks cannot top the leaderboard.
    Missing tasks default to 0.0.

    Args:
        scores: {task_id: final_score}  e.g. {1: 0.9, 2: 0.7, 3: 0.4}

    Returns:
        Weighted aggregate float 0.0 – 1.0.
    """
    task_weights = {1: 0.20, 2: 0.35, 3: 0.45}
    total = sum(scores.get(tid, 0.0) * w for tid, w in task_weights.items())
    return round(total, 4)


# ─────────────────────────────────────────────
# Pretty Printer
# ─────────────────────────────────────────────

def print_grade(result: Dict[str, Any]) -> None:
    """Human-readable grade report."""
    stars = "★★★" if result["final_score"] >= 0.8 else "★★" if result["final_score"] >= 0.5 else "★"
    print(f"\n{'─' * 56}")
    print(f"  Task {result['task_id']} Score Report")
    print(f"{'─' * 56}")
    print(f"  Final Score   : {result['final_score']:.4f}  ({stars})")
    print(f"  Termination   : {result['termination']}")
    print(f"  Steps         : {result['steps']}")
    print(f"  Budget Used   : {result['budget_used']}")
    print(f"  Uptime        : {result['uptime_pct']}%")
    if result["crash_penalty"] > 0:
        print(f"  Crash Penalty : -0.3 (non-success)")
    print(f"\n  {'Dimension':<22} {'Raw':>6}  {'Weight':>7}  {'Weighted':>9}")
    print(f"  {'─' * 50}")
    for dim, raw in result["breakdown"].items():
        w = result["weights"][dim]
        wv = result["weighted"][dim]
        print(f"  {dim:<22} {raw:>6.4f}  {w:>7.2f}  {wv:>9.6f}")
    print(f"  {'─' * 50}")
    if result["crash_penalty"] > 0:
        print(f"  {'crash_penalty':<22} {'':>6}  {'':>7}  {-result['crash_penalty']:>9.4f}")
    print(f"  {'TOTAL':<22} {'':>6}  {'':>7}  {result['final_score']:>9.4f}")
    print(f"{'─' * 56}\n")


# ─────────────────────────────────────────────
# Self-test  (run: python graders.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from environment import AutoScalingEnvironment, ACTION_SCALE_UP, ACTION_SCALE_DOWN, ACTION_HOLD

    print("=" * 60)
    print("  graders.py — Self-test (rule-based agent)")
    print("=" * 60)

    all_task_scores: Dict[int, float] = {}

    for task_id in [1, 2, 3]:
        env = AutoScalingEnvironment()
        obs = env.reset(task_id=task_id)
        task = env.task
        done = False
        final_info = {}

        while not done:
            cpu       = obs["cpu_usage"]
            queue     = obs["queue_length"]
            instances = obs["current_instances"]
            pending   = obs["pending_instances"]
            sla_cpu   = obs["sla_cpu_limit"]
            sla_q     = obs["sla_queue_limit"]
            max_inst  = obs["max_instances"]

            if (cpu > sla_cpu * 0.85 or queue > sla_q * 0.7) and \
               (instances + pending) < max_inst:
                action = ACTION_SCALE_UP
            elif cpu < 45.0 and queue < sla_q * 0.2 and instances > 1:
                action = ACTION_SCALE_DOWN
            else:
                action = ACTION_HOLD

            obs, reward, done, info = env.step(action)
            if done:
                final_info = info

        result = grade_episode(task_id=task_id, info=final_info, task=task)
        all_task_scores[task_id] = result["final_score"]
        print_grade(result)

    leaderboard = aggregate_scores(all_task_scores)
    print("=" * 56)
    print(f"  Leaderboard Score : {leaderboard:.4f}")
    print(f"  (Task 1x0.20 + Task 2x0.35 + Task 3x0.45)")
    print("=" * 56)