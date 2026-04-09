"""
graders.py - Multi-dimensional scoring for Auto-Scaling Infrastructure Agent

Scoring philosophy:
    - 6 dimensions: completion, uptime, sla, cost, stability, scaling efficiency
    - Per-task weight profiles — each task rewards different skills
    - Flat crash penalty (-0.3) as explicit additive deduction
    - aggregate_scores() rolls up all 3 tasks into one leaderboard number
    - All sub-scores clamped [0.0, 1.0] before weighting
    - Final score clamped STRICTLY to (0.001, 0.999) — validator requires
      scores to be strictly between 0 and 1 (exclusive)

Usage:
    from graders import grade_episode, aggregate_scores

    obs, reward, done, info = env.step(action)
    if done:
        result = grade_episode(task_id=1, info=info, task=env.task)
        print(result["final_score"])   # strictly in (0.001, 0.999)
        print(result["breakdown"])     # per-dimension scores
"""

from __future__ import annotations

from typing import Any, Dict

from tasks import Task, get_task


# ─────────────────────────────────────────────
# Score bounds — validator requires STRICTLY between 0 and 1
# ─────────────────────────────────────────────

_SCORE_MIN: float = 0.001
_SCORE_MAX: float = 0.999


def _clamp(value: float) -> float:
    """Clamp a score to (_SCORE_MIN, _SCORE_MAX) — strictly between 0 and 1."""
    return max(_SCORE_MIN, min(_SCORE_MAX, value))


# ─────────────────────────────────────────────
# Weight Profiles (per task)
# ─────────────────────────────────────────────
#
# Task 1 (Easy — Single Spike):
#   Primary test: react to a spike. Uptime + SLA are main signal.
#
# Task 2 (Medium — Wave Management):
#   Primary test: scale up AND back down across waves.
#   Cost and scaling efficiency both matter.
#
# Task 3 (Hard — Adaptive/Uncertainty):
#   Primary test: survive at all under tight constraints.
#   Completion gets the highest weight — finishing is the achievement.

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
    steps_completed = max(1, info.get("steps_completed", 1))
    fraction = min(steps_completed / task.max_steps, 1.0)
    if info.get("termination_reason") == "success":
        return _clamp(1.0)
    return _clamp(fraction ** 0.75)


def _score_uptime(info: Dict[str, Any], task: Task) -> float:
    uptime = max(0.0, min(100.0, info.get("uptime_percentage", 0.0)))
    if uptime >= 95.0:
        raw = 1.0
    elif uptime >= 90.0:
        raw = 0.80 + (uptime - 90.0) / 10.0 * 0.20
    elif uptime >= 70.0:
        raw = 0.40 + (uptime - 70.0) / 20.0 * 0.40
    else:
        raw = max(0.0, uptime / 70.0 * 0.40)
    return _clamp(raw)


def _score_sla(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, info.get("steps_completed", 1))
    sla_violations = info.get("sla_violation_count", 0)
    return _clamp(max(0.0, 1.0 - (sla_violations / steps)))


def _score_cost(info: Dict[str, Any], task: Task) -> float:
    total_cost = info.get("total_cost", 0.0)
    budget = task.budget
    hard_limit = budget * task.budget_failure_multiplier
    if total_cost <= budget:
        return _clamp(1.0)
    if total_cost >= hard_limit:
        return _clamp(0.0)
    overspend_fraction = (total_cost - budget) / (hard_limit - budget)
    return _clamp(max(0.0, 1.0 - overspend_fraction))


def _score_stability(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, info.get("steps_completed", 1))
    critical = info.get("critical_violation_count", 0)
    ratio = critical / steps
    if ratio == 0.0:
        raw = 1.0
    elif ratio <= 0.20:
        raw = 1.0 - (ratio / 0.20) * 0.50
    elif ratio <= 0.50:
        raw = 0.50 - ((ratio - 0.20) / 0.30) * 0.40
    else:
        raw = max(0.0, 0.10 - (ratio - 0.50) * 0.20)
    return _clamp(raw)


def _score_scaling_efficiency(info: Dict[str, Any], task: Task) -> float:
    tolerance = max(1, task.max_steps // 5)
    unnecessary = (
        info.get("unnecessary_scaleups", 0) +
        info.get("unnecessary_scaledowns", 0)
    )
    if unnecessary == 0:
        return _clamp(1.0)
    ratio = unnecessary / tolerance
    if ratio <= 1.0:
        raw = 1.0 - ratio * 0.50
    else:
        raw = max(0.0, 0.50 - (ratio - 1.0) * 0.50)
    return _clamp(raw)


# ─────────────────────────────────────────────
# Main Grader
# ─────────────────────────────────────────────

def grade_episode(
    task_id: int,
    info: Dict[str, Any],
    task: Task | None = None,
) -> Dict[str, Any]:
    """
    Score a completed episode. Returns final_score strictly in (0.001, 0.999).

    Args:
        task_id : integer task ID (1, 2, or 3)
        info    : the info dict returned by env.step() on done=True
        task    : optional Task object (fetched from registry if None)

    Returns dict with final_score, breakdown, weighted, weights, crash_penalty,
    termination, steps, budget_used, uptime_pct.
    """
    if task is None:
        task = get_task(task_id)

    weights = WEIGHT_PROFILES[task_id]

    # Raw sub-scores — each already clamped to (0.001, 0.999)
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

    # Crash penalty — flat explicit deduction on non-success
    crash_penalty = 0.3 if info.get("termination_reason") != "success" else 0.0

    raw_total = sum(weighted.values()) - crash_penalty

    # Final score MUST be strictly between 0 and 1
    final_score = _clamp(raw_total)

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

    Weights: Task 1 = 20%, Task 2 = 35%, Task 3 = 45%.
    Missing tasks default to _SCORE_MIN (not 0.0 — validator constraint).
    """
    task_weights = {1: 0.20, 2: 0.35, 3: 0.45}
    total = sum(
        scores.get(tid, _SCORE_MIN) * w
        for tid, w in task_weights.items()
    )
    return _clamp(total)


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