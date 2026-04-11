from __future__ import annotations

import math
from typing import Any, Dict

from tasks import Task, get_task

# Keep scores safely inside (0, 1) even under very coarse external rounding.
EPS = 1e-1


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        v = int(float(value))
        return v
    except Exception:
        return default


# Keep the active score safely away from the boundaries.
def strict_unit_interval(x: float) -> float:
    v = float(x)
    if not math.isfinite(v):
        return EPS
    return max(EPS, min(1.0 - EPS, v))


def strict_rounded_unit_interval(x: float, digits: int = 6) -> float:
    return strict_unit_interval(round(float(x), digits))


def _normalize_scoring_inputs(
    task_id: Any = None,
    info: Any = None,
    task: Any = None,
) -> tuple[int, Dict[str, Any], Task]:
    # Accept legacy and validator variants, e.g.:
    # grade_episode(task_id, info, task)
    # grade_episode(info, task)
    # grade_episode(info)
    # grade_episode(task_id=<...>, info=<...>, task=<...>)
    if isinstance(task_id, dict):
        if isinstance(info, Task) and task is None:
            task = info
        info = task_id
        task_id = None

    if isinstance(info, Task) and task is None:
        task = info
        info = None

    if info is None:
        info = {}
    if not isinstance(info, dict):
        try:
            info = dict(info)
        except Exception:
            info = {}

    normalized_task_id = task_id
    if normalized_task_id is None:
        normalized_task_id = info.get("task_id")
    if normalized_task_id is None and isinstance(task, Task):
        normalized_task_id = task.task_id
    if normalized_task_id is None:
        normalized_task_id = 1

    normalized_task_id = _to_int(normalized_task_id, 1)
    if normalized_task_id not in WEIGHT_PROFILES:
        normalized_task_id = 1

    if not isinstance(task, Task):
        try:
            task = get_task(normalized_task_id)
        except Exception:
            task = get_task(1)
            normalized_task_id = 1

    return normalized_task_id, info, task


WEIGHT_PROFILES: Dict[int, Dict[str, float]] = {
    1: {
        "completion": 0.25,
        "uptime": 0.25,
        "sla": 0.20,
        "cost": 0.10,
        "stability": 0.12,
        "scaling_efficiency": 0.08,
    },
    2: {
        "completion": 0.20,
        "uptime": 0.20,
        "sla": 0.15,
        "cost": 0.20,
        "stability": 0.12,
        "scaling_efficiency": 0.13,
    },
    3: {
        "completion": 0.30,
        "uptime": 0.15,
        "sla": 0.10,
        "cost": 0.20,
        "stability": 0.15,
        "scaling_efficiency": 0.10,
    },
}


def _score_completion(info: Dict[str, Any], task: Task) -> float:
    steps_completed = max(1, _to_int(info.get("steps_completed", 1), 1))
    fraction = min(steps_completed / task.max_steps, 1.0)
    if info.get("termination_reason") == "success":
        return 1.0
    return round(fraction ** 0.75, 4)


def _score_uptime(info: Dict[str, Any], task: Task) -> float:
    uptime = max(0.0, min(100.0, _to_float(info.get("uptime_percentage", 0.0), 0.0)))
    if uptime >= 95.0:
        return 1.0
    if uptime >= 90.0:
        return round(0.80 + (uptime - 90.0) / 10.0 * 0.20, 4)
    if uptime >= 70.0:
        return round(0.40 + (uptime - 70.0) / 20.0 * 0.40, 4)
    return round(max(0.0, uptime / 70.0 * 0.40), 4)


def _score_sla(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, _to_int(info.get("steps_completed", 1), 1))
    sla_violations = max(0, _to_int(info.get("sla_violation_count", 0), 0))
    return round(max(0.0, 1.0 - (sla_violations / steps)), 4)


def _score_cost(info: Dict[str, Any], task: Task) -> float:
    total_cost = _to_float(info.get("total_cost", 0.0), 0.0)
    budget = task.budget
    hard_limit = budget * task.budget_failure_multiplier
    if total_cost <= budget:
        return 1.0
    if total_cost >= hard_limit:
        return 0.0
    overspend_fraction = (total_cost - budget) / (hard_limit - budget)
    return round(max(0.0, 1.0 - overspend_fraction), 4)


def _score_stability(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, _to_int(info.get("steps_completed", 1), 1))
    critical = max(0, _to_int(info.get("critical_violation_count", 0), 0))
    ratio = critical / steps
    if ratio == 0.0:
        return 1.0
    if ratio <= 0.20:
        return round(1.0 - (ratio / 0.20) * 0.50, 4)
    if ratio <= 0.50:
        return round(0.50 - ((ratio - 0.20) / 0.30) * 0.40, 4)
    return round(max(0.0, 0.10 - (ratio - 0.50) * 0.20), 4)


def _score_scaling_efficiency(info: Dict[str, Any], task: Task) -> float:
    tolerance = max(1, task.max_steps // 5)
    unnecessary = (
        max(0, _to_int(info.get("unnecessary_scaleups", 0), 0))
        + max(0, _to_int(info.get("unnecessary_scaledowns", 0), 0))
    )
    if unnecessary == 0:
        return 1.0
    ratio = unnecessary / tolerance
    if ratio <= 1.0:
        return round(1.0 - ratio * 0.50, 4)
    return round(max(0.0, 0.50 - (ratio - 1.0) * 0.50), 4)


def grade_episode(
    task_id: Any = None,
    info: Any = None,
    task: Task | None = None,
) -> float:
    try:
        tid, details_info, details_task = _normalize_scoring_inputs(task_id, info, task)
        result = grade_episode_report(task_id=tid, info=details_info, task=details_task)
        score = strict_unit_interval(_to_float(result.get("final_score", EPS), EPS))
        return float(score)
    except Exception:
        return float(EPS)


def grade_episode_report(
    task_id: Any = None,
    info: Any = None,
    task: Task | None = None,
) -> Dict[str, Any]:
    try:
        task_id, info, task = _normalize_scoring_inputs(task_id, info, task)
    except Exception:
        task_id, info, task = 1, {}, get_task(1)

    weights = WEIGHT_PROFILES[task_id]

    breakdown = {
        "completion": _score_completion(info, task),
        "uptime": _score_uptime(info, task),
        "sla": _score_sla(info, task),
        "cost": _score_cost(info, task),
        "stability": _score_stability(info, task),
        "scaling_efficiency": _score_scaling_efficiency(info, task),
    }

    breakdown = {k: strict_unit_interval(v) for k, v in breakdown.items()}

    weighted = {
        dim: round(score * weights[dim], 6)
        for dim, score in breakdown.items()
    }

    crash_penalty = 0.3 if info.get("termination_reason") != "success" else 0.0
    raw_total = sum(weighted.values()) - crash_penalty
    final_score = strict_rounded_unit_interval(raw_total)

    return {
        "task_id": task_id,
        "final_score": final_score,
        "breakdown": breakdown,
        "weighted": weighted,
        "weights": weights,
        "crash_penalty": crash_penalty,
        "termination": info.get("termination_reason", "unknown"),
        "steps": _to_int(info.get("steps_completed", 0), 0),
        "budget_used": f"{_to_float(info.get('total_cost', 0.0), 0.0):.2f} / {task.budget:.2f}",
        "uptime_pct": _to_float(info.get("uptime_percentage", 0.0), 0.0),
    }


def grade_episode_details(
    task_id: Any = None,
    info: Any = None,
    task: Task | None = None,
) -> Dict[str, Any]:
    """Return only strict in-range score fields for validator compatibility."""
    report = grade_episode_report(task_id=task_id, info=info, task=task)
    return {
        "final_score": report["final_score"],
        "breakdown": report["breakdown"],
        "weighted": report["weighted"],
        "weights": report["weights"],
    }


def aggregate_scores(scores: Dict[int, float]) -> float:
    task_weights = {1: 0.20, 2: 0.35, 3: 0.45}
    total = sum(scores.get(tid, 0.0) * w for tid, w in task_weights.items())
    return strict_rounded_unit_interval(total)


def print_grade(result: Dict[str, Any]) -> None:
    print("\n")
    print("  Task {} Score Report".format(result["task_id"]))
    print("")
    print(f"  Final Score   : {result['final_score']:.6f}")
    print(f"  Termination   : {result['termination']}")
    print(f"  Steps         : {result['steps']}")
    print(f"  Budget Used   : {result['budget_used']}")
    print(f"  Uptime        : {result['uptime_pct']}%")
    if result["crash_penalty"] > 0:
        print("  Crash Penalty : -0.3 (non-success)")

    print(f"\n  {'Dimension':<22} {'Raw':>8}  {'Weight':>7}  {'Weighted':>10}")
    for dim, raw in result["breakdown"].items():
        w = result["weights"][dim]
        wv = result["weighted"][dim]
        print(f"  {dim:<22} {raw:>8.6f}  {w:>7.2f}  {wv:>10.6f}")

    if result["crash_penalty"] > 0:
        print(f"  {'crash_penalty':<22} {'':>8}  {'':>7}  {-result['crash_penalty']:>10.6f}")

    print(f"  {'TOTAL':<22} {'':>8}  {'':>7}  {result['final_score']:>10.6f}")
    print("\n")