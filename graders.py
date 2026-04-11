from __future__ import annotations

from typing import Any, Dict

from tasks import Task, get_task

SCORE_MIN = 0.05
SCORE_MAX = 0.95


def _clamp(x: float) -> float:
    v = float(x)
    if v <= SCORE_MIN:
        return SCORE_MIN
    if v >= SCORE_MAX:
        return SCORE_MAX
    return v


def strict_unit_interval(x: float) -> float:
    return _clamp(x)


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
    steps_completed = max(1, int(info.get("steps_completed", 1)))
    fraction = min(float(steps_completed) / float(task.max_steps), 1.0)
    if info.get("termination_reason") == "success":
        return 0.95
    return _clamp(fraction ** 0.75)


def _score_uptime(info: Dict[str, Any], task: Task) -> float:
    uptime = max(0.0, min(100.0, float(info.get("uptime_percentage", 0.0))))
    if uptime >= 95.0:
        score = 0.93
    elif uptime >= 90.0:
        score = 0.80 + (uptime - 90.0) / 10.0 * 0.13
    elif uptime >= 70.0:
        score = 0.40 + (uptime - 70.0) / 20.0 * 0.40
    else:
        score = (uptime / 70.0) * 0.40
    return _clamp(score)


def _score_sla(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, int(info.get("steps_completed", 1)))
    sla_violations = int(info.get("sla_violation_count", 0))
    score = (1.0 - float(sla_violations) / float(steps)) * 0.90
    return _clamp(score)


def _score_cost(info: Dict[str, Any], task: Task) -> float:
    total_cost = float(info.get("total_cost", 0.0))
    budget = float(task.budget)
    hard_limit = budget * float(task.budget_failure_multiplier)

    if total_cost <= budget:
        fraction_used = total_cost / budget if budget > 0 else 0.0
        score = 0.90 - fraction_used * 0.20
        return _clamp(score)

    if total_cost >= hard_limit:
        return SCORE_MIN

    overspend = (total_cost - budget) / (hard_limit - budget)
    return _clamp((1.0 - overspend) * 0.50)


def _score_stability(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, int(info.get("steps_completed", 1)))
    critical = int(info.get("critical_violation_count", 0))

    if critical == 0:
        return 0.95

    ratio = float(critical) / float(steps)
    if ratio <= 0.20:
        score = 0.90 - (ratio / 0.20) * 0.40
    elif ratio <= 0.50:
        score = 0.50 - ((ratio - 0.20) / 0.30) * 0.35
    else:
        score = max(0.06, 0.15 - (ratio - 0.50) * 0.18)

    return _clamp(score)


def _score_scaling_efficiency(info: Dict[str, Any], task: Task) -> float:
    tolerance = max(1, task.max_steps // 5)
    unnecessary = int(info.get("unnecessary_scaleups", 0)) + int(info.get("unnecessary_scaledowns", 0))

    if unnecessary == 0:
        return 0.95

    ratio = float(unnecessary) / float(tolerance)
    if ratio <= 1.0:
        score = 0.90 - ratio * 0.40
    else:
        score = max(0.06, 0.50 - (ratio - 1.0) * 0.40)

    return _clamp(score)


def grade_episode_details(
    task_id: int,
    info: Dict[str, Any],
    task: Task | None = None,
) -> Dict[str, Any]:
    if task is None:
        task = get_task(task_id)

    weights = WEIGHT_PROFILES[task_id]

    breakdown = {
        "completion": _score_completion(info, task),
        "uptime": _score_uptime(info, task),
        "sla": _score_sla(info, task),
        "cost": _score_cost(info, task),
        "stability": _score_stability(info, task),
        "scaling_efficiency": _score_scaling_efficiency(info, task),
    }

    breakdown = {k: _clamp(v) for k, v in breakdown.items()}

    weighted = {
        dim: float(breakdown[dim]) * float(weights[dim])
        for dim in breakdown
    }

    crash_penalty = 0.3 if info.get("termination_reason") != "success" else 0.0
    raw_total = sum(weighted.values()) - crash_penalty
    # HARD SAFE CLAMP (FINAL FIX)
    if raw_total <= 0.05:
        final_score = 0.05
    elif raw_total >= 0.95:
        final_score = 0.95
    else:
        final_score = raw_total

    return {
        "task_id": task_id,
        "final_score": final_score,
        "breakdown": breakdown,
        "weighted": weighted,
        "weights": weights,
        "crash_penalty": crash_penalty,
        "termination": info.get("termination_reason", "unknown"),
        "steps": int(info.get("steps_completed", 0)),
        "budget_used": f"{float(info.get('total_cost', 0.0)):.2f} / {task.budget:.2f}",
        "uptime_pct": float(info.get("uptime_percentage", 0.0)),
    }


def grade_episode(
    task_id: int,
    info: Dict[str, Any],
    task: Task | None = None,
) -> float:
    return float(grade_episode_details(task_id, info, task)["final_score"])


def aggregate_scores(scores: Dict[int, float]) -> float:
    task_weights = {1: 0.20, 2: 0.35, 3: 0.45}
    total = sum(float(scores.get(tid, SCORE_MIN)) * w for tid, w in task_weights.items())
    if total <= 0.05:
        return 0.05
    elif total >= 0.95:
        return 0.95
    return total


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