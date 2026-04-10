from __future__ import annotations

from typing import Any, Dict

from tasks import Task, get_task

EPS = 1e-6


def strict_unit_interval(x: float) -> float:
    return max(EPS, min(1.0 - EPS, float(x)))


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
    steps_completed = max(1, info.get("steps_completed", 1))
    fraction = min(steps_completed / task.max_steps, 1.0)
    if info.get("termination_reason") == "success":
        # Return just below 1.0 to stay strictly within (0, 1)
        return 1.0 - EPS
    return round(max(EPS, fraction ** 0.75), 4)


def _score_uptime(info: Dict[str, Any], task: Task) -> float:
    uptime = max(0.0, min(100.0, info.get("uptime_percentage", 0.0)))
    if uptime >= 95.0:
        return 1.0 - EPS
    if uptime >= 90.0:
        return round(0.80 + (uptime - 90.0) / 10.0 * 0.20, 4)
    if uptime >= 70.0:
        return round(0.40 + (uptime - 70.0) / 20.0 * 0.40, 4)
    return round(max(EPS, uptime / 70.0 * 0.40), 4)


def _score_sla(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, info.get("steps_completed", 1))
    sla_violations = info.get("sla_violation_count", 0)
    raw = max(0.0, 1.0 - (sla_violations / steps))
    return round(max(EPS, min(1.0 - EPS, raw)), 4)


def _score_cost(info: Dict[str, Any], task: Task) -> float:
    total_cost = info.get("total_cost", 0.0)
    budget = task.budget
    hard_limit = budget * task.budget_failure_multiplier
    if total_cost <= budget:
        return 1.0 - EPS
    if total_cost >= hard_limit:
        return EPS
    overspend_fraction = (total_cost - budget) / (hard_limit - budget)
    return round(max(EPS, min(1.0 - EPS, 1.0 - overspend_fraction)), 4)


def _score_stability(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, info.get("steps_completed", 1))
    critical = info.get("critical_violation_count", 0)
    ratio = critical / steps
    if ratio == 0.0:
        return 1.0 - EPS
    if ratio <= 0.20:
        return round(max(EPS, 1.0 - (ratio / 0.20) * 0.50), 4)
    if ratio <= 0.50:
        return round(max(EPS, 0.50 - ((ratio - 0.20) / 0.30) * 0.40), 4)
    return round(max(EPS, 0.10 - (ratio - 0.50) * 0.20), 4)


def _score_scaling_efficiency(info: Dict[str, Any], task: Task) -> float:
    tolerance = max(1, task.max_steps // 5)
    unnecessary = (
        info.get("unnecessary_scaleups", 0)
        + info.get("unnecessary_scaledowns", 0)
    )
    if unnecessary == 0:
        return 1.0 - EPS
    ratio = unnecessary / tolerance
    if ratio <= 1.0:
        return round(max(EPS, 1.0 - ratio * 0.50), 4)
    return round(max(EPS, 0.50 - (ratio - 1.0) * 0.50), 4)


def grade_episode(
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

    # Apply strict_unit_interval to every dimension
    breakdown = {k: strict_unit_interval(v) for k, v in breakdown.items()}

    weighted = {
        dim: round(score * weights[dim], 6)
        for dim, score in breakdown.items()
    }

    crash_penalty = 0.3 if info.get("termination_reason") != "success" else 0.0
    raw_total = sum(weighted.values()) - crash_penalty

    # Ensure final score is strictly within (0, 1)
    final_score = round(strict_unit_interval(raw_total), 6)

    return {
        "task_id": task_id,
        "final_score": final_score,
        "breakdown": breakdown,
        "weighted": weighted,
        "weights": weights,
        "crash_penalty": crash_penalty,
        "termination": info.get("termination_reason", "unknown"),
        "steps": info.get("steps_completed", 0),
        "budget_used": f"{info.get('total_cost', 0.0):.2f} / {task.budget:.2f}",
        "uptime_pct": info.get("uptime_percentage", 0.0),
    }


def aggregate_scores(scores: Dict[int, float]) -> float:
    task_weights = {1: 0.20, 2: 0.35, 3: 0.45}
    total = sum(scores.get(tid, 0.0) * w for tid, w in task_weights.items())
    return round(total, 6)


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