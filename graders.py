from __future__ import annotations

import math
from typing import Any, Dict

from tasks import Task, get_task

# EPS keeps every score safely inside the open interval (0, 1).
# 0.1 gives a wide margin so even coarse external rounding can't hit 0 or 1.
EPS = 1e-1   # = 0.1  →  scores live in [0.1, 0.9]


# ── helpers ──────────────────────────────────────────────────────────────────

def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def strict_unit_interval(x: float) -> float:
    """Return x clamped to the OPEN interval (0, 1) with EPS margin."""
    v = float(x)
    if not math.isfinite(v):
        return EPS
    return max(EPS, min(1.0 - EPS, v))


def strict_rounded_unit_interval(x: float, digits: int = 6) -> float:
    return strict_unit_interval(round(float(x), digits))


# ── argument normaliser ───────────────────────────────────────────────────────

def _normalize_scoring_inputs(
    task_id: Any = None,
    info: Any = None,
    task: Any = None,
) -> tuple:
    """Accept the many call signatures openenv-core and validators may use."""
    # Handle: grade_episode(info_dict)  or  grade_episode(info_dict, task)
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

    # Resolve task_id
    if task_id is None:
        task_id = info.get("task_id")
    if task_id is None and isinstance(task, Task):
        task_id = task.task_id
    if task_id is None:
        task_id = 1

    task_id = _to_int(task_id, 1)
    if task_id not in WEIGHT_PROFILES:
        task_id = 1

    if not isinstance(task, Task):
        try:
            task = get_task(task_id)
        except Exception:
            task = get_task(1)
            task_id = 1

    return task_id, info, task


# ── weight profiles ───────────────────────────────────────────────────────────

WEIGHT_PROFILES: Dict[int, Dict[str, float]] = {
    1: {"completion": 0.25, "uptime": 0.25, "sla": 0.20,
        "cost": 0.10, "stability": 0.12, "scaling_efficiency": 0.08},
    2: {"completion": 0.20, "uptime": 0.20, "sla": 0.15,
        "cost": 0.20, "stability": 0.12, "scaling_efficiency": 0.13},
    3: {"completion": 0.30, "uptime": 0.15, "sla": 0.10,
        "cost": 0.20, "stability": 0.15, "scaling_efficiency": 0.10},
}


# ── dimension scorers (raw, before clamping) ──────────────────────────────────

def _score_completion(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, _to_int(info.get("steps_completed", 1), 1))
    fraction = min(steps / task.max_steps, 1.0)
    if info.get("termination_reason") == "success":
        return 0.98   # near-perfect but never 1.0
    return round(fraction ** 0.75 * 0.95, 6)


def _score_uptime(info: Dict[str, Any], task: Task) -> float:
    uptime = max(0.0, min(100.0, _to_float(info.get("uptime_percentage", 0.0))))
    if uptime >= 99.0:
        return 0.97
    if uptime >= 95.0:
        return round(0.85 + (uptime - 95.0) / 5.0 * 0.12, 6)
    if uptime >= 90.0:
        return round(0.75 + (uptime - 90.0) / 5.0 * 0.10, 6)
    if uptime >= 70.0:
        return round(0.40 + (uptime - 70.0) / 20.0 * 0.35, 6)
    return round(max(0.01, (uptime / 70.0) * 0.40), 6)


def _score_sla(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, _to_int(info.get("steps_completed", 1), 1))
    violations = max(0, _to_int(info.get("sla_violation_count", 0), 0))
    if violations == 0:
        return 0.97   # perfect SLA — never 1.0
    return round(max(0.01, 1.0 - (violations / steps)) * 0.97, 6)


def _score_cost(info: Dict[str, Any], task: Task) -> float:
    total_cost = _to_float(info.get("total_cost", 0.0))
    budget = task.budget
    hard_limit = budget * task.budget_failure_multiplier
    if total_cost <= 0.0:
        return 0.97
    if total_cost <= budget:
        return round(min(0.97, 1.0 - (total_cost / budget) * 0.20), 6)
    if total_cost >= hard_limit:
        return 0.02
    overspend = (total_cost - budget) / (hard_limit - budget)
    return round(max(0.02, (1.0 - overspend) * 0.50), 6)


def _score_stability(info: Dict[str, Any], task: Task) -> float:
    steps = max(1, _to_int(info.get("steps_completed", 1), 1))
    critical = max(0, _to_int(info.get("critical_violation_count", 0), 0))
    if critical == 0:
        return 0.97   # perfect stability — never 1.0
    ratio = critical / steps
    if ratio <= 0.05:
        return 0.85
    if ratio <= 0.20:
        return round(0.85 - (ratio - 0.05) / 0.15 * 0.35, 6)
    if ratio <= 0.50:
        return round(0.50 - (ratio - 0.20) / 0.30 * 0.40, 6)
    return round(max(0.02, 0.10 - (ratio - 0.50) * 0.16), 6)


def _score_scaling_efficiency(info: Dict[str, Any], task: Task) -> float:
    tolerance = max(1, task.max_steps // 5)
    unnecessary = (
        max(0, _to_int(info.get("unnecessary_scaleups", 0), 0))
        + max(0, _to_int(info.get("unnecessary_scaledowns", 0), 0))
    )
    if unnecessary == 0:
        return 0.97   # perfect efficiency — never 1.0
    ratio = unnecessary / tolerance
    if ratio <= 1.0:
        return round(max(0.02, 0.97 - ratio * 0.45), 6)
    return round(max(0.02, 0.52 - (ratio - 1.0) * 0.40), 6)


# ── public API ────────────────────────────────────────────────────────────────

def grade_episode_report(
    task_id: Any = None,
    info: Any = None,
    task: Any = None,
) -> Dict[str, Any]:
    """Full scoring report (dict). Used by baseline.py and inference.py."""
    try:
        task_id, info, task = _normalize_scoring_inputs(task_id, info, task)
    except Exception:
        task_id, info, task = 1, {}, get_task(1)

    weights = WEIGHT_PROFILES[task_id]

    # Raw scores (may be near 0 or 1 but never exactly)
    raw = {
        "completion":         _score_completion(info, task),
        "uptime":             _score_uptime(info, task),
        "sla":                _score_sla(info, task),
        "cost":               _score_cost(info, task),
        "stability":          _score_stability(info, task),
        "scaling_efficiency": _score_scaling_efficiency(info, task),
    }

    # Clamp every dimension into (0, 1)
    breakdown = {k: strict_unit_interval(v) for k, v in raw.items()}

    weighted = {dim: round(score * weights[dim], 6)
                for dim, score in breakdown.items()}

    crash_penalty = 0.3 if info.get("termination_reason") != "success" else 0.0
    raw_total = sum(weighted.values()) - crash_penalty

    final_score = strict_rounded_unit_interval(raw_total)

    return {
        "task_id":     task_id,
        "final_score": final_score,
        "breakdown":   breakdown,
        "weighted":    weighted,
        "weights":     weights,
        "crash_penalty": crash_penalty,
        "termination": info.get("termination_reason", "unknown"),
        "steps":       _to_int(info.get("steps_completed", 0), 0),
        "budget_used": (f"{_to_float(info.get('total_cost', 0.0)):.2f}"
                        f" / {task.budget:.2f}"),
        "uptime_pct":  _to_float(info.get("uptime_percentage", 0.0)),
    }


def grade_episode(
    task_id: Any = None,
    info: Any = None,
    task: Any = None,
) -> float:
    """Return a single float score strictly in (0, 1). Primary validator entry-point."""
    try:
        report = grade_episode_report(task_id=task_id, info=info, task=task)
        return float(strict_unit_interval(_to_float(report.get("final_score", EPS), EPS)))
    except Exception:
        return float(EPS)


def grade_episode_score(
    task_id: Any = None,
    info: Any = None,
    task: Any = None,
) -> float:
    """Alias for grade_episode. openenv.yaml points here."""
    return grade_episode(task_id=task_id, info=info, task=task)


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
        w  = result["weights"][dim]
        wv = result["weighted"][dim]
        print(f"  {dim:<22} {raw:>8.6f}  {w:>7.2f}  {wv:>10.6f}")
    if result["crash_penalty"] > 0:
        print(f"  {'crash_penalty':<22} {'':>8}  {'':>7}  {-result['crash_penalty']:>10.6f}")
    print(f"  {'TOTAL':<22} {'':>8}  {'':>7}  {result['final_score']:>10.6f}")
    print("\n")


# ── self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from tasks import ALL_TASKS
    print("Running graders self-test …")
    ok = True
    for task in ALL_TASKS:
        for scenario, info in [
            ("success",  {"steps_completed": task.max_steps,
                          "termination_reason": "success",
                          "total_cost": task.budget * 0.75,
                          "sla_violation_count": 0,
                          "critical_violation_count": 0,
                          "uptime_percentage": 100.0,
                          "unnecessary_scaleups": 0,
                          "unnecessary_scaledowns": 0}),
            ("budget_exceeded", {"steps_completed": task.max_steps // 2,
                                  "termination_reason": "budget_exceeded",
                                  "total_cost": task.budget * 1.08,
                                  "sla_violation_count": 3,
                                  "critical_violation_count": 2,
                                  "uptime_percentage": 80.0,
                                  "unnecessary_scaleups": 5,
                                  "unnecessary_scaledowns": 1}),
            ("crash",    {"steps_completed": 1,
                          "termination_reason": "critical_overload",
                          "total_cost": 0.5,
                          "sla_violation_count": 1,
                          "critical_violation_count": 1,
                          "uptime_percentage": 0.0,
                          "unnecessary_scaleups": 0,
                          "unnecessary_scaledowns": 0}),
            ("empty",    {}),
        ]:
            score = grade_episode_score(task.task_id, info, task)
            passed = 0.0 < score < 1.0
            if not passed:
                ok = False
            print(f"  Task {task.task_id} {scenario:<18}: {score:.6f}  {'OK' if passed else 'FAIL !!!'}")
    print("Self-test:", "PASSED" if ok else "FAILED")