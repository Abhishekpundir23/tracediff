"""Structural diff between two suite runs (baseline vs current).

This is the core product surface: not "the score moved", but what the agent
now *does differently* - per task, at the step/tool/argument level:

  - outcome changes   pass rate up or down
  - sequence changes  tools added, removed, replaced, or reordered, located
                      by position in the modal trajectory
  - argument drift    same tool, same position, different arguments
  - cost/step drift   mean cost or step count moved beyond tolerance

Each task is classified (regression | improvement | behavior_change |
cost_regression | unchanged) and the report carries machine-readable detail
plus human-readable change descriptions.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

DEFAULT_COST_RATIO_TOLERANCE = 1.25
DEFAULT_STEP_RATIO_TOLERANCE = 1.5

REGRESSION = "regression"
IMPROVEMENT = "improvement"
BEHAVIOR_CHANGE = "behavior_change"
COST_REGRESSION = "cost_regression"
UNCHANGED = "unchanged"


def _sequence_changes(base: list[str], cur: list[str]) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    matcher = SequenceMatcher(a=base, b=cur, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "delete":
            changes.append(
                {
                    "op": "removed",
                    "tools": base[i1:i2],
                    "position": i1,
                    "describe": f"no longer calls {' -> '.join(base[i1:i2])} (was at position {i1})",
                }
            )
        elif op == "insert":
            changes.append(
                {
                    "op": "added",
                    "tools": cur[j1:j2],
                    "position": j1,
                    "describe": f"now calls {' -> '.join(cur[j1:j2])} at position {j1}",
                }
            )
        elif op == "replace":
            changes.append(
                {
                    "op": "replaced",
                    "before": base[i1:i2],
                    "after": cur[j1:j2],
                    "position": i1,
                    "describe": (
                        f"calls {' -> '.join(cur[j1:j2])} instead of "
                        f"{' -> '.join(base[i1:i2])} at position {i1}"
                    ),
                }
            )
    return changes


def _arg_drift(
    base_calls: list[dict[str, Any]], cur_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compare arguments of calls aligned by (tool name, occurrence index)."""

    def by_occurrence(calls: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
        seen: dict[str, int] = {}
        out: dict[tuple[str, int], dict[str, Any]] = {}
        for c in calls:
            n = seen.get(c["name"], 0)
            seen[c["name"]] = n + 1
            out[(c["name"], n)] = c.get("args") or {}
        return out

    base_map, cur_map = by_occurrence(base_calls), by_occurrence(cur_calls)
    drifts: list[dict[str, Any]] = []
    for key in sorted(base_map.keys() & cur_map.keys()):
        b, c = base_map[key], cur_map[key]
        if b == c:
            continue
        changed = sorted(
            k for k in (b.keys() | c.keys()) if b.get(k, "<absent>") != c.get(k, "<absent>")
        )
        tool, occurrence = key
        label = tool if occurrence == 0 else f"{tool} (call #{occurrence + 1})"
        deltas = ", ".join(
            f"{k}: {b.get(k, '<absent>')!r} -> {c.get(k, '<absent>')!r}" for k in changed[:4]
        )
        if len(changed) > 4:
            deltas += f", +{len(changed) - 4} more"
        drifts.append(
            {
                "tool": tool,
                "occurrence": occurrence,
                "changed_keys": changed,
                "before": {k: b.get(k) for k in changed},
                "after": {k: c.get(k) for k in changed},
                "describe": f"{label} args drifted: {deltas}",
            }
        )
    return drifts


def _diff_task(
    task_id: str,
    base: dict[str, Any],
    cur: dict[str, Any],
    cost_tolerance: float,
    step_tolerance: float,
) -> dict[str, Any]:
    entry: dict[str, Any] = {"task": task_id, "changes": []}

    pass_delta = round(cur["pass_rate"] - base["pass_rate"], 4)
    if pass_delta != 0:
        entry["pass_rate"] = {"before": base["pass_rate"], "after": cur["pass_rate"]}
        entry["changes"].append(
            f"pass rate {base['pass_rate']:.0%} -> {cur['pass_rate']:.0%}"
        )

    seq_changes = _sequence_changes(base["modal_tool_sequence"], cur["modal_tool_sequence"])
    if seq_changes:
        entry["sequence_changes"] = seq_changes
        entry["changes"].extend(c["describe"] for c in seq_changes)

    drifts = _arg_drift(base.get("modal_tool_calls") or [], cur.get("modal_tool_calls") or [])
    if drifts:
        entry["arg_drift"] = drifts
        entry["changes"].extend(d["describe"] for d in drifts)

    base_cost, cur_cost = base["mean_cost_usd"], cur["mean_cost_usd"]
    cost_ratio = (cur_cost / base_cost) if base_cost > 0 else (None if cur_cost == 0 else float("inf"))
    cost_regressed = cost_ratio is not None and cost_ratio > cost_tolerance
    if cost_regressed:
        entry["cost"] = {"before": base_cost, "after": cur_cost, "ratio": round(cost_ratio, 2)}
        entry["changes"].append(
            f"mean cost ${base_cost:.4f} -> ${cur_cost:.4f} ({cost_ratio:.2f}x)"
        )

    base_steps, cur_steps = base["mean_steps"], cur["mean_steps"]
    if base_steps > 0 and cur_steps / base_steps > step_tolerance:
        entry["steps"] = {"before": base_steps, "after": cur_steps}
        entry["changes"].append(f"mean steps {base_steps:g} -> {cur_steps:g}")

    if pass_delta < 0:
        entry["classification"] = REGRESSION
    elif pass_delta > 0:
        entry["classification"] = IMPROVEMENT
    elif cost_regressed:
        entry["classification"] = COST_REGRESSION
    elif entry["changes"]:
        entry["classification"] = BEHAVIOR_CHANGE
    else:
        entry["classification"] = UNCHANGED
    return entry


def compare(
    baseline: dict[str, Any],
    current: dict[str, Any],
    cost_tolerance: float = DEFAULT_COST_RATIO_TOLERANCE,
    step_tolerance: float = DEFAULT_STEP_RATIO_TOLERANCE,
) -> dict[str, Any]:
    """Diff two run-results documents produced by runner.run_suite."""
    warnings: list[str] = []
    if baseline.get("suite_hash") != current.get("suite_hash"):
        warnings.append(
            f"suite versions differ (baseline {baseline.get('suite_hash')}, "
            f"current {current.get('suite_hash')}): task edits, not agent changes, "
            f"may explain differences"
        )
    if baseline.get("split") != current.get("split"):
        warnings.append(
            f"splits differ (baseline '{baseline.get('split')}', current '{current.get('split')}')"
        )

    base_tasks, cur_tasks = baseline["tasks"], current["tasks"]
    common = sorted(base_tasks.keys() & cur_tasks.keys())
    entries = [
        _diff_task(tid, base_tasks[tid], cur_tasks[tid], cost_tolerance, step_tolerance)
        for tid in common
    ]

    by_class: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_class.setdefault(e["classification"], []).append(e)

    counts = {k: len(v) for k, v in by_class.items()}
    return {
        "suite": current.get("suite"),
        "suite_hash": current.get("suite_hash"),
        "split": current.get("split"),
        "baseline_version": baseline.get("tracediff_version"),
        "warnings": warnings,
        "tasks_added": sorted(cur_tasks.keys() - base_tasks.keys()),
        "tasks_removed": sorted(base_tasks.keys() - cur_tasks.keys()),
        "entries": entries,
        "summary": {
            "compared": len(common),
            "regressions": counts.get(REGRESSION, 0),
            "improvements": counts.get(IMPROVEMENT, 0),
            "behavior_changes": counts.get(BEHAVIOR_CHANGE, 0),
            "cost_regressions": counts.get(COST_REGRESSION, 0),
            "unchanged": counts.get(UNCHANGED, 0),
        },
        "has_regressions": counts.get(REGRESSION, 0) > 0 or counts.get(COST_REGRESSION, 0) > 0,
    }
