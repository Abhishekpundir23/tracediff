"""Render run results and diffs as terminal text or PR-comment markdown."""

from __future__ import annotations

from typing import Any

_CLASS_LABELS = {
    "regression": ("REGRESSION", "🔴"),
    "cost_regression": ("COST REGRESSION", "🟠"),
    "behavior_change": ("BEHAVIOR CHANGE", "🟡"),
    "improvement": ("IMPROVEMENT", "🟢"),
}


def render_run_text(results: dict[str, Any]) -> str:
    s = results["summary"]
    lines = [
        f"suite '{results['suite']}' (version {results['suite_hash']}) "
        f"split={results['split']} repeats={results['repeats']}",
        f"{s['fully_passing']}/{s['tasks']} tasks fully passing | "
        f"mean pass rate {s['mean_pass_rate']:.0%} | "
        f"total mean cost ${s['total_mean_cost_usd']:.4f}",
    ]
    for tid, agg in results["tasks"].items():
        flake = f" ({agg['sequence_variants']} trajectory variants)" if agg["sequence_variants"] > 1 else ""
        lines.append(
            f"  {'PASS' if agg['pass_rate'] == 1.0 else 'FAIL'} {tid}: "
            f"pass_rate={agg['pass_rate']:.0%} "
            f"tools={'->'.join(agg['modal_tool_sequence']) or '(none)'} "
            f"cost=${agg['mean_cost_usd']:.4f}+/-{agg['stdev_cost_usd']:.4f}{flake}"
        )
        if agg["pass_rate"] < 1.0:
            failing = next((r for r in agg["repeats"] if not r["passed"]), None)
            for f in (failing or {}).get("failures", [])[:3]:
                lines.append(f"      - {f}")
    return "\n".join(lines)


def _diff_lines(diff: dict[str, Any], markdown: bool) -> list[str]:
    s = diff["summary"]
    lines: list[str] = []
    changed = [e for e in diff["entries"] if e["classification"] != "unchanged"]

    for w in diff["warnings"]:
        lines.append(f"{'> ⚠️ ' if markdown else 'WARNING: '}{w}")
    if diff["warnings"]:
        lines.append("")

    headline = (
        f"{s['regressions']} regression(s), {s['cost_regressions']} cost regression(s), "
        f"{s['behavior_changes']} behavior change(s), {s['improvements']} improvement(s) "
        f"across {s['compared']} task(s)"
    )
    lines.append(f"**{headline}**" if markdown else headline)
    lines.append("")

    for added, label in ((diff["tasks_added"], "added"), (diff["tasks_removed"], "removed")):
        if added:
            lines.append(f"Tasks {label}: {', '.join(added)}")
    if diff["tasks_added"] or diff["tasks_removed"]:
        lines.append("")

    for entry in changed:
        label, emoji = _CLASS_LABELS[entry["classification"]]
        if markdown:
            lines.append(f"### {emoji} `{entry['task']}` — {label}")
        else:
            lines.append(f"[{label}] {entry['task']}")
        for change in entry["changes"]:
            lines.append(f"- {change}" if markdown else f"    - {change}")
        lines.append("")

    if not changed:
        lines.append("No behavioral differences detected.")
    return lines


def render_diff_text(diff: dict[str, Any]) -> str:
    return "\n".join(_diff_lines(diff, markdown=False))


def render_diff_markdown(diff: dict[str, Any]) -> str:
    header = (
        f"## tracediff: `{diff['suite']}` (version `{diff['suite_hash']}`, "
        f"split `{diff['split']}`)\n"
    )
    return header + "\n".join(_diff_lines(diff, markdown=True))
