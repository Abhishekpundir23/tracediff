"""Runner: execute an agent entrypoint over a suite split, with repeats.

The agent contract is one Python callable: it receives the task input and
returns a trace in any supported shape (Trace, serialized dict, or an
OpenAI-style message list). BYOK by construction - agentdiff never calls a
model provider itself.

Repeats are first-class because agents are stochastic: a single run is a
sample, not a measurement. Per-task results carry pass rate and cost spread
across repeats so diffs can distinguish drift from noise.
"""

from __future__ import annotations

import importlib
import statistics
import traceback as tb
from collections import Counter
from typing import Any, Callable

from agentdiff.scoring import evaluate
from agentdiff.suite import Suite, Task
from agentdiff.trace import normalize_trace


def load_agent(entrypoint: str) -> Callable[[Any], Any]:
    """Load 'package.module:function' as the agent callable."""
    if ":" not in entrypoint:
        raise ValueError(f"agent entrypoint must look like 'module:function', got '{entrypoint}'")
    module_name, func_name = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise ValueError(f"'{func_name}' in module '{module_name}' is not callable")
    return fn


def _run_once(agent_fn: Callable[[Any], Any], task: Task) -> dict[str, Any]:
    try:
        trace = normalize_trace(agent_fn(task.input))
    except Exception:
        return {
            "passed": False,
            "failures": [f"agent raised: {tb.format_exc(limit=3).strip()}"],
            "tool_sequence": [],
            "tool_calls": [],
            "metrics": {"steps": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0},
            "output": "",
            "error": True,
        }
    return evaluate(trace, task)


def _aggregate(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    costs = [r["metrics"]["cost_usd"] for r in repeats]
    steps = [r["metrics"]["steps"] for r in repeats]
    sequences = [tuple(r["tool_sequence"]) for r in repeats]
    modal_sequence = Counter(sequences).most_common(1)[0][0]
    # Tool calls from the first repeat that produced the modal sequence, so
    # diffs compare representative behavior rather than an outlier run.
    modal_calls = next(r["tool_calls"] for r in repeats if tuple(r["tool_sequence"]) == modal_sequence)
    return {
        "pass_rate": round(sum(1 for r in repeats if r["passed"]) / len(repeats), 4),
        "modal_tool_sequence": list(modal_sequence),
        "modal_tool_calls": modal_calls,
        "sequence_variants": len(set(sequences)),
        "mean_cost_usd": round(statistics.mean(costs), 6),
        "stdev_cost_usd": round(statistics.stdev(costs), 6) if len(costs) > 1 else 0.0,
        "mean_steps": round(statistics.mean(steps), 2),
        "repeats": repeats,
    }


def run_suite(
    suite: Suite,
    agent_fn: Callable[[Any], Any],
    split: str = "dev",
    repeats: int = 1,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from agentdiff import __version__

    tasks = suite.split(split)
    if not tasks:
        raise ValueError(f"split '{split}' of suite '{suite.name}' contains no tasks")

    results: dict[str, Any] = {}
    for task in tasks:
        runs = [_run_once(agent_fn, task) for _ in range(max(1, repeats))]
        results[task.id] = _aggregate(runs)
        if progress:
            agg = results[task.id]
            progress(f"  {task.id}: pass_rate={agg['pass_rate']:.0%} cost=${agg['mean_cost_usd']:.4f}")

    pass_rates = [r["pass_rate"] for r in results.values()]
    return {
        "agentdiff_version": __version__,
        "suite": suite.name,
        "suite_hash": suite.content_hash,
        "split": split,
        "repeats": max(1, repeats),
        "tasks": results,
        "summary": {
            "tasks": len(results),
            "fully_passing": sum(1 for r in results.values() if r["pass_rate"] == 1.0),
            "mean_pass_rate": round(statistics.mean(pass_rates), 4),
            "total_mean_cost_usd": round(sum(r["mean_cost_usd"] for r in results.values()), 6),
        },
    }
