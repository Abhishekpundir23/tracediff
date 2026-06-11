"""Scoring: evaluate one trace against one task's expectations.

A task passes when every declared expectation holds:
  expect.tools / expect.mode   - tool sequence matches (strict|unordered|subset)
  expect.args                  - for each named tool, at least one call's
                                 arguments contain the expected key/values
  expect.max_steps             - step budget
  expect.max_tool_calls        - tool-call budget
  expect.max_cost_usd          - cost budget
  checks[]                     - assertions on the final output

Budgets are first-class: an agent that answers correctly while doubling its
cost or looping through extra steps is a finding, not a pass.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from agentdiff.suite import Task
from agentdiff.trace import Trace

MATCH_MODES = ("strict", "unordered", "subset")


def match_tool_sequence(actual: list[str], expected: list[str], mode: str = "subset") -> bool:
    if mode == "strict":
        return actual == expected
    if mode == "unordered":
        return Counter(actual) == Counter(expected)
    if mode == "subset":
        it = iter(actual)
        return all(name in it for name in expected)
    raise ValueError(f"unknown match mode '{mode}' (expected one of {MATCH_MODES})")


def args_contain(actual: Any, expected: Any) -> bool:
    """True when expected is recursively contained in actual."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and args_contain(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and all(
            any(args_contain(a, e) for a in actual) for e in expected
        )
    return actual == expected


def _check_output(output: str, check: dict[str, Any]) -> tuple[bool, str]:
    ctype = check.get("type", "")
    value = check.get("value", "")
    if ctype == "output_contains":
        return (str(value).lower() in output.lower(), f"output contains {value!r}")
    if ctype == "output_not_contains":
        return (str(value).lower() not in output.lower(), f"output does not contain {value!r}")
    if ctype == "output_equals":
        return (output.strip() == str(value).strip(), f"output equals {value!r}")
    if ctype == "output_regex":
        return (re.search(str(value), output) is not None, f"output matches /{value}/")
    raise ValueError(f"unknown check type '{ctype}'")


def evaluate(trace: Trace, task: Task) -> dict[str, Any]:
    """Score a trace against a task. Returns a JSON-serializable evaluation."""
    failures: list[str] = []
    expect = task.expect

    if "tools" in expect:
        mode = expect.get("mode", "subset")
        if not match_tool_sequence(trace.tool_sequence(), list(expect["tools"]), mode):
            failures.append(
                f"tool sequence {trace.tool_sequence()} does not {mode}-match "
                f"expected {list(expect['tools'])}"
            )

    for tool_name, expected_args in (expect.get("args") or {}).items():
        calls = [s for s in trace.tool_calls() if s.name == tool_name]
        if not calls:
            failures.append(f"expected a call to '{tool_name}' but none was made")
        elif not any(args_contain(c.args, expected_args) for c in calls):
            failures.append(
                f"no call to '{tool_name}' had expected args {expected_args} "
                f"(saw {[c.args for c in calls]})"
            )

    if "max_steps" in expect and trace.step_count > int(expect["max_steps"]):
        failures.append(f"step budget exceeded: {trace.step_count} > {expect['max_steps']}")
    if "max_tool_calls" in expect and len(trace.tool_calls()) > int(expect["max_tool_calls"]):
        failures.append(
            f"tool-call budget exceeded: {len(trace.tool_calls())} > {expect['max_tool_calls']}"
        )
    if "max_cost_usd" in expect and trace.total_cost_usd > float(expect["max_cost_usd"]):
        failures.append(
            f"cost budget exceeded: ${trace.total_cost_usd:.4f} > ${float(expect['max_cost_usd']):.4f}"
        )

    for check in task.checks:
        ok, desc = _check_output(trace.final_output or "", check)
        if not ok:
            failures.append(f"check failed: {desc}")

    return {
        "passed": not failures,
        "failures": failures,
        "tool_sequence": trace.tool_sequence(),
        "tool_calls": [{"name": s.name, "args": s.args} for s in trace.tool_calls()],
        "metrics": {
            "steps": trace.step_count,
            "tool_calls": len(trace.tool_calls()),
            "tokens_in": trace.total_tokens_in,
            "tokens_out": trace.total_tokens_out,
            "cost_usd": round(trace.total_cost_usd, 6),
        },
        "output": (trace.final_output or "")[:2000],
    }
