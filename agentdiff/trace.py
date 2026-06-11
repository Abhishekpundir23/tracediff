"""Normalized trace model and ingestion adapters.

A Trace is the unit agentdiff scores and diffs: an ordered list of Steps an
agent took while solving one task, plus the final output it produced.

Agents can return any of:
  - a Trace object (built with the native API),
  - a dict with a "steps" list (the serialized Trace format),
  - a list of OpenAI-style chat messages (assistant messages with tool_calls,
    tool-role result messages) - the most widely emitted format across
    frameworks, so it is the default ingestion path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

TOOL_CALL = "tool_call"
LLM = "llm"
MESSAGE = "message"


@dataclass
class Step:
    kind: str  # "tool_call" | "llm" | "message"
    name: str  # tool name, model name, or message role
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        return cls(
            kind=d.get("kind", MESSAGE),
            name=d.get("name", ""),
            args=d.get("args") or {},
            result=d.get("result"),
            tokens_in=int(d.get("tokens_in") or 0),
            tokens_out=int(d.get("tokens_out") or 0),
            cost_usd=float(d.get("cost_usd") or 0.0),
        )


@dataclass
class Trace:
    steps: list[Step] = field(default_factory=list)
    final_output: str = ""

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def total_tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.steps)

    @property
    def total_tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.steps)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == TOOL_CALL]

    def tool_sequence(self) -> list[str]:
        return [s.name for s in self.tool_calls()]

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps], "final_output": self.final_output}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trace":
        return cls(
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            final_output=d.get("final_output", "") or "",
        )


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {}


def from_openai_messages(messages: list[dict[str, Any]], usage: dict[str, Any] | None = None) -> Trace:
    """Convert an OpenAI-style chat message list into a Trace.

    Assistant text content becomes message steps (the last one is the final
    output); assistant tool_calls become tool_call steps; tool-role messages
    attach their content as the result of the matching tool_call step.
    """
    steps: list[Step] = []
    final_output = ""
    by_call_id: dict[str, Step] = {}

    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            content = msg.get("content")
            if content:
                steps.append(Step(kind=MESSAGE, name="assistant", args={"content": content}))
                final_output = content if isinstance(content, str) else str(content)
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                step = Step(
                    kind=TOOL_CALL,
                    name=fn.get("name", tc.get("name", "")),
                    args=_parse_tool_arguments(fn.get("arguments", tc.get("arguments"))),
                )
                steps.append(step)
                call_id = tc.get("id")
                if call_id:
                    by_call_id[call_id] = step
        elif role == "tool":
            call_id = msg.get("tool_call_id")
            if call_id and call_id in by_call_id:
                by_call_id[call_id].result = msg.get("content")
            else:
                steps.append(
                    Step(kind=MESSAGE, name="tool", args={"content": msg.get("content")})
                )
        # system/user messages are task setup, not agent behavior: skipped.

    if usage:
        steps.append(
            Step(
                kind=LLM,
                name=str(usage.get("model", "llm")),
                tokens_in=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                tokens_out=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                cost_usd=float(usage.get("cost_usd") or 0.0),
            )
        )

    return Trace(steps=steps, final_output=final_output)


def normalize_trace(obj: Any) -> Trace:
    """Accept the supported agent return shapes and produce a Trace."""
    if isinstance(obj, Trace):
        return obj
    if isinstance(obj, dict):
        if "steps" in obj:
            return Trace.from_dict(obj)
        if "messages" in obj:
            return from_openai_messages(obj["messages"], usage=obj.get("usage"))
        raise TypeError("dict trace must contain a 'steps' or 'messages' key")
    if isinstance(obj, list):
        return from_openai_messages(obj)
    raise TypeError(f"unsupported trace type: {type(obj).__name__}")
