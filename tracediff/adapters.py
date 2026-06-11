"""Framework adapters: convert popular agent frameworks' outputs to a Trace.

All adapters are duck-typed - tracediff has no dependency on these
frameworks. Each accepts either the framework's native objects or their
dict-serialized equivalents, so traces loaded from JSON work too.

Cost: frameworks report tokens but rarely dollars. Pass
pricing=(input_usd_per_mtok, output_usd_per_mtok) to any adapter to compute
cost_usd from token counts; the Claude Agent SDK adapter uses the SDK's own
total_cost_usd when present.
"""

from __future__ import annotations

from typing import Any

from tracediff.trace import LLM, MESSAGE, TOOL_CALL, Step, Trace, _parse_tool_arguments


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cost(tokens_in: int, tokens_out: int, pricing: tuple[float, float] | None) -> float:
    if not pricing:
        return 0.0
    return tokens_in / 1_000_000 * pricing[0] + tokens_out / 1_000_000 * pricing[1]


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        parts = []
        for block in content:
            text = _get(block, "text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content) if content is not None else ""


def from_langgraph(
    state_or_messages: Any, pricing: tuple[float, float] | None = None
) -> Trace:
    """Convert LangGraph/LangChain message history to a Trace.

    Accepts the final graph state (a dict with a 'messages' key) or the
    message list itself. Handles AIMessage.tool_calls
    ({'name', 'args', 'id'} entries), ToolMessage results matched by
    tool_call_id, and usage_metadata token counts.
    """
    messages = _get(state_or_messages, "messages", state_or_messages)
    steps: list[Step] = []
    final_output = ""
    by_call_id: dict[str, Step] = {}

    for msg in messages:
        mtype = _get(msg, "type") or _get(msg, "role") or ""
        if mtype in ("ai", "assistant"):
            text = _text_of(_get(msg, "content"))
            if text:
                steps.append(Step(kind=MESSAGE, name="assistant", args={"content": text}))
                final_output = text
            for tc in _get(msg, "tool_calls") or []:
                raw_args = _get(tc, "args", _get(tc, "arguments"))
                step = Step(
                    kind=TOOL_CALL,
                    name=str(_get(tc, "name", "")),
                    args=_parse_tool_arguments(raw_args),
                )
                steps.append(step)
                call_id = _get(tc, "id")
                if call_id:
                    by_call_id[str(call_id)] = step
            usage = _get(msg, "usage_metadata")
            if usage:
                tokens_in = int(_get(usage, "input_tokens") or 0)
                tokens_out = int(_get(usage, "output_tokens") or 0)
                steps.append(
                    Step(
                        kind=LLM,
                        name="llm",
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        cost_usd=_cost(tokens_in, tokens_out, pricing),
                    )
                )
        elif mtype == "tool":
            call_id = _get(msg, "tool_call_id")
            if call_id and str(call_id) in by_call_id:
                by_call_id[str(call_id)].result = _text_of(_get(msg, "content"))

    return Trace(steps=steps, final_output=final_output)


def from_openai_agents(
    run_result: Any, pricing: tuple[float, float] | None = None
) -> Trace:
    """Convert an OpenAI Agents SDK RunResult (or its new_items list) to a Trace.

    Walks run items by their 'type' tag: tool_call_item / tool_call_output_item
    / message_output_item. Token usage is summed across raw_responses.
    """
    items = _get(run_result, "new_items", run_result if isinstance(run_result, list) else [])
    steps: list[Step] = []
    final_output = ""
    by_call_id: dict[str, Step] = {}

    for item in items:
        itype = _get(item, "type", "")
        raw = _get(item, "raw_item", item)
        if itype == "tool_call_item":
            step = Step(
                kind=TOOL_CALL,
                name=str(_get(raw, "name", "")),
                args=_parse_tool_arguments(_get(raw, "arguments")),
            )
            steps.append(step)
            call_id = _get(raw, "call_id") or _get(raw, "id")
            if call_id:
                by_call_id[str(call_id)] = step
        elif itype == "tool_call_output_item":
            call_id = _get(raw, "call_id")
            output = _get(item, "output", _get(raw, "output"))
            if call_id and str(call_id) in by_call_id:
                by_call_id[str(call_id)].result = output
        elif itype == "message_output_item":
            text = _text_of(_get(raw, "content"))
            if text:
                steps.append(Step(kind=MESSAGE, name="assistant", args={"content": text}))
                final_output = text

    explicit_final = _get(run_result, "final_output")
    if explicit_final is not None:
        final_output = str(explicit_final)

    tokens_in = tokens_out = 0
    for response in _get(run_result, "raw_responses") or []:
        usage = _get(response, "usage")
        if usage:
            tokens_in += int(_get(usage, "input_tokens") or 0)
            tokens_out += int(_get(usage, "output_tokens") or 0)
    if tokens_in or tokens_out:
        steps.append(
            Step(
                kind=LLM,
                name="llm",
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_usd=_cost(tokens_in, tokens_out, pricing),
            )
        )

    return Trace(steps=steps, final_output=final_output)


def from_claude_agent_sdk(
    messages: list[Any], pricing: tuple[float, float] | None = None
) -> Trace:
    """Convert collected Claude Agent SDK messages to a Trace.

    Pass the list of messages received from query()/the client stream.
    AssistantMessage content blocks become steps (ToolUseBlock -> tool_call,
    TextBlock -> message); ToolResultBlocks attach results by tool_use_id;
    the final ResultMessage supplies usage, the SDK-computed total_cost_usd,
    and the canonical result text. Raw Anthropic API dicts (content blocks
    with 'type' keys) work too.
    """
    steps: list[Step] = []
    final_output = ""
    by_tool_use_id: dict[str, Step] = {}

    for msg in messages:
        content = _get(msg, "content")
        if isinstance(content, list):
            for block in content:
                name, tool_input = _get(block, "name"), _get(block, "input")
                tool_use_id = _get(block, "tool_use_id")
                text = _get(block, "text")
                if name is not None and tool_input is not None:  # ToolUseBlock
                    step = Step(
                        kind=TOOL_CALL,
                        name=str(name),
                        args=_parse_tool_arguments(tool_input),
                    )
                    steps.append(step)
                    block_id = _get(block, "id")
                    if block_id:
                        by_tool_use_id[str(block_id)] = step
                elif tool_use_id is not None:  # ToolResultBlock
                    if str(tool_use_id) in by_tool_use_id:
                        by_tool_use_id[str(tool_use_id)].result = _text_of(
                            _get(block, "content")
                        )
                elif isinstance(text, str) and text:  # TextBlock
                    steps.append(Step(kind=MESSAGE, name="assistant", args={"content": text}))
                    final_output = text

        # ResultMessage: usage + SDK-computed cost + canonical result text
        usage = _get(msg, "usage")
        total_cost = _get(msg, "total_cost_usd")
        if usage or total_cost is not None:
            tokens_in = int(_get(usage, "input_tokens") or 0) if usage else 0
            tokens_out = int(_get(usage, "output_tokens") or 0) if usage else 0
            steps.append(
                Step(
                    kind=LLM,
                    name="llm",
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=float(total_cost)
                    if total_cost is not None
                    else _cost(tokens_in, tokens_out, pricing),
                )
            )
            result_text = _get(msg, "result")
            if isinstance(result_text, str) and result_text:
                final_output = result_text

    return Trace(steps=steps, final_output=final_output)
