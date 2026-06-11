import pytest

from agentdiff.trace import Trace, from_openai_messages, normalize_trace


def test_openai_adapter_extracts_tools_results_and_output():
    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"q": "x"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "found it"},
        {"role": "assistant", "content": "the answer"},
    ]
    trace = from_openai_messages(messages, usage={"prompt_tokens": 100, "completion_tokens": 20})
    assert trace.tool_sequence() == ["search"]
    assert trace.tool_calls()[0].args == {"q": "x"}
    assert trace.tool_calls()[0].result == "found it"
    assert trace.final_output == "the answer"
    assert trace.total_tokens_in == 100


def test_normalize_accepts_all_shapes():
    messages = [{"role": "assistant", "content": "hi"}]
    assert isinstance(normalize_trace(messages), Trace)
    assert isinstance(normalize_trace({"messages": messages}), Trace)
    t = normalize_trace(messages)
    assert isinstance(normalize_trace(t.to_dict()), Trace)
    assert isinstance(normalize_trace(t), Trace)
    with pytest.raises(TypeError):
        normalize_trace(42)


def test_roundtrip_serialization():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    trace = from_openai_messages(messages)
    restored = Trace.from_dict(trace.to_dict())
    assert restored.tool_sequence() == trace.tool_sequence()
    assert restored.final_output == trace.final_output


def test_malformed_tool_arguments_preserved_raw():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "not json {"}}
            ],
        }
    ]
    trace = from_openai_messages(messages)
    assert trace.tool_calls()[0].args == {"_raw": "not json {"}
