"""Adapter tests using stub objects shaped like each framework's real types,
so tracediff needs no framework dependencies to test against."""

from types import SimpleNamespace as NS

from tracediff.adapters import from_claude_agent_sdk, from_langgraph, from_openai_agents


class TestLangGraph:
    def _messages(self):
        return [
            NS(type="human", content="What's the weather in Paris?"),
            NS(
                type="ai",
                content="",
                tool_calls=[{"name": "get_weather", "args": {"city": "Paris"}, "id": "tc1"}],
                usage_metadata={"input_tokens": 200, "output_tokens": 15},
            ),
            NS(type="tool", content="18C, sunny", tool_call_id="tc1"),
            NS(
                type="ai",
                content="It's 18C and sunny in Paris.",
                tool_calls=[],
                usage_metadata={"input_tokens": 250, "output_tokens": 20},
            ),
        ]

    def test_objects(self):
        trace = from_langgraph(self._messages())
        assert trace.tool_sequence() == ["get_weather"]
        assert trace.tool_calls()[0].args == {"city": "Paris"}
        assert trace.tool_calls()[0].result == "18C, sunny"
        assert trace.final_output == "It's 18C and sunny in Paris."
        assert trace.total_tokens_in == 450

    def test_state_dict_and_role_dicts(self):
        state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": "done",
                    "tool_calls": [{"name": "f", "args": {"x": 1}, "id": "a"}],
                }
            ]
        }
        trace = from_langgraph(state)
        assert trace.tool_sequence() == ["f"]
        assert trace.final_output == "done"

    def test_pricing_computes_cost(self):
        trace = from_langgraph(self._messages(), pricing=(3.0, 15.0))
        # 450 in @ $3/M + 35 out @ $15/M
        assert abs(trace.total_cost_usd - (450 / 1e6 * 3.0 + 35 / 1e6 * 15.0)) < 1e-9


class TestOpenAIAgents:
    def _result(self):
        return NS(
            new_items=[
                NS(
                    type="tool_call_item",
                    raw_item=NS(name="search", arguments='{"q": "tracediff"}', call_id="c1"),
                ),
                NS(
                    type="tool_call_output_item",
                    raw_item={"call_id": "c1", "output": "found"},
                    output="found",
                ),
                NS(
                    type="message_output_item",
                    raw_item=NS(content=[NS(text="Here you go.")]),
                ),
            ],
            final_output="Here you go.",
            raw_responses=[NS(usage=NS(input_tokens=300, output_tokens=50))],
        )

    def test_run_result(self):
        trace = from_openai_agents(self._result())
        assert trace.tool_sequence() == ["search"]
        assert trace.tool_calls()[0].args == {"q": "tracediff"}
        assert trace.tool_calls()[0].result == "found"
        assert trace.final_output == "Here you go."
        assert trace.total_tokens_out == 50

    def test_plain_item_list(self):
        trace = from_openai_agents(self._result().new_items)
        assert trace.tool_sequence() == ["search"]
        assert trace.final_output == "Here you go."


class TestClaudeAgentSDK:
    def _messages(self):
        return [
            NS(  # AssistantMessage
                content=[
                    NS(text="Let me check.", citations=None),
                    NS(id="tu1", name="Read", input={"file_path": "notes.md"}),
                ],
                model="claude-fable-5",
            ),
            NS(content=[NS(tool_use_id="tu1", content="file contents", is_error=False)]),
            NS(content=[NS(text="The notes say hello.", citations=None)], model="claude-fable-5"),
            NS(  # ResultMessage
                subtype="success",
                usage={"input_tokens": 1200, "output_tokens": 80},
                total_cost_usd=0.0042,
                result="The notes say hello.",
            ),
        ]

    def test_sdk_messages(self):
        trace = from_claude_agent_sdk(self._messages())
        assert trace.tool_sequence() == ["Read"]
        assert trace.tool_calls()[0].args == {"file_path": "notes.md"}
        assert trace.tool_calls()[0].result == "file contents"
        assert trace.final_output == "The notes say hello."
        assert trace.total_cost_usd == 0.0042
        assert trace.total_tokens_in == 1200

    def test_raw_api_dicts(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu1", "name": "search", "input": {"q": "x"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
        trace = from_claude_agent_sdk(messages)
        assert trace.tool_sequence() == ["search"]
        assert trace.tool_calls()[0].result == "ok"
        assert trace.final_output == "answer"
