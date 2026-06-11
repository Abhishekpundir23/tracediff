"""A deterministic offline demo agent (no API keys needed).

It fakes an LLM agent that answers tasks via tool calls, returning
OpenAI-style messages. Set TRACEDIFF_DEMO_VARIANT=b to simulate a "code
change" that alters the agent's behavior - the README walkthrough diffs
variant a (baseline) against variant b to show what tracediff catches:
an extra search loop, drifted read_file args, a cost increase, and one
outright wrong answer.
"""

import os


def _tc(call_id, name, **arguments):
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _tool_result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def run(task_input):
    variant = os.environ.get("TRACEDIFF_DEMO_VARIANT", "a")
    topic = task_input.get("topic") if isinstance(task_input, dict) else str(task_input)

    if topic == "capital-question":
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [_tc("c1", "search", query="capital of France")]},
            _tool_result("c1", "Paris is the capital of France."),
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]
        usage = {"prompt_tokens": 220, "completion_tokens": 40, "cost_usd": 0.0012}
        if variant == "b":
            # regression: redundant second search (a loop) doubles the cost
            messages.insert(1, _tool_result("c2", "Paris."))
            messages.insert(
                1,
                {"role": "assistant", "content": None, "tool_calls": [_tc("c2", "search", query="France capital city")]},
            )
            usage = {"prompt_tokens": 510, "completion_tokens": 55, "cost_usd": 0.0029}
        return {"messages": messages, "usage": usage}

    if topic == "refund":
        order_id = task_input.get("order_id", "A-100")
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [_tc("c1", "lookup_order", order_id=order_id)]},
            _tool_result("c1", f"order {order_id}: eligible for refund"),
            {"role": "assistant", "content": None, "tool_calls": [_tc("c2", "issue_refund", order_id=order_id, amount=49.99)]},
            _tool_result("c2", "refund issued"),
            {"role": "assistant", "content": f"Refund issued for order {order_id}."},
        ]
        usage = {"prompt_tokens": 340, "completion_tokens": 60, "cost_usd": 0.0018}
        if variant == "b":
            # arg drift: refunds the wrong amount (silently, output unchanged)
            messages[2] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tc("c2", "issue_refund", order_id=order_id, amount=499.99)],
            }
        return {"messages": messages, "usage": usage}

    if topic == "summarize-file":
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [_tc("c1", "read_file", path="notes/meeting.md")]},
            _tool_result("c1", "Q3 roadmap discussion..."),
            {"role": "assistant", "content": "Summary: the meeting covered the Q3 roadmap."},
        ]
        usage = {"prompt_tokens": 410, "completion_tokens": 90, "cost_usd": 0.0021}
        if variant == "b":
            # outcome regression: reads the wrong file and answers wrongly
            messages = [
                {"role": "assistant", "content": None, "tool_calls": [_tc("c1", "read_file", path="notes/standup.md")]},
                _tool_result("c1", "Daily standup notes..."),
                {"role": "assistant", "content": "Summary: the meeting covered daily standup blockers."},
            ]
        return {"messages": messages, "usage": usage}

    if topic == "math":
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [_tc("c1", "calculator", expression="17*23")]},
            _tool_result("c1", "391"),
            {"role": "assistant", "content": "17 * 23 = 391"},
        ]
        return {"messages": messages, "usage": {"prompt_tokens": 150, "completion_tokens": 25, "cost_usd": 0.0006}}

    return {"messages": [{"role": "assistant", "content": f"I don't know how to handle: {topic}"}]}
