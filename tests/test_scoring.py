from agentdiff.scoring import args_contain, evaluate, match_tool_sequence
from agentdiff.suite import Task
from agentdiff.trace import Step, Trace


def _trace(tools, output="ok", cost=0.0, extra_steps=0):
    steps = [Step(kind="tool_call", name=n, args=a) for n, a in tools]
    steps += [Step(kind="message", name="assistant") for _ in range(extra_steps)]
    if cost:
        steps.append(Step(kind="llm", name="model", cost_usd=cost))
    return Trace(steps=steps, final_output=output)


class TestMatchToolSequence:
    def test_strict(self):
        assert match_tool_sequence(["a", "b"], ["a", "b"], "strict")
        assert not match_tool_sequence(["b", "a"], ["a", "b"], "strict")
        assert not match_tool_sequence(["a", "b", "c"], ["a", "b"], "strict")

    def test_unordered(self):
        assert match_tool_sequence(["b", "a"], ["a", "b"], "unordered")
        assert not match_tool_sequence(["a", "a"], ["a"], "unordered")

    def test_subset_preserves_order(self):
        assert match_tool_sequence(["x", "a", "y", "b"], ["a", "b"], "subset")
        assert not match_tool_sequence(["b", "a"], ["a", "b"], "subset")
        assert not match_tool_sequence(["a"], ["a", "b"], "subset")


class TestArgsContain:
    def test_subset_dict(self):
        assert args_contain({"q": "x", "limit": 5}, {"q": "x"})
        assert not args_contain({"q": "x"}, {"q": "y"})
        assert not args_contain({}, {"q": "x"})

    def test_nested(self):
        assert args_contain({"filter": {"lang": "en", "n": 3}}, {"filter": {"lang": "en"}})


class TestEvaluate:
    def test_passes_when_all_expectations_hold(self):
        task = Task(
            id="t",
            input=None,
            expect={"tools": ["search"], "mode": "strict", "args": {"search": {"q": "x"}}},
            checks=[{"type": "output_contains", "value": "ok"}],
        )
        result = evaluate(_trace([("search", {"q": "x"})]), task)
        assert result["passed"], result["failures"]

    def test_wrong_tool_order_fails_strict(self):
        task = Task(id="t", input=None, expect={"tools": ["a", "b"], "mode": "strict"})
        result = evaluate(_trace([("b", {}), ("a", {})]), task)
        assert not result["passed"]
        assert "strict-match" in result["failures"][0]

    def test_cost_budget(self):
        task = Task(id="t", input=None, expect={"max_cost_usd": 0.001})
        result = evaluate(_trace([], cost=0.002), task)
        assert not result["passed"]
        assert "cost budget" in result["failures"][0]

    def test_step_budget(self):
        task = Task(id="t", input=None, expect={"max_steps": 2})
        result = evaluate(_trace([("a", {})], extra_steps=3), task)
        assert not result["passed"]

    def test_missing_expected_tool_call_with_args(self):
        task = Task(id="t", input=None, expect={"args": {"refund": {"amount": 1}}})
        result = evaluate(_trace([("lookup", {})]), task)
        assert not result["passed"]
        assert "refund" in result["failures"][0]

    def test_output_regex(self):
        task = Task(id="t", input=None, checks=[{"type": "output_regex", "value": r"\d{3}"}])
        assert evaluate(_trace([], output="answer: 391"), task)["passed"]
        assert not evaluate(_trace([], output="answer: none"), task)["passed"]
