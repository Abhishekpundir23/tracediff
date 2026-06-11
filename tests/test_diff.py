from agentdiff.diff import compare


def _task_result(
    pass_rate=1.0, tools=None, calls=None, cost=0.001, steps=3.0, variants=1
):
    tools = tools or ["search"]
    calls = calls if calls is not None else [{"name": n, "args": {}} for n in tools]
    return {
        "pass_rate": pass_rate,
        "modal_tool_sequence": tools,
        "modal_tool_calls": calls,
        "sequence_variants": variants,
        "mean_cost_usd": cost,
        "stdev_cost_usd": 0.0,
        "mean_steps": steps,
        "repeats": [],
    }


def _results(tasks, suite_hash="abc123"):
    return {
        "agentdiff_version": "0.1.0",
        "suite": "s",
        "suite_hash": suite_hash,
        "split": "dev",
        "repeats": 1,
        "tasks": tasks,
        "summary": {},
    }


def test_unchanged():
    base = _results({"t1": _task_result()})
    diff = compare(base, _results({"t1": _task_result()}))
    assert diff["summary"]["unchanged"] == 1
    assert not diff["has_regressions"]


def test_pass_rate_drop_is_regression():
    base = _results({"t1": _task_result(pass_rate=1.0)})
    cur = _results({"t1": _task_result(pass_rate=0.5)})
    diff = compare(base, cur)
    assert diff["summary"]["regressions"] == 1
    assert diff["has_regressions"]


def test_added_tool_detected_with_position():
    base = _results({"t1": _task_result(tools=["search", "answer"])})
    cur = _results({"t1": _task_result(tools=["search", "search", "answer"])})
    diff = compare(base, cur)
    entry = diff["entries"][0]
    assert entry["classification"] == "behavior_change"
    ops = entry["sequence_changes"]
    assert any(c["op"] == "added" and c["tools"] == ["search"] for c in ops)


def test_replaced_tool_detected():
    base = _results({"t1": _task_result(tools=["read_file", "answer"])})
    cur = _results({"t1": _task_result(tools=["web_search", "answer"])})
    diff = compare(base, cur)
    ops = diff["entries"][0]["sequence_changes"]
    assert any(c["op"] == "replaced" and c["before"] == ["read_file"] for c in ops)


def test_arg_drift_detected_even_when_passing():
    base = _results(
        {"t1": _task_result(calls=[{"name": "refund", "args": {"amount": 49.99}}], tools=["refund"])}
    )
    cur = _results(
        {"t1": _task_result(calls=[{"name": "refund", "args": {"amount": 499.99}}], tools=["refund"])}
    )
    diff = compare(base, cur)
    entry = diff["entries"][0]
    assert entry["classification"] == "behavior_change"
    drift = entry["arg_drift"][0]
    assert drift["changed_keys"] == ["amount"]
    assert drift["before"]["amount"] == 49.99
    assert drift["after"]["amount"] == 499.99


def test_cost_regression_flagged_beyond_tolerance():
    base = _results({"t1": _task_result(cost=0.001)})
    cur = _results({"t1": _task_result(cost=0.003)})
    diff = compare(base, cur)
    assert diff["entries"][0]["classification"] == "cost_regression"
    assert diff["has_regressions"]


def test_cost_within_tolerance_not_flagged():
    base = _results({"t1": _task_result(cost=0.001)})
    cur = _results({"t1": _task_result(cost=0.0011)})
    diff = compare(base, cur)
    assert diff["entries"][0]["classification"] == "unchanged"


def test_improvement_classified():
    base = _results({"t1": _task_result(pass_rate=0.5)})
    cur = _results({"t1": _task_result(pass_rate=1.0)})
    diff = compare(base, cur)
    assert diff["summary"]["improvements"] == 1
    assert not diff["has_regressions"]


def test_suite_hash_mismatch_warns():
    base = _results({"t1": _task_result()}, suite_hash="aaa")
    cur = _results({"t1": _task_result()}, suite_hash="bbb")
    diff = compare(base, cur)
    assert diff["warnings"]


def test_added_and_removed_tasks():
    base = _results({"t1": _task_result(), "t2": _task_result()})
    cur = _results({"t1": _task_result(), "t3": _task_result()})
    diff = compare(base, cur)
    assert diff["tasks_added"] == ["t3"]
    assert diff["tasks_removed"] == ["t2"]
