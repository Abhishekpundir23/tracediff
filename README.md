# tracediff

**Structural trajectory regression testing for AI agents.** Diff what your agent *did*, not just its score.

Every eval tool can tell you "accuracy dropped 3%." tracediff tells you *why*:

```
2 regression(s), 1 cost regression(s), 1 behavior change(s) across 4 task(s)

[REGRESSION] summarize-meeting
    - pass rate 100% -> 0%
    - calls read_file instead of read_file at position 0
    - read_file args drifted: path
[BEHAVIOR CHANGE] refund-order
    - issue_refund args drifted: amount   ({"amount": 49.99} -> {"amount": 499.99})
[COST REGRESSION] capital-question
    - now calls search at position 1
    - mean cost $0.0012 -> $0.0029 (2.42x)
```

That third one is the kind of bug that never shows up in a score: the agent still answers correctly — it just silently started calling `search` twice and your bill doubled. The second one is worse: output unchanged, refund amount 10x. Score-level diffing misses both.

## Why

- **Scores hide behavior.** Pass/fail diffs and LLM-judge verdicts can't tell you "step 4's tool args drifted between commits."
- **Agents are stochastic.** A single run is a sample, not a measurement. tracediff runs repeats and reports variance, so you can tell drift from noise.
- **Cost is a first-class metric.** Research (Kapoor et al., NeurIPS 2024) showed accuracy-only evals reward agents that cost 50x more for the same results.
- **Benchmarks leak.** Most agent benchmarks have no holdout discipline. tracediff suites have a built-in dev/holdout split with a *reveal budget* — evaluating the holdout more than N times per suite version requires an explicit, recorded override.
- **BYOK by construction.** tracediff never calls a model provider. Your agent runs with your keys; tracediff scores the traces.

## Install

```bash
pip install tracediff
```

## Quickstart (60 seconds, no API keys)

The repo ships a deterministic demo agent. Run the baseline, "change the code" (set an env var), re-run, and diff:

```bash
git clone https://github.com/Abhishekpundir23/tracediff && cd tracediff/examples

tracediff run --suite suite.yaml --agent demo_agent:run --repeats 3 --out baseline.json

# simulate a code change that subtly breaks the agent
TRACEDIFF_DEMO_VARIANT=b tracediff run --suite suite.yaml --agent demo_agent:run --repeats 3 --out current.json

tracediff diff baseline.json current.json --md report.md
```

## Wiring up your agent

Expose one function. It gets the task input and returns a trace in any of three shapes:

```python
# my_agent.py
def run(task_input):
    messages, usage = my_agent_loop(task_input)   # your existing code, your keys
    return {"messages": messages, "usage": usage}  # OpenAI-style messages work as-is
```

Accepted return shapes:
1. **OpenAI-style message list** (assistant `tool_calls` + tool-role results) — works directly with most frameworks' message history.
2. **`tracediff.Trace`** — build it natively for full control (per-step tokens/cost).
3. **Serialized dict** — `{"steps": [...], "final_output": "..."}`.

## Writing a suite

```yaml
suite: my-agent-suite
seed: 7
holdout_fraction: 0.25     # deterministic split by task-id hash
max_holdout_reveals: 5     # holdout governance: budgeted, recorded reveals

tasks:
  - id: refund-order
    input: { topic: refund, order_id: A-100 }
    expect:
      tools: [lookup_order, issue_refund]   # expected tool trajectory
      mode: strict                          # strict | unordered | subset
      args:
        issue_refund: { order_id: A-100, amount: 49.99 }
      max_tool_calls: 4                     # budgets are first-class
      max_cost_usd: 0.01
    checks:
      - type: output_contains               # output_contains | output_not_contains
        value: refund                       # | output_equals | output_regex
```

The suite **version is a content hash** — edit any task and you get a new version. Diffs warn when results from different suite versions are compared, and each new version gets a fresh holdout budget.

```bash
tracediff suite suite.yaml      # inspect version hash + dev/holdout split
tracediff run --suite suite.yaml --agent my_agent:run --split holdout   # budgeted
```

## CI: structural diffs on every PR

```yaml
# .github/workflows/tracediff.yml
name: tracediff
on: pull_request

jobs:
  eval:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }

      # restore the baseline produced on main (artifact, cache, or committed file)
      - name: Restore baseline
        run: cp .tracediff/baseline.json baseline.json

      - name: Run + diff
        run: |
          pip install tracediff
          tracediff run --suite evals/suite.yaml --agent my_agent:run --repeats 3 --out current.json
          tracediff diff baseline.json current.json --md report.md   # exits 1 on regressions

      - name: Comment on PR
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            github.rest.issues.createComment({
              ...context.repo,
              issue_number: context.issue.number,
              body: fs.readFileSync('report.md', 'utf8'),
            });
```

A composite action wrapping these steps lives in [`action/`](action/action.yml).

## What gets detected

| Category | Example finding |
|---|---|
| **regression** | pass rate 100% → 33% on `summarize-meeting` |
| **behavior change** | `issue_refund` args drifted: `amount` 49.99 → 499.99 (output unchanged!) |
| **cost regression** | now calls `search` twice; mean cost 2.4x baseline |
| **improvement** | pass rate 50% → 100% |

Plus: tools added/removed/replaced/reordered with positions, step-count drift, trajectory variance across repeats (flakiness), tasks added/removed, suite-version mismatch warnings.

## Roadmap

- v0.1 (this): trace ingestion, structural scoring + budgets, repeat variance, structural diff, CLI, CI action, holdout governance
- v0.2: adapters for LangGraph / OpenAI Agents SDK / Claude Agent SDK trace exports, OpenTelemetry GenAI spans
- v0.3: automated benchmark construction — generate decontaminated, holdout-split task suites from your domain

## License

Apache-2.0
