# Real-agent example: Claude Agent SDK

Unlike the offline demo in `examples/`, this agent makes **real model calls**
via the [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/), which
uses your existing Claude Code login (`npm install -g @anthropic-ai/claude-code`,
then log in once).

```bash
pip install tracediff claude-agent-sdk

tracediff run --suite suite.yaml --agent agent:run --repeats 2 --out baseline.json

# simulate a prompt change between commits
TRACEDIFF_DEMO_VARIANT=b tracediff run --suite suite.yaml --agent agent:run --repeats 2 --out current.json

tracediff diff baseline.json current.json
```

Real output from this example (variant a's direct prompt vs variant b's vague one):

```
[REGRESSION] summarize-meeting
    - pass rate 100% -> 0%
    - now calls Glob at position 0
    - now calls Read at position 2
    - Read args drifted: file_path: 'notes/meeting.md' -> '/notes/meeting.md'
    - mean steps 9.5 -> 15.5
```

The vague prompt made the agent explore the workspace (extra Glob), read a
second file, and even format the path differently - all invisible to
score-only evals, all caught structurally. Total cost of the comparison:
about $0.10 with Haiku.
