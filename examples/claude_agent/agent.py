"""A REAL agent example: Claude Agent SDK driving Claude with file tools.

Unlike the offline demo agent, this one makes actual model calls. The Claude
CLI must be installed and authenticated (the SDK uses your existing Claude
Code login - BYOK in practice).

The TRACEDIFF_DEMO_VARIANT env var picks which prompt the agent runs with,
simulating a prompt change between two commits:
  a - direct: read the meeting notes file and summarize it
  b - vague: find the notes first, then summarize the meeting
Variant b typically makes the agent explore (extra tool calls, higher cost)
before reading - exactly the kind of silent behavioral drift tracediff
surfaces in the structural diff.
"""

import asyncio
import os
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, query

from tracediff import from_claude_agent_sdk

WORKSPACE = str(Path(__file__).resolve().parent / "workspace")


async def _collect(prompt: str) -> list:
    options = ClaudeAgentOptions(
        cwd=WORKSPACE,
        allowed_tools=["Read", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        max_turns=8,
        model="haiku",
    )
    return [message async for message in query(prompt=prompt, options=options)]


def run(task_input):
    variant = os.environ.get("TRACEDIFF_DEMO_VARIANT", "a")
    prompt = task_input["prompts"][variant]
    messages = asyncio.run(_collect(prompt))
    return from_claude_agent_sdk(messages)
