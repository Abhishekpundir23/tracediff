"""tracediff - structural trajectory regression testing for AI agents."""

from tracediff.adapters import from_claude_agent_sdk, from_langgraph, from_openai_agents
from tracediff.suite import Suite, Task, load_suite
from tracediff.trace import Step, Trace, normalize_trace

__version__ = "0.2.0"

__all__ = [
    "Step",
    "Trace",
    "normalize_trace",
    "Suite",
    "Task",
    "load_suite",
    "from_langgraph",
    "from_openai_agents",
    "from_claude_agent_sdk",
    "__version__",
]
