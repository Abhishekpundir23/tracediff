"""tracediff - structural trajectory regression testing for AI agents."""

from tracediff.trace import Step, Trace, normalize_trace
from tracediff.suite import Suite, Task, load_suite

__version__ = "0.1.0"

__all__ = ["Step", "Trace", "normalize_trace", "Suite", "Task", "load_suite", "__version__"]
