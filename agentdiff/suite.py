"""Task suites: versioned task sets with built-in holdout governance.

A suite file (YAML or JSON) defines tasks plus a deterministic dev/holdout
split. The suite version is a content hash, so any task edit produces a new
version - results from different suite versions are never silently compared.

Holdout governance: every scored run against the holdout split is recorded in
a state file keyed by suite hash. Once the reveal budget for a suite version
is spent, further holdout runs require an explicit override. This is the
discipline most public agent benchmarks lack (12/17 fail holdout criteria per
Princeton HAL's analysis) - here it is a default, not an afterthought.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_HOLDOUT_FRACTION = 0.25
DEFAULT_MAX_HOLDOUT_REVEALS = 5
STATE_FILENAME = ".agentdiff-state.json"


class HoldoutBudgetExceeded(RuntimeError):
    pass


@dataclass
class Task:
    id: str
    input: Any
    expect: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        if "id" not in d:
            raise ValueError("every task needs an 'id'")
        return cls(
            id=str(d["id"]),
            input=d.get("input"),
            expect=d.get("expect") or {},
            checks=d.get("checks") or [],
        )


@dataclass
class Suite:
    name: str
    tasks: list[Task]
    seed: int = 0
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION
    max_holdout_reveals: int = DEFAULT_MAX_HOLDOUT_REVEALS
    source_path: Path | None = None
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def content_hash(self) -> str:
        canonical = json.dumps(self._raw, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    def _is_holdout(self, task_id: str) -> bool:
        h = hashlib.sha256(f"{self.seed}:{task_id}".encode("utf-8")).digest()
        bucket = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
        return bucket < self.holdout_fraction

    def split(self, which: str) -> list[Task]:
        if which == "all":
            return list(self.tasks)
        if which == "dev":
            return [t for t in self.tasks if not self._is_holdout(t.id)]
        if which == "holdout":
            return [t for t in self.tasks if self._is_holdout(t.id)]
        raise ValueError(f"unknown split '{which}' (expected dev|holdout|all)")


def load_suite(path: str | Path) -> Suite:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict) or "tasks" not in raw:
        raise ValueError(f"{path}: suite file must be a mapping with a 'tasks' list")

    tasks = [Task.from_dict(t) for t in raw["tasks"]]
    ids = [t.id for t in tasks]
    if len(ids) != len(set(ids)):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"{path}: duplicate task ids: {dupes}")

    return Suite(
        name=str(raw.get("suite", path.stem)),
        tasks=tasks,
        seed=int(raw.get("seed", 0)),
        holdout_fraction=float(raw.get("holdout_fraction", DEFAULT_HOLDOUT_FRACTION)),
        max_holdout_reveals=int(raw.get("max_holdout_reveals", DEFAULT_MAX_HOLDOUT_REVEALS)),
        source_path=path,
        _raw=raw,
    )


def _state_path(suite: Suite, state_dir: str | Path | None) -> Path:
    if state_dir is not None:
        base = Path(state_dir)
    elif suite.source_path is not None:
        base = suite.source_path.parent
    else:
        base = Path.cwd()
    return base / STATE_FILENAME


def record_holdout_reveal(
    suite: Suite, state_dir: str | Path | None = None, override: bool = False
) -> int:
    """Record one holdout evaluation; enforce the suite's reveal budget.

    Returns the number of reveals used so far (including this one). Raises
    HoldoutBudgetExceeded when the budget is spent, unless override is set
    (the override is still recorded, so overuse stays visible).
    """
    path = _state_path(suite, state_dir)
    state: dict[str, Any] = {}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))

    key = suite.content_hash
    entry = state.get(key) or {"suite": suite.name, "holdout_reveals": 0, "overridden": 0}
    if entry["holdout_reveals"] >= suite.max_holdout_reveals and not override:
        raise HoldoutBudgetExceeded(
            f"suite '{suite.name}' (version {key}) has used all "
            f"{suite.max_holdout_reveals} holdout reveals. Evaluating the holdout split "
            f"again risks overfitting to it. Edit the suite (new version, fresh budget) "
            f"or pass --allow-holdout-overrun to proceed anyway."
        )

    entry["holdout_reveals"] += 1
    if entry["holdout_reveals"] > suite.max_holdout_reveals:
        entry["overridden"] += 1
    state[key] = entry
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return entry["holdout_reveals"]
