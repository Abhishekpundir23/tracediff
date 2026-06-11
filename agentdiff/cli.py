"""agentdiff CLI.

  agentdiff run   --suite suite.yaml --agent module:fn [--split dev] [--repeats 3] --out results.json
  agentdiff diff  baseline.json current.json [--md comment.md] [--fail-on regressions|none]
  agentdiff suite suite.yaml            # inspect version hash and split

Exit codes: `run` exits 0 once results are written (results are data, not a
verdict); `diff` exits 1 when regressions are found and --fail-on is
'regressions' (the default), which is what gates a CI pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentdiff import __version__
from agentdiff.diff import compare
from agentdiff.report import render_diff_markdown, render_diff_text, render_run_text
from agentdiff.runner import load_agent, run_suite
from agentdiff.suite import HoldoutBudgetExceeded, load_suite, record_holdout_reveal


def _cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)

    if args.split == "holdout" or args.split == "all":
        try:
            used = record_holdout_reveal(suite, override=args.allow_holdout_overrun)
        except HoldoutBudgetExceeded as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(
            f"note: holdout reveal {used}/{suite.max_holdout_reveals} "
            f"for suite version {suite.content_hash}",
            file=sys.stderr,
        )

    # The agent module typically lives in the working directory, not site-packages.
    sys.path.insert(0, str(Path.cwd()))
    agent_fn = load_agent(args.agent)

    results = run_suite(
        suite, agent_fn, split=args.split, repeats=args.repeats, progress=lambda m: print(m)
    )
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(render_run_text(results))
    print(f"\nresults written to {args.out}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    diff = compare(baseline, current, cost_tolerance=args.cost_tolerance)

    print(render_diff_text(diff))
    if args.md:
        Path(args.md).write_text(render_diff_markdown(diff), encoding="utf-8")
        print(f"\nmarkdown written to {args.md}")
    if args.json:
        Path(args.json).write_text(json.dumps(diff, indent=2), encoding="utf-8")

    if args.fail_on == "regressions" and diff["has_regressions"]:
        return 1
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    dev, holdout = suite.split("dev"), suite.split("holdout")
    print(f"suite:    {suite.name}")
    print(f"version:  {suite.content_hash}")
    print(f"tasks:    {len(suite.tasks)} ({len(dev)} dev / {len(holdout)} holdout, seed={suite.seed})")
    print(f"holdout reveal budget: {suite.max_holdout_reveals}")
    print(f"dev:      {', '.join(t.id for t in dev) or '(none)'}")
    print(f"holdout:  {', '.join(t.id for t in holdout) or '(none)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentdiff",
        description="Structural trajectory regression testing for AI agents.",
    )
    parser.add_argument("--version", action="version", version=f"agentdiff {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run an agent over a suite split and write results JSON")
    run_p.add_argument("--suite", required=True, help="path to suite YAML/JSON")
    run_p.add_argument("--agent", required=True, help="agent entrypoint, e.g. my_agent:run")
    run_p.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    run_p.add_argument("--repeats", type=int, default=1, help="runs per task (variance)")
    run_p.add_argument("--out", default="results.json", help="output path for results JSON")
    run_p.add_argument(
        "--allow-holdout-overrun",
        action="store_true",
        help="evaluate holdout even after the reveal budget is spent (recorded)",
    )
    run_p.set_defaults(fn=_cmd_run)

    diff_p = sub.add_parser("diff", help="structurally diff two results files")
    diff_p.add_argument("baseline", help="baseline results.json")
    diff_p.add_argument("current", help="current results.json")
    diff_p.add_argument("--md", help="also write a markdown report (for PR comments)")
    diff_p.add_argument("--json", help="also write the machine-readable diff")
    diff_p.add_argument("--cost-tolerance", type=float, default=1.25, help="flag cost ratio above this (default 1.25)")
    diff_p.add_argument("--fail-on", choices=("regressions", "none"), default="regressions")
    diff_p.set_defaults(fn=_cmd_diff)

    suite_p = sub.add_parser("suite", help="inspect a suite: version hash and dev/holdout split")
    suite_p.add_argument("suite", help="path to suite YAML/JSON")
    suite_p.set_defaults(fn=_cmd_suite)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
