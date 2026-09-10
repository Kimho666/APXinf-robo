#!/usr/bin/env python3
"""Will this checkpoint drive this robot correctly? Answered before loading it.

The question is **checkpoint against robot**, not checkpoint against hardware.
Nothing here looks at your GPU, your driver, your memory, or whether a tuned
kernel database exists for this device -- those failures are loud, and you find
out by loading. The failure this catches is the quiet one: a checkpoint that
loads cleanly, serves cleanly, and drives the wrong body.

Four comparisons decide it, all between *checkpoint facts* and *preset
declarations*:

=====================  =========================================  ======
action stats width     vs ``preset.action_width``                 FAIL
state stats width      vs ``preset.state_dim`` (if state is used) FAIL
camera count / keys    vs ``preset.image_keys``                   WARN
serving overrides      ``--action-dim`` / ``--discrete-state``    WARN
=====================  =========================================  ======

The widths are fatal because normalization statistics are a physical claim: 7
numbers describing a Franka's end-effector deltas, applied to a 16-DoF G1, map
the model's output through the wrong range. Too narrow and the unnormalizer
refuses the array; too wide and it silently unnormalizes joints the robot does
not have. Neither is an error at inference time.

On top of those, the engine's own ``inspect_checkpoint`` contributes facts about
the checkpoint alone -- detected layout, which ``norm_stats.json`` actually
resolved and whether it was a fallback, pi05's mandatory quantiles, the
tokenizer. Those are true no matter what is on the other end of the wire, which
is why they live in the engine and the four above live here.

Loads no weights, imports no torch, touches no network: this runs on a laptop.
``apxinf-robo inspect`` is the same check as a CLI.

    python examples/preflight_check.py --model-dir /path/to/checkpoint
    python examples/preflight_check.py --model-dir /path/to/ckpt --robot unitree_g1
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import _common  # noqa: F401  (installs the source-checkout path shim)

from apxinf_robo import (
    FAIL,
    WARN,
    available_robots,
    format_findings,
    get_robot_preset,
    inspect_for_robot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--robot",
        action="append",
        choices=available_robots(include_aliases=True),
        help="preset to check against; repeatable. Default: every registered preset",
    )
    parser.add_argument(
        "--norm-stats",
        type=pathlib.Path,
        default=None,
        help="explicit statistics file, as passed to build_robot_policy",
    )
    parser.add_argument("--quiet", action="store_true", help="drop INFO lines")
    return parser.parse_args()


def print_comparison(report, preset) -> None:
    """Show the two sides being compared, so the verdict is not a black box."""
    print(f"  {'':22s} {'checkpoint':>10s}     {'preset':>10s}")
    rows = (
        ("action stats width", report.norm.get("action"), preset.action_width),
        ("state stats width", report.norm.get("state"), preset.state_dim),
    )
    for label, facts, expected in rows:
        # A role missing from report.norm was never inspected -- state is only
        # looked up when the preset actually feeds proprioception to the model.
        if facts is None or expected is None:
            continue
        found = facts.width if facts.present and facts.width is not None else "absent"
        flag = "" if found == expected else "   <-- mismatch"
        print(f"  {label:22s} {str(found):>10s}  vs  {str(expected):>10s}{flag}")

    # The checkpoint does not record which wire keys a client will send, so
    # cameras have only one side here: the preset's slots, in slot order. The
    # WARN comes from comparing them with the keys a *deployment* overrides to.
    print(f"  {'camera slots':22s} {'':>10s}      {preset.num_views} "
          f"{list(preset.image_keys)}")
    if report.norm_stats_path is not None:
        print(f"  statistics read from   {report.norm_stats_path}")


def main() -> int:
    args = parse_args()
    robots = args.robot or list(available_robots())

    worst = 0
    for robot in robots:
        preset = get_robot_preset(robot)
        print(f"======== {robot}")

        # inspect_for_robot is check_checkpoint plus the engine's raw report, so
        # the facts behind the verdict can be shown alongside it.
        report, findings = inspect_for_robot(
            args.model_dir, robot, norm_stats=args.norm_stats
        )
        print_comparison(report, preset)
        print()

        rendered = format_findings(findings, include_info=not args.quiet)
        if rendered:
            print(rendered)

        levels = {finding.level for finding in findings}
        if FAIL in levels:
            worst = max(worst, 2)
        elif WARN in levels:
            worst = max(worst, 1)
        print()

    # Same exit-status contract as `apxinf-robo inspect`: 2 on any FAIL, 1 on a
    # WARN, 0 otherwise, so a deployment script can gate on it. With no --robot
    # this checks every registered preset, so a single-robot checkpoint exits 2
    # by design -- pass the preset you actually intend to serve.
    if worst == 2:
        print("verdict: at least one preset would serve this checkpoint wrong")
    elif worst == 1:
        print("verdict: serviceable, with warnings worth reading")
    else:
        print("verdict: clean")
    return worst


if __name__ == "__main__":
    sys.exit(main())
