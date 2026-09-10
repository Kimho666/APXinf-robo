#!/usr/bin/env python3
"""Load a checkpoint under a named robot preset and run one inference.

This is the package's headline call. ``build_robot_policy(robot, model_dir)`` is
:func:`apxinf_robo.load_policy` plus a preset -- an ``Embodiment x Convention``
pairing that supplies the camera keys, the state routing, the deployable action
width, and (for bodies that have them) the robot's own pre/post steps.

The preset is *not* negotiated at connect time. A checkpoint fine-tuned for
another robot served under the wrong preset produces wrong actions, not an
error, which is what ``preflight_check.py`` exists to catch first.

Requires the ``apxinf_py`` CUDA binding and a checkpoint directory.

    python examples/robot_policy_infer.py --model-dir /path/to/checkpoint
    python examples/robot_policy_infer.py --model-dir /path/to/ckpt --robot unitree_g1
"""

from __future__ import annotations

import argparse
import pathlib

from _common import json_object, observation_for_policy, policy_kwargs  # noqa: E402

from apxinf_robo import ROBOT_PRESETS, available_robots, build_robot_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="robot presets:\n"
        + "\n".join(f"  {preset.describe()}" for preset in ROBOT_PRESETS.values()),
    )
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument(
        "--robot",
        choices=available_robots(include_aliases=True),
        default="franka_libero",
        help="named Embodiment x Convention pairing to serve the checkpoint under",
    )
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--norm-stats",
        type=pathlib.Path,
        default=None,
        help="explicit normalization statistics; some checkpoints ship without them",
    )
    parser.add_argument(
        "--policy-options",
        type=json_object,
        default={},
        metavar="JSON",
        help="extra concrete-policy options as a JSON object",
    )
    parser.add_argument("--prompt", default="pick up the block")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    options = policy_kwargs(
        args.policy_options, device=args.device, precision=args.precision
    )
    if args.norm_stats is not None:
        options["norm_stats"] = args.norm_stats

    # The preset supplies image_keys / state_key / prompt_key / action_dim. Every
    # one of them is overridable per argument, for a deployed client that already
    # speaks a fixed dialect -- see build_robot_policy's docstring.
    policy = build_robot_policy(args.robot, args.model_dir, **options)
    try:
        metadata = policy.metadata
        print("served wire contract (published to clients on connect):")
        for field in (
            "robot",
            "robot_steps",
            "image_keys",
            "state_key",
            "state_dim",
            "prompt_key",
            "discrete_state",
        ):
            if field in metadata:
                print(f"  {field:16s} {metadata[field]!r}")
        print(f"  {'action_dim':16s} {policy.action_dim!r}")

        # `robot_steps` is the one worth reading twice: False means the preset's
        # keys are served without its arithmetic, so the wire looks right and the
        # numbers are the model's raw output.
        if not metadata.get("robot_steps", False):
            print("  note: this preset wires no robot-specific pre/post steps")

        result = policy.infer(observation_for_policy(policy, prompt=args.prompt))
        actions = result["actions"]
        print(f"actions: shape={actions.shape} dtype={actions.dtype}  (unnormalized)")
        print(f"timing:  {result['timing']}")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
