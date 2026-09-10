#!/usr/bin/env python3
"""L2 -- load a checkpoint as a policy, with no robot preset involved.

`load_policy` is the engine's own loader: it dispatches `model_dir` to the
concrete policy class for whatever family the checkpoint is (pi05, walloss) and
gives you back `infer(observation) -> {actions, ...}` with the engine's resize /
tokenize / normalize / noise chain in front of it.

What it does *not* give you is a robot. Nothing here knows a camera is mounted on
a wrist or that dimension 6 is a gripper. So the wire contract -- which keys an
observation carries, which one holds proprioception, how wide a deployable action
is -- has to come from the caller, spelled out argument by argument:

    load_policy(model_dir, image_keys=(...), state_key=..., prompt_key=...,
                action_dim=7, metadata={"state_dim": 8})

`state_dim` sits in `metadata` rather than beside `state_key` because it is not
a loader argument: the engine publishes a state width only when the checkpoint
normalizes proprioception, and a checkpoint that ignores it publishes none --
while a client still has to know how wide a vector to send.

That is the whole difference from `robot_policy_infer.py`, which is this call
plus a named preset that fills those arguments in. Use this layer when your robot
has no preset and you do not want one -- a one-off rig, a dataset replay, a
client whose dialect is already fixed. When the same arguments start appearing at
every call site, that is the signal to register a preset instead; see
`register_preset.py`.

Requires the `apxinf_py` CUDA binding and a checkpoint directory. The frames here
are synthetic, so the actions are meaningless -- the call shape is the point.

    python examples/policy_infer.py --model-dir /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import pathlib

from _common import json_object, observation_for_policy, policy_kwargs  # noqa: E402

from apxinf_robo import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--norm-stats",
        type=pathlib.Path,
        default=None,
        help="explicit normalization statistics; some checkpoints ship without them",
    )
    parser.add_argument(
        "--image-keys",
        nargs="+",
        default=["observation/image", "observation/wrist_image"],
        help="wire keys, in model view-slot order: entry i fills slot i",
    )
    parser.add_argument("--state-key", default="observation/state")
    parser.add_argument(
        "--state-dim",
        type=int,
        default=8,
        help="width of the proprioception vector a client must send; 0 means "
        "this checkpoint takes none, and --state-key is dropped",
    )
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument(
        "--action-dim",
        type=int,
        default=7,
        help="deployable width to trim to; 0 keeps the checkpoint's full vector",
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

    # The width of the state vector is not something the loader can tell you: a
    # checkpoint trained without proprioception still accepts a state_key and
    # simply ignores what arrives under it. So a server that declares one owes
    # clients the width too, and that goes in metadata -- which is what the
    # server publishes on connect. `build_robot_policy` does exactly this from
    # `preset.state_dim`; with no preset it is yours to state.
    state_key = args.state_key if args.state_dim else None
    options = policy_kwargs(
        args.policy_options,
        device=args.device,
        precision=args.precision,
        action_dim=args.action_dim,
        metadata={"state_dim": args.state_dim} if args.state_dim else None,
    )
    if args.norm_stats is not None:
        options["norm_stats"] = args.norm_stats

    # No preset, so every contract argument is stated here. Order matters for
    # image_keys: entry i fills model view slot i, and a reordered tuple feeds
    # the wrong camera to each slot without any error.
    policy = load_policy(
        args.model_dir,
        image_keys=tuple(args.image_keys),
        state_key=state_key,
        prompt_key=args.prompt_key,
        **options,
    )
    try:
        # The policy republishes what it settled on. Worth printing even though
        # you just passed it in: the loader fills in defaults you did not state,
        # and this is what a connecting client will actually be told.
        print("resolved wire contract:")
        for field, value in sorted(policy.metadata.items()):
            print(f"  {field:16s} {value!r}")
        print(f"  {'action_dim':16s} {policy.action_dim!r}")

        result = policy.infer(observation_for_policy(policy, prompt=args.prompt))
        actions = result["actions"]
        print(f"\nactions: shape={actions.shape} dtype={actions.dtype}  (unnormalized)")
        print(f"timing:  {result['timing']}")
        print("\nsame call under a named preset, with these arguments supplied for")
        print("you: examples/robot_policy_infer.py")
    finally:
        policy.close()


if __name__ == "__main__":
    main()
