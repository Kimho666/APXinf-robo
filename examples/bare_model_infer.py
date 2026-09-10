#!/usr/bin/env python3
"""L1 -- run the model directly, owning every transform yourself.

Use this layer when you already have a preprocessing stack you trust and want
only the engine's forward pass: RLinf's vendored openpi transforms enter here,
and so does anything replaying a recorded dataset that was tokenized offline.

At L1 nothing is done for you. The contract is narrow and unforgiving:

===============  ==========================================================
``rgb_u8``       ``uint8``, already at the model's ``image_size`` -- there
                 is no resize step at this layer -- in ``layout`` order
``layout``       ``"nhwc"`` or ``"nchw"``
``token_ids``    ``uint32`` array from *your* tokenizer, not a list
``noise``        optional ``float32 [action_horizon, action_dim]``; omit it
                 to use the model's internal device-side sampling stream
===============  ==========================================================

What comes back is **normalized-domain** and **untrimmed**: ``[action_horizon,
action_dim]`` at the checkpoint's full width, before any unnormalization and
before the trim to the robot's deployable width. Turning that into something a
robot can execute is also yours to do. If you do not want that job, use L2
(``robot_policy_infer.py``), which does all of it.

Requires the ``apxinf_py`` CUDA binding and a checkpoint directory. The frames
here are synthetic, so the actions are meaningless -- the call shape is the
point.

    python examples/bare_model_infer.py --model-dir /path/to/checkpoint
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import _common  # noqa: F401  (installs the source-checkout path shim)

from apxinf_robo import load_bare_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", required=True, type=pathlib.Path)
    parser.add_argument("--precision", choices=("auto", "fp8", "bf16", "int8"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--no-tactics",
        action="store_true",
        help="load the provider-default kernels instead of the tuned ones, "
        "which changes the numbers (see tests/test_parity.py)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # load_bare_model selects the same tuned GEMM tactics the L2 loader selects.
    # That is not cosmetic: without it the two layers return different numbers
    # for identical inputs, which tests/test_parity.py pins in both directions.
    model = load_bare_model(
        args.model_dir,
        device=args.device,
        precision=args.precision,
        **({"tactics": None} if args.no_tactics else {}),
    )

    # The handle states its own shape requirements. Read them rather than
    # hardcoding: image_size and action_dim are checkpoint facts, and a mismatch
    # is rejected at the binding rather than silently reinterpreted.
    print("the model's input contract:")
    print(f"  num_views      {model.num_views}")
    print(f"  image_size     {model.image_size}")
    print(f"  action_horizon {model.action_horizon}")
    print(f"  action_dim     {model.action_dim}  (checkpoint width, not the robot's)")

    size = model.image_size
    height, width = size if isinstance(size, (tuple, list)) else (size, size)
    rgb = np.zeros((model.num_views, height, width, 3), np.uint8)

    # Your tokenizer's output. uint32, and an array -- the binding does not
    # accept a Python list.
    token_ids = np.zeros((10,), np.uint32)

    # Pinning the noise makes the call reproducible. Omit the argument entirely
    # and the model samples on-device instead, which is faster and non-repeatable.
    noise = np.zeros((model.action_horizon, model.action_dim), np.float32)

    actions = np.asarray(model.infer_rgb(rgb, "nhwc", token_ids, noise))
    print()
    print(f"infer_rgb -> {actions.shape} {actions.dtype}")
    print("  these are normalized-domain, full-width actions. To deploy them you")
    print("  still owe: unnormalize with this checkpoint's statistics, then trim")
    print(f"  {model.action_dim} -> the robot's width. L2 does both for you.")


if __name__ == "__main__":
    main()
