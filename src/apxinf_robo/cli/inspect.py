"""``apxinf-robo inspect`` -- run the preflight and print it, without serving.

The same checks ``serve`` runs before it loads weights, available on their own so
a checkpoint can be vetted on a machine with no GPU. Nothing here loads weights,
imports torch, or touches the network.

Exit status is the worst finding: ``2`` for a FAIL, ``1`` for a WARN (unless
``--warn-ok``), ``0`` otherwise -- so this is usable as a deployment gate.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from apxinf.checkpoints import FORMATS as CHECKPOINT_FORMATS

from ..embodiments.base import ROBOT_PRESETS, available_robots
from ..preflight import FAIL, WARN, check_checkpoint, format_findings


def _split_keys(value: str) -> tuple:
    keys = tuple(part.strip() for part in value.split(",") if part.strip())
    if not keys:
        raise argparse.ArgumentTypeError("--image-keys needs at least one camera key")
    return keys


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("model_dir", type=pathlib.Path, help="checkpoint directory")
    parser.add_argument(
        "--robot",
        choices=available_robots(include_aliases=True),
        default="franka_libero",
        help="embodiment preset to check the checkpoint against",
    )
    parser.add_argument("--image-keys", type=_split_keys, default=None)
    parser.add_argument("--action-dim", type=int, default=None)
    parser.add_argument("--norm-key", default="actions")
    parser.add_argument("--tokenizer", type=pathlib.Path, default=None)
    parser.add_argument("--ckpt-format", choices=CHECKPOINT_FORMATS, default="auto")
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--norm-stats", type=pathlib.Path, default=None)
    parser.add_argument(
        "--discrete-state", dest="discrete_state", action="store_true", default=None
    )
    parser.add_argument("--no-discrete-state", dest="discrete_state", action="store_false")
    parser.add_argument("--quiet", action="store_true", help="drop INFO lines")
    parser.add_argument(
        "--warn-ok",
        action="store_true",
        help="exit 0 on warnings; only a FAIL is an error",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    robot_help = "\n".join(f"  {p.describe()}" for p in ROBOT_PRESETS.values())
    parser = argparse.ArgumentParser(
        prog="apxinf-robo inspect",
        description="Check a checkpoint against the robot preset that would serve it",
        epilog=f"robot presets (--robot):\n{robot_help}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    return add_arguments(parser)


def run(args: argparse.Namespace) -> int:
    findings = check_checkpoint(
        args.model_dir,
        args.robot,
        norm_key=args.norm_key,
        discrete_state=args.discrete_state,
        image_keys=args.image_keys,
        action_dim=args.action_dim,
        tokenizer_path=args.tokenizer,
        checkpoint_format=args.ckpt_format,
        asset_id=args.asset_id,
        norm_stats=args.norm_stats,
    )
    rendered = format_findings(findings, include_info=not args.quiet)
    if rendered:
        print(rendered)
    levels = {f.level for f in findings}
    if FAIL in levels:
        return 2
    if WARN in levels and not args.warn_ok:
        return 1
    return 0


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
