"""``apxinf-robo`` -- the command-line entry point.

Subcommands are imported lazily. ``inspect`` reads files and needs neither a GPU
nor the ``apxinf_py`` binding; ``serve`` needs both. Importing the dispatcher
must not drag in CUDA, so nothing is imported until a subcommand is chosen.
"""

from __future__ import annotations

import argparse
import sys

_COMMANDS = {
    "serve": ("apxinf_robo.cli.serve", "serve a policy over the OpenPI WebSocket protocol"),
    "inspect": ("apxinf_robo.cli.inspect", "check a checkpoint against a robot preset"),
    "eval-libero": ("apxinf_robo.cli.eval_libero", "run a LIBERO evaluation suite"),
    "capture-libero": (
        "apxinf_robo.cli.capture_libero",
        "write LIBERO observations for ApxInf's FP8 calibration",
    ),
}


def _load(name: str):
    import importlib

    module_name, _ = _COMMANDS[name]
    return importlib.import_module(module_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apxinf-robo",
        description="Robot-facing tooling over the ApxInf inference engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands:\n"
        + "\n".join(f"  {name:<16} {help_}" for name, (_, help_) in _COMMANDS.items())
        + "\n\nRun `apxinf-robo <command> --help` for a command's options.",
    )
    parser.add_argument("command", choices=sorted(_COMMANDS), help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0
    if argv[0] not in _COMMANDS:
        build_parser().error(
            f"unknown command {argv[0]!r}; choose from {', '.join(sorted(_COMMANDS))}"
        )
    result = _load(argv[0]).main(argv[1:])
    return 0 if result is None else int(result)


if __name__ == "__main__":
    sys.exit(main())
