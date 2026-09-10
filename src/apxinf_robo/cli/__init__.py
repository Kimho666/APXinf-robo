"""Command-line entry points for apxinf-robo.

The console script is ``apxinf-robo``; see :mod:`apxinf_robo.cli.__main__` for
the dispatcher. Each subcommand module also runs standalone via
``python -m apxinf_robo.cli.<name>``.

Porting-time tooling is not here, and neither are demonstrations.
``scripts/from_openpi.py`` diffs a checkpoint's own ``metadata.pt`` against a
preset, which is a question you answer once rather than a command you operate;
``examples/g1_adapter_smoke.py`` shows how to prove a new body's adapter chain
runs, which is a pattern you copy rather than a service you run.
"""

from __future__ import annotations

__all__ = ["serve", "inspect", "eval_libero", "capture_libero"]
