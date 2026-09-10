"""Evaluation-environment glue: simulator frames in, policy observations out.

An environment is the third thing that varies independently of the checkpoint
and the body. LIBERO's MuJoCo images arrive rotated 180 degrees; robosuite hands
out XYZW quaternions where IsaacLab hands out WXYZ; ``OSC_POSE`` expects
normalized deltas within a controller-specific output range. None of that is a
fact about the model, so none of it lives in the engine.

Two shapes of adapter live here, and they are not interchangeable:

**In-process** (this package) -- the caller holds a live
:class:`~apxinf.policies.base.Policy` and calls it directly. Import
:mod:`apxinf_robo.envs.libero` and friends.

**Out-of-process** (``scripts/connect_*.py``) -- the simulator host has no ApxInf
installed and speaks to a WebSocket server. Those scripts are single-file by
design: copy one to the simulator machine and run it. They intentionally
duplicate the arithmetic here rather than import it, because importing it would
mean installing this package on a host that cannot build the engine.

Modules import their simulator lazily, inside the function that needs it, so
importing this package never requires MuJoCo, robosuite, or IsaacLab.
"""

from __future__ import annotations

__all__ = ["libero"]
