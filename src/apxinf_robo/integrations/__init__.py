"""Third-party framework integrations.

Each module here makes an ApxInf policy usable *from inside* another framework's
own loop, rather than the other way round: the framework keeps its environments,
its batching, and its action dispatch, and only the policy object is swapped.

* :mod:`.lerobot` -- a ``PI0Policy``-shaped facade for a hand-written lerobot
  control loop.
* :mod:`.rlinf` -- a ``BasePolicy``-shaped action model for RLinf's rollout
  workers, for **evaluation** (the engine underneath has no autograd).

Nothing here is imported by :mod:`apxinf_robo` itself. Each module imports its
host framework lazily, so this package installs and tests without lerobot, RLinf,
or torch present.
"""

from __future__ import annotations

__all__ = ["lerobot", "rlinf"]
