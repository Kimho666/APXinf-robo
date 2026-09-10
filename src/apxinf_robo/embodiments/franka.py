"""Franka Emika Panda."""

from __future__ import annotations

from .base import Embodiment

__all__ = ["FRANKA"]

#: Franka Emika Panda under LIBERO: 2 cameras, 8 state values, 7 actions.
FRANKA = Embodiment(name="franka", num_cameras=2, action_dim=7, state_dim=8, action_width=7)
