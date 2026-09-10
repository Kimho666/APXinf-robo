"""The Unitree G1 body: camera count, state/action widths, and its builder."""

from __future__ import annotations

from ..base import Embodiment
from .builder import build_unitree_g1_policy
from .steps import G1_ROBOT_DIM

__all__ = ["UNITREE_G1_BODY"]

#: Unitree G1: 3 cameras and 16 state/action values. The encode step owns output
#: selection, so ``action_dim`` remains ``None`` while ``action_width`` is 16.
UNITREE_G1_BODY = Embodiment(
    name="unitree_g1",
    num_cameras=3,
    action_dim=None,
    state_dim=G1_ROBOT_DIM,
    action_width=G1_ROBOT_DIM,
    builder=build_unitree_g1_policy,
    builder_kwargs={
        "use_delta_joint_actions": True,
        "adapt_to_pi": True,
        # Match OpenPI's float64 normalization. State values near a discretization
        # bin edge can otherwise produce different token ids.
        "norm_dtype": "float64",
    },
)
