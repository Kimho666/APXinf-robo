"""Native LIBERO simulator observations, translated for an ApxInf policy.

This is the **in-process** path: the caller holds a live ``Policy`` and feeds it
frames directly. For the out-of-process path -- a simulator host with no ApxInf
installed, talking to a WebSocket server -- copy ``scripts/connect_libero.py``
instead; it is deliberately standalone and shares no code with this module.

The two paths do not build identical state vectors, and that is not an
oversight: ``libero_state`` here emits one gripper coordinate (7 values) to match
the evaluation checkpoints this repository was built against, while the
standalone script emits both raw finger joints (8 values) for the PI0.5-LIBERO
Panda contract. Which one is right is a property of the checkpoint, so the choice
belongs to whoever picks the checkpoint -- check it against
:func:`~apxinf_robo.preflight.check_checkpoint` rather than assuming.
"""

from __future__ import annotations

import math
import pathlib
from typing import Tuple

import numpy as np

__all__ = [
    "quat_to_axis_angle",
    "libero_images",
    "libero_state",
    "make_env",
    "to_apxinf_observation",
]


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / denominator).astype(np.float32)


def libero_images(base: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Orient raw LIBERO frames; the selected policy owns model-specific resize."""
    return np.stack(
        [np.ascontiguousarray(base[::-1, ::-1]), np.ascontiguousarray(wrist[::-1, ::-1])]
    )


def libero_state(observation) -> np.ndarray:
    """Convert LIBERO's two mirrored finger joints to one gripper coordinate."""
    gripper = np.asarray(observation["robot0_gripper_qpos"]).reshape(-1)
    if gripper.size != 2:
        raise ValueError(f"robot0_gripper_qpos must have 2 values, got {gripper.size}")
    return np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            gripper[:1],
        )
    ).astype(np.float32, copy=False)


def make_env(task, seed: int):
    """Build the same off-screen LIBERO environment used by evaluation."""
    try:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:
        raise ImportError(
            "native LIBERO observations require the LIBERO and MuJoCo evaluation "
            "dependencies; install them with `pip install apxinf-robo[libero]` plus "
            "LIBERO itself, as described in README.md"
        ) from error

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    return env


def to_apxinf_observation(
    observation,
    *,
    prompt: str,
    image_keys: Tuple[str, str],
    prompt_key: str,
    state_key: str,
) -> dict:
    """Convert one raw simulator frame using the evaluation-time convention.

    The keys are passed in rather than read from a preset because a server may
    have been started with overrides; take them from the policy's published
    ``metadata`` so what is sent matches what is served.
    """
    images = libero_images(
        observation["agentview_image"],
        observation["robot0_eye_in_hand_image"],
    )
    state = libero_state(observation)
    return {
        image_keys[0]: images[0],
        image_keys[1]: images[1],
        state_key: state,
        prompt_key: prompt,
    }
