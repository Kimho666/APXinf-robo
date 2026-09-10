"""Shared helpers for the ``apxinf-robo`` examples.

Keeps each example short: a ``sys.path`` shim so they run straight from a source
checkout without ``pip install``, plus a synthetic observation builder so no
dataset or simulator is needed to see the API work.

The engine has its own copy of this file (``apxinf/python/apxinf/examples/
_common.py``) that builds an observation from a *policy*. This one can
also build from a *preset*, which is the difference this package exists for: a
preset states the wire contract before any weights are loaded, so an example can
show the contract on a machine with no GPU.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

# Run from a source checkout without ``pip install``: put src/ on the path. (When
# apxinf-robo is installed this is a harmless no-op.) The engine is a submodule,
# so it needs its own entry -- and ``apxinf_py``, the CUDA binding, must still be
# built separately for anything that loads weights.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT / "apxinf" / "python" / "apxinf"):
    if _entry.is_dir() and str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))


def json_object(value: str) -> Dict[str, Any]:
    """Parse a JSON object for model-specific policy options."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"invalid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def policy_kwargs(
    options: Mapping[str, Any],
    *,
    device: str,
    precision: str,
    action_dim: int = 0,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge generic CLI flags with concrete-policy options.

    Dedicated flags win over duplicate JSON keys. Caller metadata is preserved
    field by field, with the example's own metadata taking precedence.
    """
    kwargs = dict(options)
    if metadata is not None:
        caller_metadata = kwargs.pop("metadata", {})
        if not isinstance(caller_metadata, dict):
            raise ValueError("policy-options metadata must be a JSON object")
        kwargs["metadata"] = {**caller_metadata, **metadata}
    kwargs.update(device=device, precision=precision)
    if action_dim:
        kwargs["action_dim"] = action_dim
    return kwargs


def synthetic_observation(
    *,
    image_keys: Sequence[str],
    state_key: Optional[str] = None,
    prompt_key: str = "prompt",
    height: int = 256,
    width: int = 256,
    state_dim: int = 8,
    prompt: str = "pick up the block",
    seed: int = 0,
) -> Dict[str, Any]:
    """Build one raw observation dict shaped like the policy expects.

    Random ``uint8`` camera frames (raw ``HWC``; the policy's own resize step
    handles them) plus a float32 state vector and a text prompt. This only
    exercises the interface end to end -- the actions it yields are meaningless.

    ``image_keys`` has no default on purpose: camera wire keys are a client's
    convention, and every caller here can read the real ones off a preset or a
    policy. A helper that guessed them would drift from what is actually served.
    """
    rng = np.random.default_rng(seed)
    observation: Dict[str, Any] = {
        key: rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        for key in image_keys
    }
    if state_key is not None:
        observation[state_key] = rng.standard_normal(state_dim).astype(np.float32)
    observation[prompt_key] = prompt
    return observation


def observation_for_preset(preset, *, prompt: str = "pick up the block", seed: int = 0):
    """Build a synthetic observation from a preset's declared wire contract.

    No weights, no GPU: a ``RobotPreset`` names its cameras, its state key, and
    its state width up front, which is the whole point of having presets. Use
    :func:`observation_for_policy` once a policy is loaded -- that reads what is
    *actually* being served, including any ``--image-keys`` override.
    """
    state_key = preset.state_key if preset.state_dim else None
    return synthetic_observation(
        image_keys=preset.image_keys,
        state_key=state_key,
        prompt_key=preset.prompt_key,
        state_dim=preset.state_dim or 0,
        prompt=prompt,
        seed=seed,
    )


def observation_for_policy(policy, *, prompt: str = "pick up the block", seed: int = 0):
    """Build a synthetic observation using only the public ``Policy`` contract."""
    metadata = getattr(policy, "metadata", None) or {}
    image_keys = metadata.get("image_keys")
    if (
        not isinstance(image_keys, (list, tuple))
        or not image_keys
        or not all(isinstance(key, str) and key for key in image_keys)
    ):
        raise ValueError(
            f"policy metadata image_keys must be non-empty strings, got {image_keys!r}"
        )
    state_key = metadata.get("state_key")
    state_dim = metadata.get("state_dim")
    if state_key is not None and not isinstance(state_dim, int):
        raise ValueError("policy metadata must declare state_dim when state_key is set")
    return synthetic_observation(
        image_keys=tuple(image_keys),
        state_key=state_key,
        prompt_key=metadata.get("prompt_key", "prompt"),
        state_dim=state_dim or 0,
        prompt=prompt,
        seed=seed,
    )
