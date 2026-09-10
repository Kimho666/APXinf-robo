"""RLinf integration: an ApxInf policy as an RLinf embodied action model.

See :mod:`.adapter`. The RLinf-side change is six lines across two files, one
per registry RLinf keeps. At the end of ``rlinf/models/__init__.py``, after
RLinf's own ``_register_builtin_models()``::

    from apxinf_robo.integrations.rlinf import register as _register_apxinf

    _register_apxinf(force=True)

and at the end of ``rlinf/config.py``::

    from apxinf_robo.integrations.rlinf import register_model_types

    register_model_types(force=True)

Both are needed. The first puts the builder where the rollout worker looks; the
second puts the *name* where the env worker looks, and the env worker never
imports ``rlinf.models``. Without it the run loads the checkpoint, infers, and
then raises ``NotImplementedError`` on the first env step.

The config stays in *this* repository -- ``configs/rlinf/`` -- and is reached
with hydra's ``--config-path``, so it is versioned with the adapter it
configures rather than copied into someone else's tree::

    RLINF=/path/to/RLinf scripts/rlinf_eval.sh libero_10_apxinf_robo_pi05_eval \\
        rollout.model.model_path=/ckpt/pi05_libero
"""

from __future__ import annotations

from .adapter import (
    APXINF_MODEL_TYPES,
    RLINF_PROMPT_KEY,
    RLINF_STATE_KEY,
    RLINF_VIEW_KEYS,
    ApxInfActionModel,
    get_model,
    register,
    register_model_types,
)

__all__ = [
    "APXINF_MODEL_TYPES",
    "ApxInfActionModel",
    "RLINF_VIEW_KEYS",
    "RLINF_STATE_KEY",
    "RLINF_PROMPT_KEY",
    "get_model",
    "register",
    "register_model_types",
]
