"""Serve an ApxInf policy to RLinf's rollout workers as a ``BasePolicy``.

RLinf's embodied stack talks to a policy through exactly three things:

* ``rlinf.models.register_model(model_type, builder)`` -- a public registry, so
  adding a backend is a call, not a patch;
* ``rlinf.config.SupportedModel.register(model_type)`` -- a *second* public
  registry, holding the same names for a different purpose (see
  :func:`register_model_types`);
* ``BasePolicy.predict_action_batch(env_obs=..., mode=...) ->
  (actions, result)`` -- the abstract method every action model implements.

:class:`ApxInfActionModel` implements the last and :func:`get_model` is the
first's builder, so the RLinf-side change to run this is **six lines across two
files**; see :func:`register`. The config stays in this repository and is
reached with ``--config-path``; see ``scripts/rlinf_eval.sh``. Nothing in
RLinf's rollout worker, channel plumbing, or env scheduling has to know ApxInf
exists.

**Which axis this sits on.** RLinf has two: ``rollout.rollout_backend`` picks
the *worker* (the rollout loop and its transport) and ``rollout.model.model_type``
picks the *action model*. Only the second has a registry -- the first is a
hardcoded ``if``/``elif`` in ``evaluations/eval_embodied_agent.py`` -- so this
integration rides the ``huggingface`` worker and registers on the model axis.
ApxInf is an *engine* that hosts several models, which is why the registered
names are per-pair (``pi05_apxinf``, ``walloss_apxinf``) rather than a bare
``apxinf``: the slot names a model, so a model is what goes in it. See
:data:`APXINF_MODEL_TYPES`.

**Evaluation only.** ``mode="train"``, ``default_forward``, and every RL
bookkeeping hook raise :class:`NotImplementedError`. That is structural, not a
gap to fill later: the engine underneath is a Rust inference runtime with no
autograd, so there is no gradient to return. The returned ``result`` carries
``prev_logprobs=None`` / ``prev_values=None`` / ``forward_inputs={}``, which is
what the other action models return in eval mode.

**Batching.** ``Policy.infer`` takes one observation, so this adapter loops the
batch and stacks. RLinf's own openpi bridge does not have to: it builds one
``Observation`` across the whole batch and calls ``sample_actions`` once. The
difference is the engine's rather than the adapter's -- ApxInf batches *inside*
a single inference (views, flow steps), not across environments, so there is no
batched entry point underneath to forward to.

**The bridge itself** is one mapping, and it is the entire reason this file
exists. RLinf hands over positional view stacks (``main_images`` /
``wrist_images`` / ``extra_view_images``); an ApxInf policy wants named wire keys
from a robot preset. Slot *i* of the first goes to key *i* of the second --
which is the same slot order the checkpoint's weights were trained with.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

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

#: RLinf ``model_type`` -> the ApxInf policy that name selects.
#:
#: One entry per (model, engine) pair. RLinf's model axis names a *model*, and
#: ApxInf is an *engine* hosting several, so registering a bare ``apxinf`` would
#: put an engine in a model's slot and leave the second model nowhere to go.
#: RLinf's own sglang path solves this the same way -- one adapter per
#: ``cfg.model_type``, keyed in ``rlinf/models/embodiment/sglang_adapter.py``.
#:
#: Only ``pi05_apxinf`` has been evaluated end to end on LIBERO. ``walloss_apxinf``
#: runs the identical code path with a different ``AutoPolicy`` dispatch, so it
#: fails loudly at load time if a checkpoint and a preset disagree rather than
#: quietly returning worse actions.
APXINF_MODEL_TYPES = {
    "pi05_apxinf": "pi05",
    "walloss_apxinf": "walloss",
}

#: RLinf's env-observation view stacks, in the slot order they fill.
#: Matches ``rlinf.models.embodiment.evo1.utils.data_pipeline.build_evo1_inputs``.
RLINF_VIEW_KEYS = ("main_images", "wrist_images", "extra_view_images")
RLINF_STATE_KEY = "states"
RLINF_PROMPT_KEY = "task_descriptions"


def _base_policy():
    """RLinf's ``BasePolicy``, or ``object`` when RLinf is not installed.

    Subclassing matters in a real deployment (RLinf's workers reach for the
    inherited ``forward`` dispatch), but the observation bridge below is pure
    numpy and is worth testing without a multi-gigabyte RL framework present.
    Falling back keeps ``tests/test_rlinf_adapter.py`` runnable on a laptop.
    """
    try:
        from rlinf.models.embodiment.base_policy import BasePolicy
    except ImportError:  # pragma: no cover - depends on the environment
        return object
    return BasePolicy


def _to_numpy(value) -> np.ndarray:
    """Detach a torch tensor to numpy, or pass a numpy array through."""
    if isinstance(value, np.ndarray):
        return value
    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(value)


def _to_hwc_uint8(frame: np.ndarray, source_key: str) -> np.ndarray:
    """Normalize one frame to ``HWC`` ``uint8``, the form ApxInf's pipeline eats.

    RLinf's LIBERO envs already produce ``HWC`` ``uint8``, so this is a no-op on
    the common path. The conversions exist for the envs that do not: a ``CHW``
    layout is transposed, and float images in ``[0, 1]`` are scaled. Getting this
    wrong is silent -- the model still runs, it just sees garbage -- so it is
    checked here rather than assumed.
    """
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"{source_key}: expected a 3-D frame, got shape {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype == np.uint8:
        return array
    array = np.asarray(array, dtype=np.float32)
    scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
    return np.clip(array * scale, 0, 255).astype(np.uint8)


class ApxInfActionModel(_base_policy()):
    """An ApxInf policy wearing RLinf's ``BasePolicy`` interface.

    >>> model = ApxInfActionModel.from_pretrained(       # doctest: +SKIP
    ...     "/ckpt/pi05_libero", robot="franka_libero", num_action_chunks=8)
    >>> actions, result = model.predict_action_batch(env_obs=obs)  # doctest: +SKIP
    >>> actions.shape                                     # doctest: +SKIP
    (16, 8, 7)
    """

    def __init__(
        self,
        policy,
        *,
        image_keys: Sequence[str],
        state_key: Optional[str],
        prompt_key: str = "prompt",
        num_action_chunks: Optional[int] = None,
        action_dim: Optional[int] = None,
        view_keys: Sequence[str] = RLINF_VIEW_KEYS,
    ) -> None:
        self.policy = policy
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.prompt_key = prompt_key
        self.num_action_chunks = num_action_chunks
        self.action_dim = action_dim
        self.view_keys = tuple(view_keys)

    # --- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        robot: str = "franka_libero",
        model_type: Optional[str] = None,
        num_action_chunks: Optional[int] = None,
        action_dim: Optional[int] = None,
        view_keys: Sequence[str] = RLINF_VIEW_KEYS,
        **policy_kwargs: Any,
    ) -> "ApxInfActionModel":
        """Load ``model_dir`` under a robot preset and wrap it for RLinf.

        The preset decides the wire keys, the state routing and the deployable
        action width -- the same object ``apxinf-robo serve --robot`` uses, so a
        checkpoint evaluated here and a checkpoint served over the websocket are
        driven by one definition rather than two.

        ``model_type`` is *ApxInf's* (``pi05`` / ``walloss``), not RLinf's. It is
        named explicitly rather than left to ``**policy_kwargs`` because the two
        registries collide on the word: RLinf's ``model_type`` selects this
        builder, ApxInf's selects the policy class the builder loads. ``None``
        lets ``AutoPolicy`` sniff the checkpoint layout.
        """
        from ...presets import build_robot_policy, get_robot_preset

        preset = get_robot_preset(robot)
        policy = build_robot_policy(
            robot,
            model_dir,
            action_dim=action_dim,
            model_type=model_type,
            metadata={"integration": "rlinf"},
            **policy_kwargs,
        )
        return cls(
            policy,
            image_keys=policy.metadata.get("image_keys", preset.image_keys),
            state_key=policy.metadata.get("state_key", preset.state_key),
            prompt_key=preset.prompt_key,
            num_action_chunks=num_action_chunks,
            action_dim=action_dim if action_dim is not None else preset.action_dim,
            view_keys=view_keys,
        )

    # --- the bridge --------------------------------------------------------

    def build_observation(self, env_obs: Mapping[str, Any], index: int) -> Dict[str, Any]:
        """One RLinf batch element -> one ApxInf observation dict.

        Positional view stacks become named wire keys by slot order. A stack the
        preset has no key for is dropped, and a key with no stack behind it is an
        error: a missing camera degrades silently in the model (it just sees
        fewer patches) and would be found only as a mysterious success-rate drop.
        """
        stacks = [key for key in self.view_keys if env_obs.get(key) is not None]
        if len(stacks) < len(self.image_keys):
            raise ValueError(
                f"this policy serves {len(self.image_keys)} cameras "
                f"{list(self.image_keys)} but env_obs supplies {len(stacks)} view "
                f"stacks {stacks}; RLinf's view keys are {list(self.view_keys)}"
            )

        observation: Dict[str, Any] = {}
        for wire_key, source_key in zip(self.image_keys, stacks):
            frame = _to_numpy(env_obs[source_key])[index]
            _assign(observation, wire_key, _to_hwc_uint8(frame, source_key))

        if self.state_key is not None:
            state = env_obs.get(RLINF_STATE_KEY)
            if state is None:
                raise ValueError(
                    f"this policy reads state from {self.state_key!r} but env_obs has "
                    f"no {RLINF_STATE_KEY!r}; pass discrete_state=False to drop it"
                )
            _assign(observation, self.state_key, _to_numpy(state)[index].astype(np.float32))

        prompts = env_obs.get(RLINF_PROMPT_KEY)
        if prompts is None:
            raise ValueError(f"env_obs has no {RLINF_PROMPT_KEY!r}; the model needs a prompt")
        _assign(observation, self.prompt_key, str(prompts[index]))
        return observation

    def batch_size(self, env_obs: Mapping[str, Any]) -> int:
        for key in self.view_keys:
            value = env_obs.get(key)
            if value is not None:
                return int(_to_numpy(value).shape[0])
        raise ValueError(f"env_obs has none of the view keys {list(self.view_keys)}")

    # --- BasePolicy surface ------------------------------------------------

    def predict_action_batch(
        self,
        env_obs: Mapping[str, Any],
        mode: str = "eval",
        **kwargs: Any,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Env-space action chunks ``[B, chunk, dim]`` plus RLinf's result dict.

        ``kwargs`` (``calculate_logprobs`` / ``calculate_values`` / ``return_obs``)
        are accepted and ignored, matching the other eval-mode action models:
        RLinf's worker passes them unconditionally.
        """
        if mode != "eval":
            raise NotImplementedError(
                f"ApxInfActionModel is evaluation-only; got mode={mode!r}. The ApxInf "
                "engine is a Rust inference runtime with no autograd, so there is no "
                "training-mode rollout data (logprobs, denoising chain) to return. "
                "Use RLinf's own openpi/openvla backends to train, and this one to "
                "evaluate the exported checkpoint."
            )
        del kwargs

        chunks = []
        for index in range(self.batch_size(env_obs)):
            result = self.policy.infer(self.build_observation(env_obs, index))
            chunks.append(np.asarray(result["actions"], dtype=np.float32))

        actions = np.stack(chunks, axis=0)
        if self.num_action_chunks is not None:
            actions = actions[:, : self.num_action_chunks]
        return actions, {"prev_logprobs": None, "prev_values": None, "forward_inputs": {}}

    def default_forward(self, **kwargs: Any):
        raise NotImplementedError(
            "ApxInfActionModel has no forward pass: the engine underneath is a Rust "
            "inference runtime with no autograd. Only predict_action_batch(mode='eval') "
            "is supported."
        )

    # --- lifecycle ---------------------------------------------------------

    def eval(self) -> "ApxInfActionModel":
        """No-op, for call-shape compatibility. The engine is always in eval."""
        return self

    def to(self, *args: Any, **kwargs: Any) -> "ApxInfActionModel":
        """No-op. The device was chosen at load time and cannot be changed."""
        return self

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()


def _assign(observation: Dict[str, Any], key: str, value: Any) -> None:
    """Write ``value`` at a possibly nested slash path (``images/cam_high``)."""
    head, _, tail = key.partition("/")
    if not tail:
        observation[key] = value
        return
    node = observation.setdefault(head, {})
    _assign(node, tail, value)


def get_model(cfg, torch_dtype=None, *, model_type: Optional[str] = None) -> ApxInfActionModel:
    """RLinf model-registry builder for the names in :data:`APXINF_MODEL_TYPES`.

    Reads ``cfg.model_path`` plus an optional nested ``cfg.apxinf`` block::

        model:
          model_type: pi05_apxinf      # RLinf's axis: selects this builder
          model_path: /ckpt/pi05_libero
          apxinf:
            model_type: pi05           # ApxInf's axis: selects the policy class
            robot: franka_libero
            precision: bf16
            device: cuda:0
            num_action_chunks: 8

    The two ``model_type`` keys are different registries and the nesting is what
    keeps them apart. :func:`register` binds the ApxInf one per registered name,
    so the block key is an override and can be omitted; omitting both lets
    ``AutoPolicy`` sniff the checkpoint layout.

    ``torch_dtype`` is accepted for registry-signature compatibility and ignored:
    the engine's numeric mode is chosen by ``precision``, which selects a compiled
    kernel path (bf16 / fp8 / int8), not a torch dtype.
    """
    del torch_dtype

    model_path = getattr(cfg, "model_path", None)
    if not model_path:
        raise ValueError(
            f"model_type {getattr(cfg, 'model_type', '?')!r} requires cfg.model_path "
            "(a checkpoint dir)"
        )

    block = getattr(cfg, "apxinf", None) or {}
    get = block.get if isinstance(block, Mapping) else (lambda k, d=None: getattr(block, k, d))

    return ApxInfActionModel.from_pretrained(
        model_path,
        robot=get("robot", "franka_libero"),
        model_type=get("model_type", model_type),
        num_action_chunks=get("num_action_chunks", None),
        action_dim=get("action_dim", None),
        device=get("device", "cuda:0"),
        precision=get("precision", "bf16"),
        **(dict(get("policy_kwargs", {}) or {})),
    )


def register_model_types(
    *, model_types: Optional[Mapping[str, str]] = None, force: bool = False
) -> None:
    """Add ApxInf's model types to RLinf's *name* registry, ``SupportedModel``.

    Call this at import time from the **end of ``rlinf/config.py``**::

        from apxinf_robo.integrations.rlinf import register_model_types

        register_model_types(force=True)

    This is a different registry from the one :func:`register` writes to, and it
    needs its own hook because a different *process* reads it. ``SupportedModel``
    (``rlinf/config.py``) interns the legal ``model_type`` strings; the env
    worker looks ours up in ``rlinf/envs/action_utils.py`` to decide how to
    post-process a chunk before stepping the simulator, and an unregistered name
    raises ``NotImplementedError`` there -- *after* the policy has loaded and
    produced its first actions, which is a confusing place to find out.

    ``rlinf/models/__init__.py`` imports ``rlinf.config``, but the env worker
    imports only ``rlinf.envs.action_utils`` and so never reaches
    ``rlinf.models``. Registering both names from the models hook therefore
    populates the rollout process and leaves the env process without them.
    ``rlinf.config`` is the module both import, and it is deliberately the light
    one -- which is why this half is split out rather than folded into
    :func:`register`, whose ``from rlinf.models import ...`` would drag the whole
    torch model zoo into every env worker.

    For LIBERO the registration is all that is needed: ``prepare_actions_for_libero``
    rewrites the gripper axis only for the OpenVLA family and passes everything
    else -- ``openpi`` included -- through untouched, which is the right contract
    for a PI0.5 checkpoint that already emits env-space actions.
    """
    from rlinf.config import SupportedModel

    table = APXINF_MODEL_TYPES if model_types is None else dict(model_types)
    for rlinf_model_type in table:
        SupportedModel.register(rlinf_model_type, force=force)


def register(*, model_types: Optional[Mapping[str, str]] = None, force: bool = False) -> None:
    """Add ApxInf's model types to RLinf's *builder* registry, and to its names.

    Call this at *import time* from ``rlinf/models/__init__.py``, after RLinf's
    own ``_register_builtin_models()``::

        from apxinf_robo.integrations.rlinf import register as _register_apxinf

        _register_apxinf(force=True)

    Together with the :func:`register_model_types` call in ``rlinf/config.py``
    that is the whole RLinf-side code change: six lines, two files, one per
    registry. Both hooks are needed and neither subsumes the other -- this one
    reaches the rollout worker, which builds the policy; that one reaches the env
    worker, which never imports ``rlinf.models``. It is called here as well so
    that a rollout-only process is correct after a single import.

    It has to be that file rather than the launcher. ``_MODEL_REGISTRY`` is
    module-global state, and RLinf's rollout and env workers are separate Ray
    processes that import ``rlinf.models`` fresh. Registering from ``main()``
    populates the driver -- enough for ``validate_cfg`` to accept the name -- and
    leaves the worker that actually calls ``get_model`` with the built-in
    registry only, so the run fails downstream with a null model rather than an
    unknown model type.

    ``model_types`` defaults to :data:`APXINF_MODEL_TYPES`; pass a mapping to
    register a different set of RLinf-name -> ApxInf-policy pairs.
    """
    from rlinf.models import register_model

    register_model_types(model_types=model_types, force=force)

    table = APXINF_MODEL_TYPES if model_types is None else dict(model_types)
    for rlinf_model_type, apxinf_model_type in table.items():
        register_model(
            rlinf_model_type,
            partial(get_model, model_type=apxinf_model_type),
            force=force,
        )
