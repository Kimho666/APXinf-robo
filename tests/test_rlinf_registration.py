"""The two functions that touch RLinf's API, without RLinf.

``tests/test_rlinf_adapter.py`` covers the observation bridge -- the part that
runs once per env step and decides whether the success rate is right. This file
covers the part that runs *once*, at process start, and decides whether there is
a model at all: :func:`get_model` reading a config block, and :func:`register`
putting builders into RLinf's registry.

They are worth their own tests because they are the only two places that touch a
foreign contract. ``get_model``'s signature is dictated by
``rlinf.models._MODEL_REGISTRY`` (called as ``builder(cfg, torch_dtype)``), and a
wrong key in the ``apxinf:`` block does not raise -- it silently falls back to a
default, which means the wrong robot preset, the wrong precision, or a checkpoint
loaded onto the wrong device, none of which announce themselves.

RLinf itself is stubbed. ``register`` needs exactly two symbols from it --
``rlinf.models.register_model`` and ``rlinf.config.SupportedModel`` -- so faking
those is faithful rather than approximate.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from apxinf_robo.integrations.rlinf import (
    APXINF_MODEL_TYPES,
    adapter,
    get_model,
    register,
    register_model_types,
)

MODEL_PATH = "/ckpt/pi05_libero"
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = "configs/rlinf/libero_10_apxinf_robo_pi05_eval.yaml"


class Namespace:
    """An attribute-style config node, the shape OmegaConf hands the builder."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def cfg(*, apxinf=None, model_path=MODEL_PATH, model_type="pi05_apxinf", **extra):
    fields = {"model_path": model_path, "model_type": model_type, **extra}
    if apxinf is not None:
        fields["apxinf"] = apxinf
    return Namespace(**fields)


@pytest.fixture
def loaded(monkeypatch):
    """Record what ``from_pretrained`` was asked for, without loading anything."""
    calls = []

    def record(model_dir, **kwargs):
        calls.append({"model_dir": model_dir, **kwargs})
        return f"policy for {model_dir}"

    monkeypatch.setattr(adapter.ApxInfActionModel, "from_pretrained", record)
    return calls


@pytest.fixture
def registered(monkeypatch):
    """Stand in for RLinf's two registries and record what lands in each.

    ``rlinf.models.register_model`` holds builders and ``rlinf.config.SupportedModel``
    interns legal ``model_type`` strings. They live in different modules on
    purpose -- the env worker imports only the second -- so they are faked as two
    modules rather than one, and the fixture returns both ledgers.
    """
    entries = []
    names = []

    models = types.ModuleType("rlinf.models")
    models.register_model = lambda name, builder, force=False: entries.append(
        {"name": name, "builder": builder, "force": force}
    )

    class SupportedModel:
        @classmethod
        def register(cls, value, force=False):
            names.append({"name": value, "force": force})

    config = types.ModuleType("rlinf.config")
    config.SupportedModel = SupportedModel

    package = types.ModuleType("rlinf")
    package.models = models
    package.config = config
    monkeypatch.setitem(sys.modules, "rlinf", package)
    monkeypatch.setitem(sys.modules, "rlinf.models", models)
    monkeypatch.setitem(sys.modules, "rlinf.config", config)
    return types.SimpleNamespace(builders=entries, names=names)


# --- get_model: reading the config block --------------------------------------


def test_a_checkpointless_config_fails_loudly_rather_than_at_the_first_frame(loaded):
    with pytest.raises(ValueError, match="model_path"):
        get_model(cfg(model_path=None))


def test_the_error_names_the_model_type_that_was_selected(loaded):
    with pytest.raises(ValueError, match="walloss_apxinf"):
        get_model(cfg(model_path="", model_type="walloss_apxinf"))


def test_every_block_key_reaches_its_own_argument(loaded):
    get_model(
        cfg(
            apxinf={
                "robot": "unitree_g1",
                "model_type": "walloss",
                "device": "cuda:3",
                "precision": "fp8",
                "num_action_chunks": 4,
                "action_dim": 14,
            }
        )
    )
    assert loaded == [
        {
            "model_dir": MODEL_PATH,
            "robot": "unitree_g1",
            "model_type": "walloss",
            "device": "cuda:3",
            "precision": "fp8",
            "num_action_chunks": 4,
            "action_dim": 14,
        }
    ]


def test_an_attribute_style_block_reads_the_same_as_a_mapping(loaded):
    get_model(cfg(apxinf=Namespace(robot="unitree_g1", precision="fp8")))
    assert loaded[0]["robot"] == "unitree_g1"
    assert loaded[0]["precision"] == "fp8"


def test_the_shipped_config_drives_the_builder_as_omegaconf_hands_it_over(loaded):
    # The two tests above cover the branches; this one covers the real type. A
    # DictConfig is a Mapping *and* attribute-addressable, so either branch would
    # look fine in isolation while the shipped YAML silently took the other one.
    # It also pins the `policy_kwargs` the config sets, `norm_stats` above all:
    # that one is interpolated off `model_path`, so it is the only entry here
    # that breaks if the interpolation is moved or misspelled.
    omegaconf = pytest.importorskip("omegaconf")
    config = omegaconf.OmegaConf.load(REPO_ROOT / CONFIG)
    get_model(config.rollout.model)
    assert loaded[0] == {
        "model_dir": "/path/to/model/RLinf-Pi05-LIBERO-SFT",
        "robot": "franka_libero",
        "model_type": "pi05",
        "device": "cuda:0",
        "precision": "bf16",
        "num_action_chunks": 5,
        "action_dim": 7,
        "num_flow_steps": 5,
        "action_horizon": 10,
        "flow_start_time": 1.0,
        "seed": 0,
        "norm_stats": "/path/to/model/RLinf-Pi05-LIBERO-SFT/norm_stats.json",
    }


def test_a_missing_block_still_loads_under_the_documented_defaults(loaded):
    get_model(cfg())
    assert loaded[0]["robot"] == "franka_libero"
    assert loaded[0]["precision"] == "bf16"
    assert loaded[0]["device"] == "cuda:0"


def test_policy_kwargs_expand_rather_than_arriving_as_one_argument(loaded):
    get_model(cfg(apxinf={"policy_kwargs": {"num_flow_steps": 10, "norm_stats": "/n.json"}}))
    assert loaded[0]["num_flow_steps"] == 10
    assert loaded[0]["norm_stats"] == "/n.json"
    assert "policy_kwargs" not in loaded[0]


def test_the_registry_passes_a_torch_dtype_and_it_is_ignored(loaded):
    # RLinf calls builders as ``builder(cfg, torch_dtype)``. The engine's numeric
    # mode is a compiled kernel path chosen by ``precision``, not a torch dtype.
    get_model(cfg(apxinf={"precision": "bf16"}), "torch.float32")
    assert loaded[0]["precision"] == "bf16"


# --- the two model_type registries stay apart ---------------------------------


def test_the_registered_apxinf_model_type_reaches_the_policy(loaded):
    get_model(cfg(), model_type="pi05")
    assert loaded[0]["model_type"] == "pi05"


def test_the_block_overrides_what_registration_bound(loaded):
    get_model(cfg(apxinf={"model_type": "walloss"}), model_type="pi05")
    assert loaded[0]["model_type"] == "walloss"


def test_naming_neither_leaves_the_checkpoint_to_speak_for_itself(loaded):
    get_model(cfg())
    assert loaded[0]["model_type"] is None


# --- the imports the builder makes on the way to the engine -------------------


def test_from_pretrained_can_reach_presets_without_the_engine_installed():
    # Every other test in this file monkeypatches ``from_pretrained``, so the
    # deferred ``from ...presets import ...`` inside it never runs and a wrong
    # number of dots stays invisible until a GPU box tries to load a checkpoint.
    # Resolving the module by name is enough: it is the level count that breaks,
    # not the contents.
    import importlib

    presets = importlib.import_module("apxinf_robo.presets")
    assert hasattr(presets, "build_robot_policy")
    assert hasattr(presets, "get_robot_preset")
    assert adapter.__name__.startswith("apxinf_robo.integrations.rlinf")


# --- register: what lands in RLinf's two registries ---------------------------


def test_every_pair_is_registered_under_its_rlinf_name(registered):
    register()
    assert [entry["name"] for entry in registered.builders] == list(APXINF_MODEL_TYPES)


def test_each_registered_builder_is_bound_to_its_own_policy(registered, loaded):
    register()
    for entry in registered.builders:
        entry["builder"](cfg(), None)
    assert [call["model_type"] for call in loaded] == list(APXINF_MODEL_TYPES.values())


def test_force_reaches_rlinf_so_a_reimport_is_not_a_duplicate_name_error(registered):
    register(force=True)
    assert all(entry["force"] for entry in registered.builders)
    assert all(entry["force"] for entry in registered.names)


def test_a_caller_supplied_table_replaces_the_built_in_pairs(registered, loaded):
    register(model_types={"my_pi05": "pi05"})
    assert [entry["name"] for entry in registered.builders] == ["my_pi05"]
    assert [entry["name"] for entry in registered.names] == ["my_pi05"]
    registered.builders[0]["builder"](cfg(), None)
    assert loaded[0]["model_type"] == "pi05"


def test_the_registered_names_say_engine_as_well_as_model():
    # The slot names a model and ApxInf is an engine hosting several, so a bare
    # "apxinf" would claim the whole axis and leave the second model nowhere.
    assert set(APXINF_MODEL_TYPES) == {"pi05_apxinf", "walloss_apxinf"}
    assert "apxinf" not in APXINF_MODEL_TYPES


# --- the name registry, which a different process reads -----------------------


def test_the_names_also_reach_supported_model_not_only_the_builder_registry(registered):
    # Found end to end on Thor: with the builder registered and the name not,
    # the run loads the checkpoint, infers a chunk, and only then dies in
    # `prepare_actions_for_libero` -- one Ray process later than the mistake.
    register()
    assert [entry["name"] for entry in registered.names] == list(APXINF_MODEL_TYPES)


def test_the_name_registry_can_be_populated_without_importing_rlinf_models(monkeypatch):
    # This is the whole reason it is a separate function. The env worker imports
    # `rlinf.config` and never `rlinf.models`, so the light half has to work with
    # the heavy half absent -- otherwise the documented `rlinf/config.py` hook
    # would drag the torch model zoo into every env process.
    names = []

    class SupportedModel:
        @classmethod
        def register(cls, value, force=False):
            names.append(value)

    config = types.ModuleType("rlinf.config")
    config.SupportedModel = SupportedModel
    package = types.ModuleType("rlinf")
    package.config = config
    monkeypatch.setitem(sys.modules, "rlinf", package)
    monkeypatch.setitem(sys.modules, "rlinf.config", config)
    monkeypatch.setitem(sys.modules, "rlinf.models", None)  # importing it would raise

    register_model_types(force=True)
    assert names == list(APXINF_MODEL_TYPES)
