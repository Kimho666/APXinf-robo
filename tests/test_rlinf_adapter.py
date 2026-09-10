"""The RLinf bridge, without RLinf.

The whole integration reduces to one mapping — RLinf's *positional* view stacks
(``main_images`` / ``wrist_images`` / ``extra_view_images``) become a preset's
*named* wire keys, slot by slot. That mapping is where a deployment goes wrong
quietly: swap two cameras and the model still returns well-shaped actions, just
worse ones, and the only symptom is a success rate nobody can explain.

So these tests drive :class:`ApxInfActionModel` against a recording stub policy.
No RLinf, no checkpoint, no CUDA — ``_base_policy()`` falls back to ``object``
when RLinf is absent, which is exactly what makes this file runnable.
"""

from __future__ import annotations

import numpy as np
import pytest

from apxinf_robo.integrations.rlinf import (
    RLINF_PROMPT_KEY,
    RLINF_STATE_KEY,
    RLINF_VIEW_KEYS,
    ApxInfActionModel,
)

BATCH = 3
CHUNK = 8
ACTION_DIM = 7
IMAGE = 4  # tiny frames; the bridge never looks at pixel content


class RecordingPolicy:
    """Stands in for an ApxInf policy: records observations, returns ramps."""

    def __init__(self, horizon: int = 16, action_dim: int = ACTION_DIM) -> None:
        self.observations: list = []
        self.horizon = horizon
        self.action_dim = action_dim
        self.metadata: dict = {}
        self.closed = False

    def infer(self, observation):
        self.observations.append(observation)
        index = len(self.observations) - 1
        actions = np.full((self.horizon, self.action_dim), float(index), np.float32)
        return {"actions": actions}

    def close(self) -> None:
        self.closed = True


def env_obs(
    *,
    batch: int = BATCH,
    views: int = 2,
    state_dim: int = 8,
    prompts=None,
    dtype=np.uint8,
):
    """An RLinf-shaped observation batch: ``[B, H, W, C]`` stacks plus state."""
    obs = {}
    for slot in range(views):
        frame = np.full((batch, IMAGE, IMAGE, 3), slot + 1, dtype=np.uint8)
        obs[RLINF_VIEW_KEYS[slot]] = frame.astype(dtype)
    obs[RLINF_STATE_KEY] = np.arange(batch * state_dim, dtype=np.float64).reshape(
        batch, state_dim
    )
    obs[RLINF_PROMPT_KEY] = prompts or [f"task {i}" for i in range(batch)]
    return obs


def model(policy=None, **kwargs) -> ApxInfActionModel:
    return ApxInfActionModel(
        policy if policy is not None else RecordingPolicy(),
        image_keys=kwargs.pop("image_keys", ("observation/image", "observation/wrist_image")),
        state_key=kwargs.pop("state_key", "observation/state"),
        **kwargs,
    )


# --- the mapping --------------------------------------------------------------


def test_view_stacks_fill_wire_keys_in_slot_order():
    # Slot order is baked into the checkpoint's weights: main -> key 0, wrist ->
    # key 1. Getting this backwards is the silent failure this test exists for.
    observation = model().build_observation(env_obs(), index=0)

    assert set(observation) == {"observation", "prompt"}
    assert np.all(observation["observation"]["image"] == 1)  # main_images
    assert np.all(observation["observation"]["wrist_image"] == 2)  # wrist_images


def test_the_batch_element_is_the_one_selected():
    obs = env_obs()
    obs[RLINF_VIEW_KEYS[0]] = np.stack(
        [np.full((IMAGE, IMAGE, 3), 10 + i, np.uint8) for i in range(BATCH)]
    )

    observation = model().build_observation(obs, index=2)

    assert np.all(observation["observation"]["image"] == 12)
    assert observation["prompt"] == "task 2"
    np.testing.assert_allclose(
        observation["observation"]["state"], obs[RLINF_STATE_KEY][2]
    )
    assert observation["observation"]["state"].dtype == np.float32


def test_a_third_camera_is_dropped_when_the_preset_serves_two():
    obs = env_obs(views=3)

    observation = model().build_observation(obs, index=0)

    # extra_view_images has no key to land in; it must not overwrite one either.
    assert np.all(observation["observation"]["image"] == 1)
    assert np.all(observation["observation"]["wrist_image"] == 2)


def test_a_missing_camera_is_an_error_rather_than_a_smaller_observation():
    # A policy that silently sees one camera instead of two still runs; it just
    # gets worse. That is only ever found as an unexplained success-rate drop.
    obs = env_obs(views=1)

    with pytest.raises(ValueError, match="2 cameras"):
        model().build_observation(obs, index=0)


def test_a_missing_prompt_is_an_error():
    obs = env_obs()
    del obs[RLINF_PROMPT_KEY]

    with pytest.raises(ValueError, match=RLINF_PROMPT_KEY):
        model().build_observation(obs, index=0)


def test_a_missing_state_is_an_error_only_when_the_policy_reads_state():
    obs = env_obs()
    del obs[RLINF_STATE_KEY]

    with pytest.raises(ValueError, match=RLINF_STATE_KEY):
        model().build_observation(obs, index=0)

    stateless = model(state_key=None).build_observation(obs, index=0)
    assert "state" not in stateless["observation"]


def test_flat_wire_keys_stay_flat():
    observation = model(
        image_keys=("base_0_rgb", "left_wrist_0_rgb"), state_key="state"
    ).build_observation(env_obs(), index=0)

    assert set(observation) == {"base_0_rgb", "left_wrist_0_rgb", "state", "prompt"}


# --- frame normalization ------------------------------------------------------


def test_chw_frames_are_transposed():
    obs = env_obs()
    obs[RLINF_VIEW_KEYS[0]] = np.zeros((BATCH, 3, IMAGE, IMAGE), np.uint8)

    frame = model().build_observation(obs, index=0)["observation"]["image"]

    assert frame.shape == (IMAGE, IMAGE, 3)


def test_unit_range_float_frames_are_scaled_to_bytes():
    obs = env_obs()
    obs[RLINF_VIEW_KEYS[0]] = np.full((BATCH, IMAGE, IMAGE, 3), 0.5, np.float32)

    frame = model().build_observation(obs, index=0)["observation"]["image"]

    assert frame.dtype == np.uint8
    assert np.all(frame == 127)


def test_byte_range_float_frames_are_not_scaled_again():
    obs = env_obs()
    obs[RLINF_VIEW_KEYS[0]] = np.full((BATCH, IMAGE, IMAGE, 3), 200.0, np.float32)

    frame = model().build_observation(obs, index=0)["observation"]["image"]

    assert np.all(frame == 200)


def test_torch_style_tensors_are_detached():
    class FakeTensor:
        def __init__(self, array):
            self._array = array

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._array

    obs = env_obs()
    obs[RLINF_VIEW_KEYS[0]] = FakeTensor(obs[RLINF_VIEW_KEYS[0]])
    obs[RLINF_STATE_KEY] = FakeTensor(obs[RLINF_STATE_KEY])

    observation = model().build_observation(obs, index=1)

    assert observation["observation"]["image"].dtype == np.uint8


# --- the BasePolicy surface ---------------------------------------------------


def test_predict_action_batch_returns_one_chunk_per_environment():
    policy = RecordingPolicy()
    actions, result = model(policy, num_action_chunks=CHUNK).predict_action_batch(
        env_obs=env_obs()
    )

    assert actions.shape == (BATCH, CHUNK, ACTION_DIM)
    assert actions.dtype == np.float32
    # One infer per environment, in batch order.
    assert len(policy.observations) == BATCH
    assert [obs["prompt"] for obs in policy.observations] == ["task 0", "task 1", "task 2"]
    # Chunk i came from infer call i, not from a reshuffle.
    for index in range(BATCH):
        assert np.all(actions[index] == index)
    assert result == {"prev_logprobs": None, "prev_values": None, "forward_inputs": {}}


def test_an_unset_chunk_length_returns_the_policys_full_horizon():
    policy = RecordingPolicy(horizon=16)

    actions, _ = model(policy).predict_action_batch(env_obs=env_obs())

    assert actions.shape == (BATCH, 16, ACTION_DIM)


def test_rlinf_worker_kwargs_are_accepted_and_ignored():
    # RLinf's rollout worker passes these unconditionally; refusing them would
    # break the call site rather than surface anything useful.
    actions, _ = model(num_action_chunks=CHUNK).predict_action_batch(
        env_obs=env_obs(), mode="eval", return_obs=True, calculate_logprobs=True
    )
    assert actions.shape == (BATCH, CHUNK, ACTION_DIM)


def test_training_mode_fails_loudly_rather_than_returning_eval_data():
    # The engine is a Rust inference runtime with no autograd. Returning
    # eval-shaped data under mode="train" would give an RL loop zero-signal
    # gradients that look like a converging run.
    with pytest.raises(NotImplementedError, match="evaluation-only"):
        model().predict_action_batch(env_obs=env_obs(), mode="train")


def test_there_is_no_forward_pass():
    with pytest.raises(NotImplementedError, match="no forward pass"):
        model().default_forward()


def test_batch_size_reads_whichever_view_stack_is_present():
    obs = env_obs(batch=5)
    del obs[RLINF_VIEW_KEYS[0]]
    obs[RLINF_VIEW_KEYS[0]] = obs.pop(RLINF_VIEW_KEYS[1])

    assert model().batch_size(obs) == 5

    with pytest.raises(ValueError, match="view keys"):
        model().batch_size({RLINF_PROMPT_KEY: ["x"]})


def test_lifecycle_hooks_are_no_ops_that_still_close_the_policy():
    policy = RecordingPolicy()
    adapter = model(policy)

    assert adapter.eval() is adapter
    assert adapter.to("cuda:1") is adapter
    adapter.close()

    assert policy.closed
