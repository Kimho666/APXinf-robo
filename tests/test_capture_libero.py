"""``capture-libero``: the seam between a simulator and the engine's calibrator.

ApxInf builds an FP8 profile from observations *on disk*. This command is what
puts them there. Two things are worth holding still:

* the frames are **task-balanced** -- a small ``--samples`` covers the whole
  suite rather than over-sampling whichever task is enumerated first, because a
  profile calibrated on one task's lighting is a profile that clips on the rest;
* the NPZ field names come from a **robot preset**, so the calibration input is
  by construction the same wire dialect the checkpoint is later served with.

The simulator itself is stubbed: these run offline, with no LIBERO and no MuJoCo.
"""

from __future__ import annotations

import pathlib
import unittest
from unittest import mock

import numpy as np

from apxinf_robo.cli import capture_libero
from apxinf_robo.envs.libero import libero_images, libero_state


class _Task:
    def __init__(self, task_id):
        self.language = f"task {task_id}"


class _Suite:
    n_tasks = 2

    def get_task(self, task_id):
        return _Task(task_id)

    def get_task_init_states(self, task_id):
        return np.asarray([[task_id, 0], [task_id, 1]], dtype=np.float32)


class _Env:
    def __init__(self):
        self.value = 0
        self.closed = False
        self.steps = 0
        self.last_raw = None

    def reset(self):
        pass

    def set_init_state(self, initial_state):
        self.value = int(initial_state[0] * 10 + initial_state[1])
        return self._observation()

    def step(self, _action):
        self.steps += 1
        return self._observation(), 0.0, False, {}

    def _observation(self):
        # Deliberately not uniform: a frame of one value hides an orientation bug,
        # since flipping it is a no-op.
        ramp = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        self.last_raw = {
            "agentview_image": (ramp + self.value).astype(np.uint8),
            "robot0_eye_in_hand_image": (ramp + self.value + 20).astype(np.uint8),
            "robot0_eef_pos": np.arange(3, dtype=np.float32),
            "robot0_eef_quat": np.asarray([0, 0, 0, 1], np.float32),
            "robot0_gripper_qpos": np.asarray([0.1, 0.2], np.float32),
        }
        return self.last_raw

    def close(self):
        self.closed = True


def _stub_simulator(envs=None):
    """Patch out LIBERO itself, keeping the selection and conversion under test."""
    envs = envs if envs is not None else []

    def make_env(*_args):
        env = _Env()
        envs.append(env)
        return env

    return (
        mock.patch.object(capture_libero, "load_suite", return_value=_Suite()),
        mock.patch.object(capture_libero, "make_env", side_effect=make_env),
    )


def _capture(**kwargs):
    suite_patch, env_patch = _stub_simulator(kwargs.pop("envs", None))
    with suite_patch, env_patch:
        return capture_libero.capture_observations(
            "libero_10",
            image_keys=kwargs.pop("image_keys", ("observation/image", "observation/wrist_image")),
            prompt_key=kwargs.pop("prompt_key", "prompt"),
            state_key=kwargs.pop("state_key", "observation/state"),
            sample_count=kwargs.pop("sample_count", 2),
            seed=kwargs.pop("seed", 7),
            **kwargs,
        )


class TaskBalanceTest(unittest.TestCase):
    def test_every_task_is_covered_before_any_is_repeated(self):
        indices = capture_libero.task_stratified_indices(
            [0, 0, 0, 1, 1, 2], sample_count=5, seed=7
        )

        self.assertEqual(len(indices), 5)
        selected = [[0, 0, 0, 1, 1, 2][index] for index in indices]
        self.assertEqual(set(selected[:3]), {0, 1, 2})
        self.assertLessEqual(max(selected.count(task) for task in set(selected)), 2)

    def test_the_same_seed_selects_the_same_frames(self):
        first = capture_libero.task_stratified_indices(
            [0, 0, 0, 1, 1, 2], sample_count=4, seed=3
        )
        again = capture_libero.task_stratified_indices(
            [0, 0, 0, 1, 1, 2], sample_count=4, seed=3
        )
        self.assertEqual(first, again)

    def test_too_few_samples_to_cover_the_suite_is_an_error_not_a_silent_subset(self):
        # Silently calibrating on 1 of 3 tasks is the failure this guards: the
        # profile still writes, and only clips later on the tasks it never saw.
        with self.assertRaisesRegex(ValueError, "cannot cover all 3 tasks"):
            capture_libero.task_stratified_indices(
                [0, 0, 1, 1, 2, 2], sample_count=2, seed=0
            )

    def test_asking_for_more_frames_than_exist_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "from 3 frames"):
            capture_libero.task_stratified_indices([0, 1, 2], sample_count=4, seed=0)

    def test_a_non_positive_sample_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            capture_libero.task_stratified_indices([0, 1], sample_count=0, seed=0)


class CaptureTest(unittest.TestCase):
    def test_capture_is_balanced_across_tasks(self):
        observations = _capture()

        self.assertEqual(len(observations), 2)
        self.assertEqual(
            {observation["prompt"] for observation in observations},
            {"task 0", "task 1"},
        )

    def test_observations_use_the_conversion_evaluation_uses(self):
        # Not a re-implementation: the same envs.libero helpers, so a change to
        # camera orientation or state layout moves calibration and evaluation
        # together or fails here.
        envs: list[_Env] = []
        observations = _capture(sample_count=2, envs=envs)
        raw = envs[0].last_raw
        expected = libero_images(
            raw["agentview_image"], raw["robot0_eye_in_hand_image"]
        )

        np.testing.assert_array_equal(observations[0]["observation/image"], expected[0])
        np.testing.assert_array_equal(
            observations[0]["observation/wrist_image"], expected[1]
        )
        np.testing.assert_array_equal(
            observations[0]["observation/state"], libero_state(raw)
        )
        self.assertEqual(observations[0]["observation/state"].dtype, np.float32)

    def test_the_frames_are_taken_after_the_settle_window(self):
        envs: list[_Env] = []
        _capture(sample_count=2, envs=envs)

        self.assertTrue(envs, "expected at least one environment")
        for env in envs:
            self.assertEqual(env.steps, capture_libero.WAIT_STEPS)

    def test_every_environment_is_closed(self):
        envs: list[_Env] = []
        _capture(sample_count=2, envs=envs)

        self.assertTrue(all(env.closed for env in envs))

    def test_wire_keys_are_whatever_the_caller_names(self):
        observations = _capture(
            sample_count=2,
            image_keys=("images/cam_high", "images/cam_wrist"),
            prompt_key="task",
            state_key="state",
        )

        self.assertEqual(
            set(observations[0]),
            {"images/cam_high", "images/cam_wrist", "state", "task"},
        )

    def test_a_convention_without_exactly_two_cameras_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            _capture(image_keys=("observation/image",))


class WriteTest(unittest.TestCase):
    def _observations(self, count):
        return [
            {
                "observation/image": np.full((2, 2, 3), index, np.uint8),
                "observation/state": np.zeros(8, np.float32),
                "prompt": f"task {index}",
            }
            for index in range(count)
        ]

    def test_filenames_sort_in_capture_order(self):
        # calibrate_pi05 --input-dir sorts the glob, so unpadded names would put
        # sample 10 before sample 2 and change the data identity between runs.
        with _tmpdir() as root:
            written = capture_libero.write_npz_observations(
                self._observations(12), root, prefix="libero"
            )

            self.assertEqual(
                [path.name for path in sorted(root.glob("*.npz"))],
                [path.name for path in written],
            )

    def test_a_written_observation_round_trips_through_numpy(self):
        with _tmpdir() as root:
            capture_libero.write_npz_observations(
                self._observations(1), root, prefix="libero"
            )
            with np.load(next(root.glob("*.npz")), allow_pickle=False) as sample:
                self.assertEqual(sample["prompt"].item(), "task 0")
                self.assertEqual(sample["observation/image"].shape, (2, 2, 3))
                self.assertEqual(sample["observation/state"].dtype, np.float32)

    def test_an_occupied_directory_is_refused_without_force(self):
        # The calibrator reads the *whole* directory, so mixing two captures
        # silently calibrates on a set nobody selected.
        with _tmpdir() as root:
            capture_libero.write_npz_observations(
                self._observations(2), root, prefix="libero"
            )
            with self.assertRaisesRegex(ValueError, "--force"):
                capture_libero.write_npz_observations(
                    self._observations(2), root, prefix="libero"
                )

    def test_force_replaces_the_previous_capture_rather_than_adding_to_it(self):
        with _tmpdir() as root:
            capture_libero.write_npz_observations(
                self._observations(5), root, prefix="libero"
            )
            capture_libero.write_npz_observations(
                self._observations(2), root, prefix="libero", force=True
            )

            self.assertEqual(len(sorted(root.glob("*.npz"))), 2)


class ArgumentTest(unittest.TestCase):
    def test_defaults_name_the_libero_preset(self):
        args = capture_libero.parse_args(["--output-dir", "/tmp/out"])

        self.assertEqual(args.robot, capture_libero.LIBERO_PRESET)
        self.assertEqual(args.suite, "libero_10")
        self.assertIsNone(args.samples)

    def test_a_non_positive_sample_count_is_rejected(self):
        with self.assertRaises(SystemExit):
            capture_libero.parse_args(["--output-dir", "/tmp/out", "--samples", "0"])

    def test_run_names_the_npz_fields_after_the_preset(self):
        from apxinf_robo.presets import get_robot_preset

        convention = get_robot_preset(capture_libero.LIBERO_PRESET).convention
        captured = {}

        def fake_capture(suite, **kwargs):
            captured.update(suite=suite, **kwargs)
            return ()

        with _tmpdir() as root, mock.patch.object(
            capture_libero, "capture_observations", side_effect=fake_capture
        ):
            capture_libero.run(
                capture_libero.parse_args(["--output-dir", str(root), "--samples", "4"])
            )

        self.assertEqual(captured["suite"], "libero_10")
        self.assertEqual(tuple(captured["image_keys"]), tuple(convention.image_keys))
        self.assertEqual(captured["state_key"], convention.state_key)
        self.assertEqual(captured["prompt_key"], convention.prompt_key)
        self.assertEqual(captured["sample_count"], 4)


class _tmpdir:
    def __init__(self):
        import tempfile

        self._directory = tempfile.TemporaryDirectory()

    def __enter__(self) -> pathlib.Path:
        return pathlib.Path(self._directory.name)

    def __exit__(self, *_exc):
        self._directory.cleanup()
        return False


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
