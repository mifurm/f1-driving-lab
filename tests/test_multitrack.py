import tempfile
import unittest
from pathlib import Path

import numpy as np

from circuits import CIRCUITS, HOLDOUT_CIRCUITS, TRAINING_CIRCUITS
from driving import Config, Environment, NeuralDriver, SCENARIOS, Track, simulate
from multitrack import MultiTrackEnvironment, balanced_indices, curriculum


class CircuitTests(unittest.TestCase):
    def test_ten_new_training_circuits_and_two_disjoint_holdouts(self):
        self.assertEqual(len(TRAINING_CIRCUITS), 11)
        self.assertEqual(len(HOLDOUT_CIRCUITS), 2)
        self.assertFalse({c.id for c in TRAINING_CIRCUITS} & {c.id for c in HOLDOUT_CIRCUITS})
        signatures = []
        for circuit in CIRCUITS:
            track = circuit.build(Config())
            signatures.append((track.n, track.xy.tobytes()))
            self.assertTrue(np.isfinite(track.curvature).all())
        self.assertEqual(len(set(signatures)), len(CIRCUITS))

    def test_teacher_can_drive_every_circuit_from_all_fixed_starts(self):
        config = Config(max_steps=2200)
        for circuit in CIRCUITS:
            track = circuit.build(config)
            for scenario in SCENARIOS:
                with self.subTest(track=circuit.name, scenario=scenario["name"]):
                    metrics, _ = simulate(track, config, None, scenario)
                    self.assertEqual(metrics["completed"], 1, metrics)

    def test_invalid_track_geometry_is_rejected(self):
        for knots in (
            [[0, 0], [20, 0], [20, 20], [0, 20], [1, 1]],
            [[0, 0], [20, 0], [20, 0], [0, 20], [0, 0]],
            [[0, 0], [20, 0], [20, np.nan], [0, 20], [0, 0]],
        ):
            with self.subTest(knots=knots):
                with self.assertRaises(ValueError):
                    Track(Config(), knots)

    def test_curvature_inputs_preserve_original_observations_and_checkpoint_shape(self):
        config = Config()
        track = CIRCUITS[3].build(config)
        environment = MultiTrackEnvironment(track, config)
        state = environment.make_state(0.28, lateral=1.2, heading_error=0.04)
        features, _ = environment.observe(state)
        original, _ = Environment(track, config).observe(state)
        np.testing.assert_array_equal(features[:13], original)
        self.assertEqual(features.shape, (16,))
        self.assertGreaterEqual(features[14], 0)
        driver = NeuralDriver(5, input_size=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "multi.npz"
            driver.save(path)
            np.testing.assert_array_equal(driver.predict(features), NeuralDriver.load(path).predict(features))

    def test_balanced_batches_retain_each_track_and_recent_examples(self):
        history = [np.arange(100), np.arange(100, 200), np.arange(200, 300)]
        recent = [pool[-10:] for pool in history]
        indices = balanced_indices(np.random.default_rng(7), history, recent, per_track=20)
        for index, pool in enumerate(history):
            selected = indices[index * 20:(index + 1) * 20]
            self.assertTrue(np.isin(selected, pool).all())
            self.assertTrue(np.isin(selected[10:], recent[index]).all())
        with self.assertRaises(ValueError):
            balanced_indices(np.random.default_rng(1), history, recent, per_track=3)

    def test_curriculum_changes_only_at_defined_boundaries(self):
        self.assertEqual(curriculum(40)["name"], "Foundation")
        self.assertEqual(curriculum(41)["name"], "Recovery")
        self.assertEqual(curriculum(80)["name"], "Recovery")
        self.assertEqual(curriculum(81)["name"], "Challenge")
        self.assertGreater(curriculum(120)["lateral"], curriculum(1)["lateral"])


if __name__ == "__main__":
    unittest.main()
