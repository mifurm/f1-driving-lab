import tempfile
import unittest
from pathlib import Path

import numpy as np

from driving import Config, Environment, NeuralDriver, SCENARIOS, Track, simulate


class DrivingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.track = Track(cls.config)

    def test_teacher_completes_all_fixed_starts(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario["name"]):
                metrics, trace = simulate(self.track, self.config, None, scenario)
                self.assertEqual(metrics["completed"], 1, metrics)
                self.assertTrue(np.isfinite(trace).all())

    def test_off_track_and_lateral_budget_are_enforced(self):
        environment = Environment(self.track, self.config)
        environment.state = environment.make_state(0, lateral=10, speed=0)
        self.assertEqual(environment.step(np.array([0, -1]))[1], "off_track")
        environment.state = environment.make_state(0, speed=36)
        self.assertEqual(environment.step(np.array([1, 1]))[1], "lateral_acceleration_limit")

    def test_progress_crosses_start_finish(self):
        scenario = dict(SCENARIOS[0], fraction=0.995)
        metrics, trace = simulate(self.track, self.config, None, scenario)
        self.assertEqual(metrics["completed"], 1, metrics)
        self.assertGreater(trace[-1, 7], 0.99)
        self.assertGreaterEqual(float(np.min(np.diff(trace[:, 7]))), -0.001)

    def test_weight_updates_learn_and_checkpoint_preserves_prediction(self):
        rng = np.random.default_rng(5)
        features = rng.normal(0, 0.4, (256, 13))
        targets = np.tanh(np.column_stack([
            features[:, 0] * 0.2, features[:, 3] * 0.3 - 0.1,
        ]))
        driver = NeuralDriver(7)
        before = np.mean((driver.predict(features) - targets) ** 2)
        for _ in range(100):
            driver.train_batch(features, targets, 0.002)
        after = np.mean((driver.predict(features) - targets) ** 2)
        self.assertLess(after, before / 10)
        self.assertEqual(driver.updates, 100)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            driver.save(path)
            restored = NeuralDriver.load(path)
            np.testing.assert_array_equal(driver.predict(features), restored.predict(features))

    def test_observation_and_teacher_are_finite_at_seam(self):
        environment = Environment(self.track, self.config)
        for fraction in (0, 0.99999, 0.3):
            state = environment.make_state(fraction, lateral=-2, heading_error=0.1)
            features, location = environment.observe(state)
            self.assertEqual(features.shape, (13,))
            self.assertTrue(np.isfinite(features).all())
            action = environment.teacher(state, location)
            self.assertTrue(np.isfinite(action).all())
            self.assertTrue((np.abs(action) <= 1).all())


if __name__ == "__main__":
    unittest.main()
