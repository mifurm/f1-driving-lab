import unittest

import numpy as np

from driving import NeuralDriver, SCENARIOS
from race_sim import (
    GEAR_RATIOS, RACE_CONFIG, RaceEnvironment, complex_circuits,
    engine_rpm, race_simulate,
)
from race_train import collect_rollout


class RaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.tracks = complex_circuits()

    def test_teacher_drives_harder_circuits_with_real_telemetry(self):
        for circuit in self.catalog:
            for scenario in SCENARIOS:
                with self.subTest(track=circuit["name"], start=scenario["name"]):
                    metric, trace = race_simulate(self.tracks[circuit["id"]], RACE_CONFIG, None, scenario)
                    self.assertEqual(metric["completed"], 1, metric)
                    self.assertEqual(trace.shape[1], 17)
                    self.assertTrue(np.isfinite(trace).all())
                    self.assertTrue((trace[:, 10] == np.round(trace[:, 10])).all())
                    self.assertGreater(metric["braking_time_s"], 0)
                    self.assertGreater(metric["gear_changes"], 0)

    def test_complexity_adds_corner_direction_changes(self):
        for circuit in self.catalog:
            with self.subTest(track=circuit["name"]):
                self.assertGreater(circuit["reversals"], circuit["previous_reversals"])
                self.assertEqual(len(circuit["added_sectors"]), 4)

    def test_brakes_decelerate_and_do_not_apply_throttle(self):
        environment = RaceEnvironment(self.tracks[1], RACE_CONFIG)
        environment.state = environment.make_state(0, speed=22)
        before = environment.state.speed
        environment.step(np.array([0, -1]))
        self.assertLess(environment.state.speed, before)
        self.assertEqual(environment.state.brake, 1)
        self.assertEqual(environment.state.throttle, 0)
        self.assertLess(environment.state.acceleration, 0)

    def test_gear_ratio_changes_acceleration_and_shift_interrupts_torque(self):
        accelerations = []
        for gear in (1, 8):
            environment = RaceEnvironment(self.tracks[1], RACE_CONFIG)
            environment.state = environment.make_state(0, speed=12)
            environment.state.gear = gear
            environment.state.rpm = engine_rpm(12, gear)
            environment.state.shift_cooldown = 1
            environment.step(np.array([0, 1]))
            accelerations.append(environment.state.acceleration)
        self.assertGreater(accelerations[0], accelerations[1])
        environment = RaceEnvironment(self.tracks[1], RACE_CONFIG)
        environment.state = environment.make_state(0, speed=12)
        environment.state.gear = 1
        environment.step(np.array([0, 1]))
        self.assertEqual(environment.state.gear, 2)
        self.assertTrue(environment.state.shift_active)
        self.assertLess(environment.state.acceleration, 0)
        self.assertEqual(len(GEAR_RATIOS), 8)

    def test_recorded_training_episode_does_not_join_reset_teleports(self):
        features, targets, trace, details = collect_rollout(
            self.tracks[2], RACE_CONFIG, NeuralDriver(31, input_size=16),
            np.random.default_rng(9), 1,
        )
        self.assertEqual(features.shape, (RACE_CONFIG.rollout_steps, 16))
        self.assertEqual(targets.shape, (RACE_CONFIG.rollout_steps, 2))
        self.assertEqual(details["model_round"], 0)
        self.assertAlmostEqual(details["duration_s"], (len(trace) - 1) * RACE_CONFIG.dt)
        distances = np.linalg.norm(np.diff(trace[:, 1:3], axis=0), axis=1)
        self.assertLessEqual(float(np.max(distances)), RACE_CONFIG.max_speed * RACE_CONFIG.dt + 1e-6)


if __name__ == "__main__":
    unittest.main()
