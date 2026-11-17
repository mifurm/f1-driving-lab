import math
import unittest

import numpy as np

from circuits import CIRCUITS
from driving import (
    Config, Environment, LEARNING_RATE_SCHEDULES, MAX_LEARNING_RATE, NUMERICAL_DIVERGENCE,
    NeuralDriver, SCENARIOS, Track, learning_rate_at, simulate, tracks_from_catalog,
)
from multitrack import MultiTrackEnvironment
from race_sim import RACE_CONFIG, RaceEnvironment, complex_circuits, race_simulate
from wall_sim import WALL_CONFIG, WallEnvironment, corner_scenarios, wall_simulate


def reference_adam(driver, features, targets, rate):
    """The original update rule (no decay, no clipping) written out independently."""
    activations = [features]
    for weight, bias in zip(driver.weights, driver.biases):
        activations.append(np.tanh(activations[-1] @ weight + bias))
    importance = np.array([4.0, 1.0])
    gradient = (activations[-1] - targets) * importance / len(features)
    gradients = []
    for layer in range(2, -1, -1):
        gradient = gradient * (1 - activations[layer + 1] ** 2)
        gradients[0:0] = [activations[layer].T @ gradient, gradient.sum(axis=0)]
        gradient = gradient @ driver.weights[layer].T
    driver.updates += 1
    for index, (parameter, derivative) in enumerate(zip(driver.parameters, gradients)):
        driver.first[index] = 0.9 * driver.first[index] + 0.1 * derivative
        driver.second[index] = 0.999 * driver.second[index] + 0.001 * derivative ** 2
        first = driver.first[index] / (1 - 0.9 ** driver.updates)
        second = driver.second[index] / (1 - 0.999 ** driver.updates)
        parameter -= rate * first / (np.sqrt(second) + 1e-8)


class DivergenceGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        cls.track = Track(cls.config)

    def test_non_finite_actions_end_the_drive_without_corrupting_state(self):
        cases = [
            (Environment(self.track, self.config), self.config),
            (RaceEnvironment(self.track, RACE_CONFIG), RACE_CONFIG),
            (WallEnvironment(self.track, WALL_CONFIG), WALL_CONFIG),
        ]
        for environment, _ in cases:
            for action in ([np.nan, 0.0], [0.0, np.inf], [-np.inf, np.nan]):
                with self.subTest(environment=type(environment).__name__, action=action):
                    environment.state = environment.make_state(0.1, speed=12)
                    before = (environment.state.x, environment.state.y, environment.state.speed)
                    location, reason, acceleration = environment.step(np.array(action))
                    self.assertEqual(reason, NUMERICAL_DIVERGENCE)
                    self.assertEqual(acceleration, 0.0)
                    self.assertTrue(np.isfinite(location).all())
                    self.assertEqual(before, (environment.state.x, environment.state.y, environment.state.speed))

    def test_wrongly_shaped_actions_are_programming_errors(self):
        environment = Environment(self.track, self.config)
        for action in ([0.0], [0.0, 0.0, 0.0], [[0.0, 0.0]]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                environment.step(action)

    def test_diverged_network_is_reported_as_divergence_not_timeout_with_finite_metrics(self):
        driver = NeuralDriver(1)
        driver.weights[0][...] = np.nan
        metrics, trace = simulate(self.track, self.config, driver, SCENARIOS[0])
        self.assertEqual(metrics["end_reason"], NUMERICAL_DIVERGENCE)
        self.assertEqual(metrics["completed"], 0)
        self.assertTrue(np.isfinite(trace).all())
        self.assertTrue(np.isfinite(metrics["progress"]))
        racing, race_trace = race_simulate(self.track, RACE_CONFIG, _NanDriver(), SCENARIOS[0])
        self.assertEqual(racing["end_reason"], NUMERICAL_DIVERGENCE)
        self.assertTrue(np.isfinite(race_trace).all())
        wall, wall_trace = wall_simulate(self.track, WALL_CONFIG, _NanDriver(), corner_scenarios(self.track)[0])
        self.assertEqual(wall["end_reason"], NUMERICAL_DIVERGENCE)
        self.assertEqual(wall["success"], 0)
        self.assertTrue(np.isfinite(wall_trace).all())


class _NanDriver:
    def predict(self, features):
        return np.array([np.nan, np.nan])


class TrainingGuardTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.features = rng.normal(0, 0.5, (64, 13))
        self.targets = np.tanh(rng.normal(0, 0.5, (64, 2)))

    def test_default_update_matches_the_original_adam_rule(self):
        driver, reference = NeuralDriver(3), NeuralDriver(3)
        for _ in range(6):
            driver.train_batch(self.features, self.targets, 0.002)
            reference_adam(reference, self.features, self.targets, 0.002)
        for ours, theirs in zip(driver.parameters, reference.parameters):
            np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-14)
        self.assertEqual(driver.updates, 6)

    def test_invalid_hyper_parameters_are_rejected_before_any_update(self):
        driver = NeuralDriver(3)
        before = [item.copy() for item in driver.parameters]
        for rate in (0, -0.001, math.nan, math.inf, MAX_LEARNING_RATE * 1.01):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                driver.train_batch(self.features, self.targets, rate)
        for kwargs in ({"weight_decay": -0.1}, {"weight_decay": 1.5}, {"weight_decay": math.nan},
                       {"max_grad_norm": 0}, {"max_grad_norm": -1.0}, {"max_grad_norm": math.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                driver.train_batch(self.features, self.targets, 0.001, **kwargs)
        bad = self.features.copy()
        bad[2, 3] = np.nan
        with self.assertRaises(ValueError):
            driver.train_batch(bad, self.targets, 0.001)
        self.assertEqual(driver.updates, 0)
        for parameter, saved in zip(driver.parameters, before):
            np.testing.assert_array_equal(parameter, saved)

    def test_non_finite_loss_raises_and_leaves_parameters_unchanged(self):
        driver = NeuralDriver(3)
        driver.weights[2][...] = np.inf
        before = [item.copy() for item in driver.parameters]
        with self.assertRaises(FloatingPointError):
            driver.train_batch(self.features, self.targets, 0.001)
        self.assertEqual(driver.updates, 0)
        for parameter, saved in zip(driver.parameters, before):
            np.testing.assert_array_equal(parameter, saved)

    def test_weight_decay_shrinks_weights_exactly_and_never_touches_biases(self):
        plain, decayed = NeuralDriver(5), NeuralDriver(5)
        start = [item.copy() for item in plain.parameters]
        plain.train_batch(self.features, self.targets, 0.004)
        decayed.train_batch(self.features, self.targets, 0.004, weight_decay=0.25)
        for index, (a, b, first) in enumerate(zip(plain.parameters, decayed.parameters, start)):
            if index % 2 == 0:
                np.testing.assert_allclose(b, a - 0.004 * 0.25 * first, rtol=0, atol=1e-15)
            else:
                np.testing.assert_array_equal(a, b)

    def test_gradient_clipping_rescales_to_the_requested_global_norm(self):
        free, clipped = NeuralDriver(5), NeuralDriver(5)
        free.train_batch(self.features, self.targets, 0.001)
        self.assertFalse(free.last_clipped)
        norm = free.last_gradient_norm
        self.assertGreater(norm, 0)
        limit = norm / 4
        clipped.train_batch(self.features, self.targets, 0.001, max_grad_norm=limit)
        self.assertTrue(clipped.last_clipped)
        self.assertAlmostEqual(clipped.last_gradient_norm, norm, places=12)
        first_moment_norm = math.sqrt(sum(float(np.sum(item ** 2)) for item in clipped.first))
        self.assertAlmostEqual(first_moment_norm, 0.1 * limit, places=12)
        loose = NeuralDriver(5)
        loose.train_batch(self.features, self.targets, 0.001, max_grad_norm=norm * 2)
        self.assertFalse(loose.last_clipped)
        for a, b in zip(free.parameters, loose.parameters):
            np.testing.assert_array_equal(a, b)

    def test_learning_rate_schedules(self):
        self.assertEqual(set(LEARNING_RATE_SCHEDULES), {"constant", "linear", "cosine", "step"})
        for schedule in LEARNING_RATE_SCHEDULES:
            first, last = learning_rate_at(schedule, 0.01, 1, 100), learning_rate_at(schedule, 0.01, 100, 100)
            self.assertAlmostEqual(first, 0.01)
            if schedule == "constant":
                self.assertEqual(last, 0.01)
            else:
                self.assertLess(last, first)
                rates = [learning_rate_at(schedule, 0.01, u, 100) for u in range(1, 101)]
                self.assertTrue(all(b <= a + 1e-15 for a, b in zip(rates, rates[1:])), schedule)
        self.assertAlmostEqual(learning_rate_at("linear", 1.0, 100, 100), 0.1)
        self.assertAlmostEqual(learning_rate_at("cosine", 1.0, 100, 100), 0.1)
        self.assertAlmostEqual(learning_rate_at("step", 1.0, 40, 100), 0.5)
        self.assertAlmostEqual(learning_rate_at("step", 1.0, 100, 100), 0.25)
        self.assertEqual(learning_rate_at("linear", 0.5, 1, 1), 0.5)
        with self.assertRaises(ValueError):
            learning_rate_at("warmup", 1.0, 1, 10)


class CircuitValidationTests(unittest.TestCase):
    @staticmethod
    def figure_eight():
        angle = np.linspace(0, 2 * np.pi, 41)
        knots = np.column_stack([200 * np.sin(angle), 120 * np.sin(2 * angle)])
        knots[-1] = knots[0]
        return knots

    def test_degenerate_layouts_are_rejected(self):
        layouts = {
            "figure eight": self.figure_eight(),
            "hairpin tighter than the road": np.array([[0, 0], [100, 0], [100, 6], [0, 6], [0, 0]], float),
            "parallel straights overlap": np.array([[0, 0], [200, 0], [200, 10], [0, 10], [0, 0]], float),
        }
        for name, knots in layouts.items():
            with self.subTest(layout=name), self.assertRaisesRegex(ValueError, "Invalid circuit geometry"):
                Track(Config(), knots)

    def test_report_mode_keeps_the_geometry_and_lists_every_problem(self):
        track = Track(Config(), self.figure_eight(), validate="report")
        self.assertTrue(track.geometry["problems"])
        self.assertGreater(track.geometry["edge_crossings"], 0)
        with self.assertRaises(ValueError):
            Track(Config(), self.figure_eight(), validate="strict")

    def test_tight_but_valid_circuit_is_accepted(self):
        angle = np.linspace(0, 2 * np.pi, 25)
        ring = np.column_stack([60 * np.cos(angle), 60 * np.sin(angle)])
        ring[-1] = ring[0]
        track = Track(Config(), ring)
        self.assertEqual(track.geometry["problems"], [])
        self.assertAlmostEqual(track.geometry["min_radius_m"], 60, delta=1.5)

    def test_exactly_the_known_legacy_circuits_have_inner_wall_folds(self):
        folded = {circuit.name for circuit in CIRCUITS if circuit.build(Config()).geometry["problems"]}
        self.assertEqual(folded, {"Apex Park", "Square Sprint"})
        catalog, tracks = complex_circuits()
        gpx_folded = {item["name"] for item in catalog if tracks[item["id"]].geometry["problems"]}
        self.assertEqual(gpx_folded, {"Apex Park GP-X", "Square Sprint GP-X"})
        for circuit in CIRCUITS:
            geometry = circuit.build(Config()).geometry
            if geometry["problems"]:
                self.assertEqual(geometry["edge_crossings"] > 0, True)
                self.assertLess(geometry["min_radius_m"], Config().track_width / 2)

    def test_catalog_helper_validates_new_circuits_but_tolerates_the_legacy_ids(self):
        knots = self.figure_eight().tolist()
        with self.assertRaises(ValueError):
            tracks_from_catalog(Config(), [{"id": 13, "knots": knots}])
        tracks = tracks_from_catalog(Config(), [{"id": 5, "knots": knots}])
        self.assertTrue(tracks[5].geometry["problems"])


class FeatureNameTests(unittest.TestCase):
    def test_every_environment_names_exactly_the_inputs_it_produces(self):
        track = Track(Config())
        for environment in (Environment(track, Config()), MultiTrackEnvironment(track, Config()),
                            RaceEnvironment(track, RACE_CONFIG), WallEnvironment(track, WALL_CONFIG)):
            with self.subTest(environment=type(environment).__name__):
                features, _ = environment.observe()
                self.assertEqual(len(environment.feature_names), len(features))
                self.assertEqual(len(set(environment.feature_names)), len(features))
        self.assertEqual(WallEnvironment.feature_names[-2:], [
            "projected_footprint_wall_margin_half_width", "speed_dependent_steering_budget_fraction"])


if __name__ == "__main__":
    unittest.main()
