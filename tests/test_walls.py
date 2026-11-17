import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from race_sim import RaceEnvironment
from wall_sim import WALL_CONFIG, WallEnvironment, add_margin_feature, corner_scenarios, footprint, wall_simulate
from race_sim import complex_circuits


class WallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cls.tracks = complex_circuits(WALL_CONFIG)

    def test_rotated_front_corner_contacts_wall_with_center_inside(self):
        environment = WallEnvironment(self.tracks[1], WALL_CONFIG)
        state = environment.make_state(.05, lateral=7.7, heading_error=.6, speed=8)
        self.assertLess(abs(environment.track.locate(state.x, state.y)[1]), 9)
        self.assertLessEqual(environment.walls.clearance(state), 1e-8)
        self.assertEqual(footprint(state).shape, (4, 2))

    def test_swept_contact_is_not_only_an_endpoint_check(self):
        environment = WallEnvironment(self.tracks[1], WALL_CONFIG)
        environment.state = environment.make_state(.05, lateral=7.6, heading_error=.2, speed=12)
        _, reason, _ = environment.step(np.array([0, 0]))
        if reason != "wall_contact":
            _, reason, _ = environment.step(np.array([0, 0]))
        self.assertEqual(reason, "wall_contact")
        self.assertLessEqual(environment.swept_clearance, 1e-8)

    def test_rotation_sweep_detects_contact_between_clear_endpoints(self):
        environment = WallEnvironment(self.tracks[1], WALL_CONFIG)
        before = environment.make_state(.05, lateral=7.3, speed=0)
        after = replace(before, heading=before.heading + np.pi)
        environment.state = before
        self.assertGreater(environment.walls.clearance(before), 0)
        self.assertGreater(environment.walls.clearance(after), 0)
        def rotate(instance, action):
            instance.state = after
            return instance.track.locate(after.x, after.y), "", 0
        with patch.object(RaceEnvironment, "step", rotate):
            _, reason, _ = environment.step(np.array([0, -1]))
        self.assertEqual(reason, "wall_contact")

    def test_teacher_can_recover_each_side_and_take_all_corners(self):
        for item in self.catalog:
            for scenario in corner_scenarios(self.tracks[item["id"]]):
                with self.subTest(track=item["name"], scenario=scenario["name"]):
                    metric, trace = wall_simulate(self.tracks[item["id"]], WALL_CONFIG, None, scenario)
                    self.assertEqual(metric["success"], 1, metric)
                    self.assertEqual(trace.shape[1], 23)
                    self.assertGreater(metric["min_clearance_m"], 0)

    def test_margin_feature_tracks_heading_and_edge_risk(self):
        environment = WallEnvironment(self.tracks[1], WALL_CONFIG)
        center, _ = environment.observe(environment.make_state(.05))
        edge, _ = environment.observe(environment.make_state(.05, lateral=6, heading_error=.3))
        fast, _ = environment.observe(environment.make_state(.05, speed=24))
        self.assertEqual(len(center), 18)
        self.assertLess(edge[-2], center[-2])
        self.assertLess(fast[-1], center[-1])
        np.testing.assert_allclose(add_margin_feature(edge[:16], WALL_CONFIG), edge)


if __name__ == "__main__":
    unittest.main()
