import unittest

import numpy as np

from tools.make_fresh_pack import SCHEMA, make_pack
from wall_sim import WALL_CONFIG


class FreshPackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pack = make_pack(seed=7, circuits=2)

    def test_pack_is_reproducible_from_its_private_seed_and_does_not_leak_it(self):
        again = make_pack(seed=7, circuits=2)
        self.assertEqual(again["fingerprint"], self.pack["fingerprint"])
        self.assertNotEqual(make_pack(seed=8, circuits=1)["fingerprint"], self.pack["fingerprint"])
        self.assertNotIn('"seed"', str(self.pack))

    def test_every_circuit_is_strictly_valid_and_drivable_by_the_expert(self):
        self.assertEqual(self.pack["schema"], SCHEMA)
        self.assertEqual(len(self.pack["tracks"]), 2)
        for item in self.pack["tracks"]:
            self.assertGreaterEqual(item["id"], 101)
            for key in ("xy", "left", "right", "tangent", "normal", "heading", "curvature"):
                self.assertEqual(len(item[key]), len(item["xy"]), key)
            self.assertEqual(len(self.pack["scenarios"][str(item["id"])]), 4)
            self.assertTrue(all(row["success"] for row in self.pack["expert_reference"][str(item["id"])]))
            # The strict rule: the tightest bend is wider than half the road, so the inner wall cannot fold over.
            self.assertLess(np.max(np.abs(item["curvature"])), 2 / WALL_CONFIG.track_width)

    def test_a_pack_cannot_be_empty(self):
        with self.assertRaises(ValueError):
            make_pack(seed=1, circuits=0)


if __name__ == "__main__":
    unittest.main()
