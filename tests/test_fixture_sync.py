import json
import unittest

from tools.build_fixtures import FIXTURE, compute


def assert_equivalent(case, actual, wanted, path="fixture"):
    """Recursive comparison: exact for integers, strings and flags; 1e-9 for floats (the file stores 12 digits)."""
    if isinstance(wanted, dict):
        case.assertEqual(set(actual), set(wanted), path)
        for key in wanted:
            assert_equivalent(case, actual[key], wanted[key], f"{path}.{key}")
    elif isinstance(wanted, list):
        case.assertEqual(len(actual), len(wanted), f"{path} length")
        for index, (a, w) in enumerate(zip(actual, wanted)):
            assert_equivalent(case, a, w, f"{path}[{index}]")
    elif isinstance(wanted, bool) or isinstance(wanted, str) or wanted is None:
        case.assertEqual(actual, wanted, path)
    elif isinstance(wanted, int) and isinstance(actual, int):
        case.assertEqual(actual, wanted, path)
    else:
        case.assertAlmostEqual(float(actual), float(wanted), delta=1e-9 * max(1.0, abs(float(wanted))), msg=path)


class FixtureSyncTests(unittest.TestCase):
    def test_committed_engine_fixture_still_matches_the_python_simulator_and_numpy_references(self):
        stored = json.loads(FIXTURE.read_text())
        assert_equivalent(self, json.loads(json.dumps(compute(stored["inputs"]))), stored["expected"])

    def test_fixture_covers_every_failure_mode_the_engine_must_agree_on(self):
        reasons = {t["end_reason"] for t in json.loads(FIXTURE.read_text())["expected"]["trajectories"]}
        self.assertIn("section_complete", reasons)
        self.assertIn("lateral_acceleration_limit", reasons)
        labels = {t["label"] for t in json.loads(FIXTURE.read_text())["expected"]["trajectories"]}
        self.assertEqual(labels, {"expert", "rules", "pretrained", "delay 3", "grip 0.65", "noise 1"})


if __name__ == "__main__":
    unittest.main()
