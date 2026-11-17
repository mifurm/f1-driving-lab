"""Teacher tool: build a fresh test pack of unseen circuits for the "Generalization and leakage" module.

The three test circuits that ship inside the lab are sealed by classroom protocol only: their labels are in the
page. A teacher who needs a test that no student can have inspected generates a pack from a seed kept private
and hands the JSON file out after students freeze their models. The seed is not written into the pack.

    python tools/make_fresh_pack.py --seed 20260601 --circuits 3 --output fresh_pack.json

Every candidate circuit passes the same strict geometry validation as the training circuits, and a circuit is
kept only if the expert completes all four of its sections, so a failure measures the student's model rather than
an impossible start. The pack's fingerprint identifies its contents (circuit geometry and sections).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from circuits import radial  # noqa: E402
from driving import Track  # noqa: E402
from student_experiments import export_track  # noqa: E402
from wall_sim import WALL_CONFIG, corner_scenarios, wall_simulate  # noqa: E402

SCHEMA = "f1-ml-lab-fresh-pack-v1"
FIRST_ID = 101
MAX_CANDIDATES = 400


def candidate_knots(rng):
    frequencies = rng.choice([2, 3, 4, 5], size=int(rng.integers(2, 4)), replace=False)
    harmonics = [(int(f), float(rng.uniform(0.04, 0.18)), float(rng.uniform(0, 2 * np.pi))) for f in frequencies]
    return radial(float(rng.uniform(230, 320)), float(rng.uniform(160, 250)), harmonics, count=60)


def expert_reference(track, scenarios):
    rows = []
    for scenario in scenarios:
        metric, _ = wall_simulate(track, WALL_CONFIG, None, scenario)
        rows.append({"scenario": scenario["name"], "success": metric["success"], "end_reason": metric["end_reason"],
                     "min_clearance_m": metric["min_clearance_m"]})
    return rows


def fingerprint(tracks, scenarios):
    canonical = json.dumps({"tracks": tracks, "scenarios": scenarios}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def make_pack(seed, circuits=3):
    if circuits < 1:
        raise ValueError("A pack needs at least one circuit")
    rng, tracks, scenarios, references = np.random.default_rng(seed), [], {}, {}
    rejected = {"geometry": 0, "expert_infeasible": 0}
    for _ in range(MAX_CANDIDATES):
        if len(tracks) == circuits:
            break
        try:
            track = Track(WALL_CONFIG, candidate_knots(rng), validate="error")
        except ValueError:
            rejected["geometry"] += 1
            continue
        sections = corner_scenarios(track)
        reference = expert_reference(track, sections)
        if not all(row["success"] for row in reference):
            rejected["expert_infeasible"] += 1
            continue
        identifier = FIRST_ID + len(tracks)
        tracks.append(export_track({"id": identifier, "name": f"Fresh circuit {len(tracks) + 1}", "split": "fresh_pack"}, track))
        scenarios[str(identifier)], references[str(identifier)] = sections, reference
    if len(tracks) < circuits:
        raise RuntimeError(f"Only {len(tracks)} of {circuits} circuits passed validation in {MAX_CANDIDATES} candidates")
    return {"schema": SCHEMA, "fingerprint": fingerprint(tracks, scenarios), "tracks": tracks, "scenarios": scenarios,
            "expert_reference": references,
            "provenance": {"generator": "tools/make_fresh_pack.py", "rejected_candidates": rejected,
                           "validation": "strict circuit geometry; expert completes every section"},
            "protocol": "Freeze a model, then evaluate it once on this pack. Training on these circuits makes the result invalid."}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, required=True, help="private seed; the same seed always gives the same pack")
    parser.add_argument("--circuits", type=int, default=3)
    parser.add_argument("--output", default="fresh_pack.json")
    arguments = parser.parse_args()
    pack = make_pack(arguments.seed, arguments.circuits)
    Path(arguments.output).write_text(json.dumps(pack, separators=(",", ":")))
    print(f"Wrote {arguments.output}: {len(pack['tracks'])} circuits, fingerprint {pack['fingerprint']}, "
          f"rejected candidates {pack['provenance']['rejected_candidates']}.")


if __name__ == "__main__":
    main()
