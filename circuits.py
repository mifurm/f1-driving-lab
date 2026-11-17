"""Original circuit catalog: eleven training layouts and two unseen test layouts."""

from dataclasses import dataclass

import numpy as np

from driving import LEGACY_CIRCUIT_IDS, Track


def close(points):
    return np.vstack([np.asarray(points, dtype=float), points[0]])


def radial(x_radius, y_radius, harmonics, count=24):
    angle = np.linspace(0, 2 * np.pi, count, endpoint=False)
    radius = np.ones(count)
    for frequency, amplitude, phase in harmonics:
        radius += amplitude * np.cos(frequency * angle + phase)
    return close(np.column_stack([
        x_radius * radius * np.cos(angle), y_radius * radius * np.sin(angle),
    ]))


@dataclass(frozen=True)
class Circuit:
    id: int
    name: str
    style: str
    split: str
    knots: np.ndarray | None

    def build(self, config):
        return Track(config, self.knots, validate="report" if self.id in LEGACY_CIRCUIT_IDS else "error")


CIRCUITS = [
    Circuit(0, "Apex Park", "Original technical circuit", "train", None),
    Circuit(1, "Velocity Ring", "Fast oval with broad bends", "train",
            radial(270, 145, [])),
    Circuit(2, "Chicane Coast", "Alternating bends and long exits", "train", close([
        [0, -140], [90, -140], [140, -100], [190, -145], [265, -100],
        [300, -20], [230, 65], [250, 160], [140, 215], [0, 175],
        [-105, 205], [-230, 110], [-250, -40], [-150, -140],
    ])),
    Circuit(3, "Hairpin Harbor", "Tight turnarounds and asymmetric straights", "train", close([
        [0, -180], [200, -180], [270, -100], [235, -20], [130, 5],
        [160, 85], [235, 150], [200, 230], [80, 250], [-70, 190],
        [-200, 245], [-265, 150], [-195, 40], [-240, -90], [-120, -180],
    ])),
    Circuit(4, "Serpentine Valley", "Repeated changes in corner direction", "train",
            radial(265, 200, [(3, 0.19, 0.3), (5, 0.07, 1.1)])),
    Circuit(5, "Triangle GP", "Three long sectors and rounded apexes", "train", close([
        [0, -190], [95, -30], [210, 150], [110, 195],
        [-110, 195], [-210, 150], [-95, -30],
    ])),
    Circuit(6, "Square Sprint", "Long straights joined by short corners", "train", close([
        [200, -130], [260, -80], [260, 80], [200, 130],
        [-200, 130], [-260, 80], [-260, -80], [-200, -130],
    ])),
    Circuit(7, "Double Apex", "Two lobes with a narrow flowing middle", "train",
            radial(255, 205, [(2, 0.30, 0), (3, 0.07, 0.6)])),
    Circuit(8, "Crescent Run", "Off-center sweep and changing-radius bends", "train",
            radial(240, 220, [(1, 0.23, 0.5), (2, 0.14, 1.5)])),
    Circuit(9, "Technical Orchard", "Four lobes and frequent braking zones", "train",
            radial(265, 245, [(4, 0.13, 0.4), (3, 0.07, 1.7)])),
    Circuit(10, "High-Speed Mesa", "Extended fast sector and uneven end turns", "train",
            radial(330, 155, [(2, 0.09, 1.0), (3, 0.08, 2.0)])),
    Circuit(11, "Unseen Aurora", "Unseen mixture of three- and four-lobe bends", "holdout",
            radial(250, 225, [(3, 0.15, 2.3), (4, 0.065, 0.8)])),
    Circuit(12, "Unseen Delta", "Unseen asymmetric elongated technical loop", "holdout",
            radial(295, 185, [(2, 0.20, 2.2), (5, 0.055, 1.3), (1, 0.05, 0.7)])),
]
TRAINING_CIRCUITS = [circuit for circuit in CIRCUITS if circuit.split == "train"]
HOLDOUT_CIRCUITS = [circuit for circuit in CIRCUITS if circuit.split == "holdout"]
