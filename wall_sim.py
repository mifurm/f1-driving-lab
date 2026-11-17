"""Corner recovery with oriented vehicle footprints and swept wall contact."""

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from driving import wrap_angle
from race_sim import RACE_CONFIG, TRACE_COLUMNS, RaceEnvironment, telemetry_row


WALL_CONFIG = replace(RACE_CONFIG, seed=2030, max_steps=350)
CAR_LENGTH = 4.8
WALL_COLUMNS = TRACE_COLUMNS + [
    "wall_clearance_m", "heading_error_rad", "section_progress_fraction",
    "left_center_clearance_m", "right_center_clearance_m", "wall_contact",
]


def add_margin_feature(features, config):
    features = np.asarray(features)
    margin = 1 - np.abs(features[..., 0]) - (
        CAR_LENGTH / 2 * np.abs(features[..., 1])
        + config.car_width / 2 * np.abs(features[..., 2])
    ) / (config.track_width / 2)
    budget = np.arctan(7.5 * config.wheelbase / np.maximum(
        (features[..., 3] * config.max_speed) ** 2, 1)) / config.max_steer
    return np.concatenate([features, margin[..., None], budget[..., None]], axis=-1)


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def point_segment_distance(points, starts, ends):
    delta = ends - starts
    weights = np.clip(np.sum((points - starts) * delta, axis=-1)
                      / np.maximum(np.sum(delta * delta, axis=-1), 1e-15), 0, 1)
    return np.linalg.norm(points - starts - weights[..., None] * delta, axis=-1)


def footprint(state, width=1.8):
    local = np.array([[CAR_LENGTH / 2, width / 2], [CAR_LENGTH / 2, -width / 2],
                      [-CAR_LENGTH / 2, -width / 2], [-CAR_LENGTH / 2, width / 2]])
    c, s = np.cos(state.heading), np.sin(state.heading)
    return local @ np.array([[c, s], [-s, c]]) + [state.x, state.y]


class Walls:
    def __init__(self, track):
        self.track = track
        self.tree = cKDTree(track.xy)
        self.offsets = np.arange(-7, 8)

    def clearance(self, state):
        polygon = footprint(state, self.track.config.car_width)
        index = int(self.tree.query([state.x, state.y])[1])
        indices = (index + self.offsets) % self.track.n
        starts = np.concatenate([self.track.left[indices], self.track.right[indices]])
        ends = np.concatenate([self.track.left[(indices + 1) % self.track.n],
                               self.track.right[(indices + 1) % self.track.n]])
        a, b = polygon[:, None], np.roll(polygon, -1, axis=0)[:, None]
        c, d = starts[None], ends[None]
        ab, cd = b - a, d - c
        denominator = cross(ab, cd)
        nonparallel = np.abs(denominator) > 1e-12
        safe_denominator = np.where(nonparallel, denominator, 1)
        t, u = cross(c - a, cd) / safe_denominator, cross(c - a, ab) / safe_denominator
        intersects = nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        distance = min(
            float(np.min(point_segment_distance(a, c, d))),
            float(np.min(point_segment_distance(c, a, b))),
            float(np.min(point_segment_distance(d, a, b))),
        )
        if intersects.any():
            return 0.0
        _, lateral, _ = self.track.locate(state.x, state.y)
        return -distance if abs(lateral) > self.track.config.track_width / 2 else distance


class WallEnvironment(RaceEnvironment):
    """No steering shield: the model's controls are applied unchanged."""

    def __init__(self, track, config):
        super().__init__(track, config)
        self.walls = Walls(track)
        self.contact = False
        self.swept_clearance = self.walls.clearance(self.state)

    def observe(self, state=None):
        features, location = super().observe(state)
        return add_margin_feature(features, self.config), location

    def teacher(self, state=None, location=None):
        state = self.state if state is None else state
        location = self.track.locate(state.x, state.y) if location is None else location
        action = super().teacher(state, location).copy()
        edge = abs(location[1])
        heading_error = float(wrap_angle(state.heading - location[2]))
        if edge > 3:
            lookahead = max(6.0, 12.0 - edge)
            delta = self.track.point(location[0] + lookahead) - [state.x, state.y]
            left = -np.sin(state.heading) * delta[0] + np.cos(state.heading) * delta[1]
            steer = np.arctan2(2 * self.config.wheelbase * left, np.dot(delta, delta))
            limit = np.arctan(7.5 * self.config.wheelbase / max(state.speed ** 2, 1))
            steer = np.clip(steer, -limit, limit)
            action[0] = np.clip(steer / self.config.max_steer, -1, 1)
            target = min((action[1] + 1) * self.config.max_speed / 2, max(6, 15 - edge))
            if location[1] * heading_error > 0:
                target = min(target, 7.0)
            action[1] = 2 * target / self.config.max_speed - 1
        return action

    def step(self, action):
        before = self.state
        location, reason, acceleration = super().step(action)
        after = self.state
        distance = np.hypot(after.x - before.x, after.y - before.y)
        angle = float(wrap_angle(after.heading - before.heading))
        count = max(1, int(np.ceil(max(distance / 0.5, abs(angle) / 0.025))))
        clearances = []
        for fraction in np.linspace(0, 1, count + 1):
            pose = replace(after, x=before.x + fraction * (after.x - before.x),
                           y=before.y + fraction * (after.y - before.y),
                           heading=before.heading + fraction * angle)
            clearances.append(self.walls.clearance(pose))
        self.swept_clearance = min(clearances)
        self.contact = self.swept_clearance <= 1e-8
        return location, "wall_contact" if self.contact else reason, acceleration


def corner_scenarios(track):
    peaks = [int(np.argmax(track.curvature)), int(np.argmin(track.curvature))]
    starts = [float((track.s[index] - 55) % track.length / track.length) for index in peaks]
    return [
        {"name": "Left-turn section", "fraction": starts[0], "lateral": 0, "heading_error": 0, "speed": 16},
        {"name": "Right-turn section", "fraction": starts[1], "lateral": 0, "heading_error": 0, "speed": 16},
        {"name": "Left-wall outward recovery", "fraction": starts[0], "lateral": 6.2, "heading_error": .26, "speed": 12},
        {"name": "Right-wall outward recovery", "fraction": starts[1], "lateral": -6.2, "heading_error": -.26, "speed": 12},
    ]


def wall_row(environment, elapsed, location, progress, goal):
    half_width = environment.config.track_width / 2
    return telemetry_row(environment, elapsed, location, progress) + [
        environment.swept_clearance,
        float(wrap_angle(environment.state.heading - location[2])),
        float(np.clip(progress / goal, 0, 1)),
        half_width - location[1], half_width + location[1], int(environment.contact),
    ]


def wall_simulate(track, config, driver, scenario, goal=150.0):
    environment = WallEnvironment(track, config)
    environment.state = environment.make_state(scenario["fraction"], scenario["lateral"],
                                               scenario["heading_error"], scenario["speed"])
    environment.swept_clearance = environment.walls.clearance(environment.state)
    environment.contact = environment.swept_clearance <= 1e-8
    _, location = environment.observe()
    previous, progress = location[0], 0.0
    trace = [wall_row(environment, 0, location, progress, goal)]
    reason = "wall_contact" if environment.contact else "timeout"
    if not environment.contact:
        for step in range(config.max_steps):
            features, location = environment.observe()
            action = environment.teacher(location=location) if driver is None else (
                driver(features) if callable(driver) else driver.predict(features))
            location, failure, _ = environment.step(action)
            progress += (location[0] - previous + track.length / 2) % track.length - track.length / 2
            previous = location[0]
            trace.append(wall_row(environment, (step + 1) * config.dt, location, progress, goal))
            if failure or progress >= goal:
                reason = failure or "section_complete"
                break
    trace = np.asarray(trace)
    tail = trace[-min(len(trace), 20):]
    centered = float(np.mean(np.abs(tail[:, 6]))) < 1.0
    success = reason == "section_complete" and centered
    return {
        "scenario": scenario["name"], "success": int(success),
        "section_complete": int(reason == "section_complete"), "centered_exit": bool(centered),
        "wall_contacts": int(np.any(trace[:, 22])), "end_reason": reason,
        "progress": float(np.clip(progress / goal, 0, 1)), "duration_s": float(trace[-1, 0]),
        "min_clearance_m": float(np.min(trace[:, 17])),
        "exit_mean_lateral_m": float(np.mean(np.abs(tail[:, 6]))),
        "mean_lateral_m": float(np.mean(np.abs(trace[:, 6]))),
        "peak_speed_kmh": float(np.max(trace[:, 4]) * 3.6),
        "braking_s": float(np.count_nonzero(trace[:, 9] > .01) * config.dt),
    }, trace
    feature_names = RaceEnvironment.feature_names + [
        "projected_footprint_wall_margin_half_width", "speed_dependent_steering_budget_fraction",
    ]
