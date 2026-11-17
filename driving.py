"""Deterministic fictional driving environment and NumPy imitation learner."""

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

NUMERICAL_DIVERGENCE = "numerical_divergence"
MAX_LEARNING_RATE = 0.05
LEARNING_RATE_SCHEDULES = ("constant", "linear", "cosine", "step")


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


SAMPLE_SNAP = 1e-9


def sample_index(position, n):
    """Centreline sample that holds `position`, measured in units of the sample spacing.

    The curvature inputs are piecewise constant, so a position that is an exact multiple of the spacing
    (for example a car whose nearest road point is a centreline vertex) must not flip to the neighbouring
    sample because of one ulp of rounding noise, which differs between libm builds and between the Python
    and JavaScript engines. Positions within SAMPLE_SNAP of a boundary belong to the sample that starts there."""
    return np.floor(np.asarray(position) + SAMPLE_SNAP).astype(int) % n


def check_action(action):
    """Return (action, is_finite); a wrongly shaped action is a programming error."""
    action = np.asarray(action, dtype=float)
    if action.shape != (2,):
        raise ValueError(f"Expected a (steering, target-speed) pair, got shape {action.shape}")
    return action, bool(np.isfinite(action).all())


def learning_rate_at(schedule, base, update, total):
    """Learning rate used by 1-based `update` of `total` planned updates."""
    if schedule not in LEARNING_RATE_SCHEDULES:
        raise ValueError(f"Unknown learning-rate schedule {schedule!r}")
    progress = (update - 1) / (total - 1) if total > 1 else 0.0
    if schedule == "constant":
        return base
    if schedule == "linear":
        return base * (1 - 0.9 * progress)
    if schedule == "cosine":
        return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
    return base * 0.5 ** min(2, math.floor(3 * progress))


def _segments_cross(p, q, r, s):
    """Strict proper crossing of segment p-q with segment r-s, vectorised."""
    def orientation(a, b, c):
        return (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return (orientation(p, q, r) * orientation(p, q, s) < 0) & (orientation(r, s, p) * orientation(r, s, q) < 0)


def circuit_geometry_report(xy, left, right, curvature, ds, track_width):
    """Physical validity of a closed circuit; `problems` is empty for a drivable road."""
    n, half_width = len(xy), track_width / 2
    max_curvature = float(np.max(np.abs(curvature)))
    min_radius = math.inf if max_curvature == 0 else 1 / max_curvature
    problems = []
    if min_radius <= half_width:
        problems.append(f"tightest bend has radius {min_radius:.1f} m, not larger than half the road "
                        f"width ({half_width:.1f} m), so the inner wall folds over")
    starts, ends = np.vstack([left, right]), np.vstack([np.roll(left, -1, axis=0), np.roll(right, -1, axis=0)])
    wall, index = np.repeat([0, 1], n), np.tile(np.arange(n), 2)
    reach = float(np.max(np.linalg.norm(ends - starts, axis=1))) + 1e-9
    pairs = cKDTree((starts + ends) / 2).query_pairs(r=reach, output_type="ndarray")
    crossings = 0
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        apart = np.abs(index[a] - index[b])
        neighbours = (wall[a] == wall[b]) & (np.minimum(apart, n - apart) <= 1)
        a, b = a[~neighbours], b[~neighbours]
        crossings = int(np.count_nonzero(_segments_cross(starts[a], ends[a], starts[b], ends[b])))
    if crossings:
        problems.append(f"the road edges cross themselves or each other at {crossings} places "
                        "(self-intersecting or overlapping circuit)")
    exclusion = max(3 * track_width, 60.0)
    separation = math.inf
    close = cKDTree(xy).query_pairs(r=2 * track_width, output_type="ndarray")
    if len(close):
        apart = np.abs(close[:, 0] - close[:, 1])
        distant = np.minimum(apart, n - apart) * ds > exclusion
        if distant.any():
            gaps = np.linalg.norm(xy[close[distant, 0]] - xy[close[distant, 1]], axis=1)
            separation = float(np.min(gaps))
    if separation < track_width:
        problems.append(f"two parts of the road more than {exclusion:.0f} m apart along the track come within "
                        f"{separation:.1f} m, closer than the road width ({track_width:.1f} m)")
    return {"min_radius_m": min_radius, "edge_crossings": crossings,
            "min_separation_m": None if math.isinf(separation) else separation, "problems": problems}


@dataclass(frozen=True)
class Config:
    seed: int = 2026
    rounds: int = 120
    dt: float = 0.1
    track_width: float = 18.0
    car_width: float = 1.8
    wheelbase: float = 2.8
    max_steer: float = 0.5
    steering_rate: float = 0.8
    max_speed: float = 36.0
    max_acceleration: float = 4.0
    max_braking: float = 7.0
    max_lateral_acceleration: float = 9.0
    max_steps: int = 1600
    initial_recovery_samples: int = 10000
    rollout_steps: int = 300
    updates_per_round: int = 24
    batch_size: int = 256
    learning_rate: float = 0.001


class Track:
    """Closed circuit. `validate` is "error" (reject physically invalid geometry) or "report"
    (keep the geometry and describe its problems in `self.geometry`). The default Apex Park
    and the legacy catalog (see `tracks_from_catalog`) use "report" so earlier results stay
    reproducible; any other knots are validated strictly unless the caller says otherwise."""

    def __init__(self, config, knots=None, validate=None):
        if validate not in (None, "error", "report"):
            raise ValueError("validate must be 'error' or 'report'")
        validate = validate or ("report" if knots is None else "error")
        self.config = config
        default_knots = [
            [0, -140], [140, -140], [220, -80], [210, 0],
            [110, 40], [200, 100], [190, 180], [70, 210],
            [-40, 140], [-160, 180], [-220, 80], [-170, 0],
            [-220, -80], [-100, -140], [0, -140],
        ]
        knots = np.asarray(default_knots if knots is None else knots, dtype=float)
        if (knots.ndim != 2 or knots.shape[1] != 2 or len(knots) < 5
                or not np.isfinite(knots).all()):
            raise ValueError("Track knots must be a finite N x 2 array with at least five points")
        if not np.array_equal(knots[0], knots[-1]):
            raise ValueError("Track knots must explicitly close the circuit")
        if np.any(np.linalg.norm(np.diff(knots, axis=0), axis=1) < 1e-6):
            raise ValueError("Consecutive track knots must be distinct")
        spline = CubicSpline(np.arange(len(knots)), knots, bc_type="periodic")
        dense = spline(np.linspace(0, len(knots) - 1, 12001))
        cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        self.length = float(cumulative[-1])
        self.n = math.ceil(self.length / 1.5)
        self.ds = self.length / self.n
        self.s = np.arange(self.n) * self.ds
        self.xy = np.column_stack([
            np.interp(self.s, cumulative, dense[:, axis]) for axis in (0, 1)
        ])
        derivative = (np.roll(self.xy, -1, axis=0) - np.roll(self.xy, 1, axis=0))
        self.tangent = derivative / np.linalg.norm(derivative, axis=1)[:, None]
        self.normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])
        self.heading = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        self.curvature = wrap_angle(
            np.roll(self.heading, -1) - np.roll(self.heading, 1)
        ) / (2 * self.ds)
        self.left = self.xy + self.normal * config.track_width / 2
        self.right = self.xy - self.normal * config.track_width / 2
        self.geometry = circuit_geometry_report(
            self.xy, self.left, self.right, self.curvature, self.ds, config.track_width)
        if self.geometry["problems"] and validate == "error":
            raise ValueError("Invalid circuit geometry: " + "; ".join(self.geometry["problems"]))

    def point(self, distance):
        fractional = (distance % self.length) / self.ds
        index = np.floor(fractional).astype(int) % self.n
        weight = np.asarray(fractional - np.floor(fractional))
        return (self.xy[index] * (1 - weight[..., None])
                + self.xy[(index + 1) % self.n] * weight[..., None])

    def locate(self, x, y):
        position = np.array([x, y])
        index = int(np.argmin(np.sum((self.xy - position) ** 2, axis=1)))
        candidates = np.array([(index - 1) % self.n, index])
        start = self.xy[candidates]
        segment = self.xy[(candidates + 1) % self.n] - start
        fraction = np.clip(np.sum((position - start) * segment, axis=1)
                           / np.sum(segment * segment, axis=1), 0, 1)
        projections = start + segment * fraction[:, None]
        closest = int(np.argmin(np.sum((projections - position) ** 2, axis=1)))
        index = int(candidates[closest])
        weight = float(fraction[closest])
        tangent = self.tangent[index] * (1 - weight) + self.tangent[(index + 1) % self.n] * weight
        tangent /= np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        lateral = float(np.dot(position - projections[closest], normal))
        distance = ((index + weight) * self.ds) % self.length
        heading = float(np.arctan2(tangent[1], tangent[0]))
        return distance, lateral, heading

    def save(self, path):
        np.savez_compressed(path, xy=self.xy, left=self.left, right=self.right,
                            s=self.s, curvature=self.curvature, length=self.length)


# Circuits 0-12 predate geometry validation. Apex Park (0) and Square Sprint (6), and their
# GP-X variants, have small inner-wall loops at bends tighter than half the road width.
# They are kept unchanged so every saved result stays reproducible.
LEGACY_CIRCUIT_IDS = frozenset(range(13))


def track_from_catalog(config, item):
    mode = "report" if item["id"] in LEGACY_CIRCUIT_IDS else "error"
    return Track(config, item["knots"], validate=mode)


def tracks_from_catalog(config, catalog):
    return {item["id"]: track_from_catalog(config, item) for item in catalog}


@dataclass
class State:
    x: float
    y: float
    heading: float
    speed: float
    steering: float = 0.0


class Environment:
    preview_distances = np.array([10.0, 20.0, 40.0, 70.0])
    feature_names = [
        "lateral_error_half_width", "sin_heading_error", "cos_heading_error",
        "speed_fraction", "previous_steering_fraction",
        "preview_10m_forward", "preview_10m_left",
        "preview_20m_forward", "preview_20m_left",
        "preview_40m_forward", "preview_40m_left",
        "preview_70m_forward", "preview_70m_left",
    ]

    def __init__(self, track, config):
        self.track = track
        self.config = config
        self.state = self.make_state(0.0)

    def make_state(self, fraction, lateral=0.0, heading_error=0.0, speed=12.0):
        distance = fraction * self.track.length
        index = round(distance / self.track.ds) % self.track.n
        point = self.track.point(distance) + self.track.normal[index] * lateral
        return State(float(point[0]), float(point[1]),
                     float(self.track.heading[index] + heading_error), speed)

    def observe(self, state=None):
        state = self.state if state is None else state
        distance, lateral, road_heading = self.track.locate(state.x, state.y)
        delta = self.track.point(distance + self.preview_distances) - [state.x, state.y]
        cosine, sine = np.cos(state.heading), np.sin(state.heading)
        local = np.column_stack([
            delta[:, 0] * cosine + delta[:, 1] * sine,
            -delta[:, 0] * sine + delta[:, 1] * cosine,
        ]) / self.preview_distances[:, None]
        error = wrap_angle(state.heading - road_heading)
        features = np.r_[
            lateral / (self.config.track_width / 2),
            np.sin(error), np.cos(error),
            state.speed / self.config.max_speed,
            state.steering / self.config.max_steer,
            local.ravel(),
        ]
        return features, (distance, lateral, road_heading)

    def teacher(self, state=None, location=None):
        state = self.state if state is None else state
        config = self.config
        if location is None:
            location = self.track.locate(state.x, state.y)
        distance = location[0]
        lookahead = 8.0 + 0.45 * state.speed
        delta = self.track.point(distance + lookahead) - [state.x, state.y]
        left = -np.sin(state.heading) * delta[0] + np.cos(state.heading) * delta[1]
        steer = np.arctan2(2 * config.wheelbase * left, np.dot(delta, delta))
        steer = float(np.clip(steer, -config.max_steer, config.max_steer))
        preview = distance + np.arange(0, 66, 3)
        indices = sample_index((preview % self.track.length) / self.track.ds, self.track.n)
        curvature = float(np.max(np.abs(self.track.curvature[indices])))
        road_speed = np.sqrt(3.5 / max(curvature, 0.003))
        recovery_speed = np.sqrt(4.5 * config.wheelbase / max(abs(np.tan(steer)), 0.005))
        target_speed = float(np.clip(min(road_speed, recovery_speed, 32.0), 6.0, config.max_speed))
        return np.array([steer / config.max_steer,
                         2 * target_speed / config.max_speed - 1])

    def step(self, action):
        config = self.config
        state = self.state
        action, finite = check_action(action)
        if not finite:
            return self.track.locate(state.x, state.y), NUMERICAL_DIVERGENCE, 0.0
        requested_steer = float(np.clip(action[0], -1, 1) * config.max_steer)
        target_speed = float(np.clip((action[1] + 1) * config.max_speed / 2, 0, config.max_speed))
        steering = state.steering + float(np.clip(
            requested_steer - state.steering,
            -config.steering_rate * config.dt, config.steering_rate * config.dt,
        ))
        acceleration = float(np.clip(1.8 * (target_speed - state.speed),
                                    -config.max_braking, config.max_acceleration))
        acceleration -= 0.002 * state.speed ** 2
        speed = float(np.clip(state.speed + acceleration * config.dt, 0, config.max_speed))
        mean_speed = (state.speed + speed) / 2
        heading_change = mean_speed / config.wheelbase * np.tan(steering) * config.dt
        midpoint_heading = state.heading + heading_change / 2
        self.state = State(
            state.x + mean_speed * np.cos(midpoint_heading) * config.dt,
            state.y + mean_speed * np.sin(midpoint_heading) * config.dt,
            float(wrap_angle(state.heading + heading_change)), speed, steering,
        )
        location = self.track.locate(self.state.x, self.state.y)
        lateral_acceleration = speed ** 2 * abs(np.tan(steering)) / config.wheelbase
        reason = ""
        if abs(location[1]) + config.car_width / 2 > config.track_width / 2:
            reason = "off_track"
        elif lateral_acceleration > config.max_lateral_acceleration:
            reason = "lateral_acceleration_limit"
        return location, reason, float(lateral_acceleration)


class NeuralDriver:
    """Small two-hidden-layer tanh MLP with explicit Adam updates."""

    def __init__(self, seed, input_size=13, hidden_sizes=(48, 32)):
        rng = np.random.default_rng(seed)
        sizes = [input_size, *hidden_sizes, 2]
        self.weights = [
            rng.normal(0, np.sqrt(1 / sizes[i]), (sizes[i], sizes[i + 1]))
            for i in range(3)
        ]
        self.biases = [np.zeros(size) for size in sizes[1:]]
        self.parameters = [item for pair in zip(self.weights, self.biases) for item in pair]
        self.first = [np.zeros_like(item) for item in self.parameters]
        self.second = [np.zeros_like(item) for item in self.parameters]
        self.updates = 0
        self.last_gradient_norm = 0.0
        self.last_clipped = False

    def predict(self, features):
        value = np.asarray(features)
        for weight, bias in zip(self.weights, self.biases):
            value = np.tanh(value @ weight + bias)
        return value

    def train_batch(self, features, targets, learning_rate, weight_decay=0.0, max_grad_norm=None):
        """One Adam update on a minibatch; returns the weighted mean squared error.

        `weight_decay` is decoupled (AdamW-style) and applies to weights, not biases.
        `max_grad_norm` rescales the whole gradient when its global L2 norm is larger.
        Non-finite inputs, loss or gradients raise before any parameter changes.
        """
        if not (math.isfinite(learning_rate) and 0 < learning_rate <= MAX_LEARNING_RATE):
            raise ValueError(f"learning_rate must be in (0, {MAX_LEARNING_RATE}], got {learning_rate!r}")
        if not (math.isfinite(weight_decay) and 0 <= weight_decay <= 1):
            raise ValueError(f"weight_decay must be in [0, 1], got {weight_decay!r}")
        if max_grad_norm is not None and not (math.isfinite(max_grad_norm) and max_grad_norm > 0):
            raise ValueError(f"max_grad_norm must be positive and finite, got {max_grad_norm!r}")
        if not (np.isfinite(features).all() and np.isfinite(targets).all()):
            raise ValueError("Training batch contains NaN or infinite values")
        activations = [features]
        for weight, bias in zip(self.weights, self.biases):
            activations.append(np.tanh(activations[-1] @ weight + bias))
        residual = activations[-1] - targets
        # Steering has the larger closed-loop safety impact.
        importance = np.array([4.0, 1.0])
        loss = float(np.mean(residual ** 2 * importance))
        gradient = residual * importance / len(features)
        gradients = []
        for layer in range(2, -1, -1):
            gradient *= 1 - activations[layer + 1] ** 2
            weight_gradient = activations[layer].T @ gradient
            bias_gradient = np.sum(gradient, axis=0)
            gradients[0:0] = [weight_gradient, bias_gradient]
            gradient = gradient @ self.weights[layer].T
        norm = float(np.sqrt(sum(float(np.sum(item * item)) for item in gradients)))
        if not (math.isfinite(loss) and math.isfinite(norm)):
            raise FloatingPointError(
                f"Non-finite loss or gradient at update {self.updates + 1}; parameters were left "
                "unchanged. Lower the learning rate or check the training data.")
        self.last_gradient_norm = norm
        self.last_clipped = max_grad_norm is not None and norm > max_grad_norm
        scale = max_grad_norm / norm if self.last_clipped else 1.0
        self.updates += 1
        for index, (parameter, derivative) in enumerate(zip(self.parameters, gradients)):
            if self.last_clipped:
                derivative = derivative * scale
            self.first[index] = 0.9 * self.first[index] + 0.1 * derivative
            self.second[index] = 0.999 * self.second[index] + 0.001 * derivative ** 2
            first = self.first[index] / (1 - 0.9 ** self.updates)
            second = self.second[index] / (1 - 0.999 ** self.updates)
            if weight_decay and index % 2 == 0:
                parameter -= learning_rate * weight_decay * parameter
            parameter -= learning_rate * first / (np.sqrt(second) + 1e-8)
        return loss

    def save(self, path):
        arrays = {f"w{i}": value for i, value in enumerate(self.weights)}
        arrays.update({f"b{i}": value for i, value in enumerate(self.biases)})
        arrays["updates"] = np.array(self.updates)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path) as arrays:
            driver = cls(0, input_size=arrays["w0"].shape[0],
                         hidden_sizes=(arrays["w0"].shape[1], arrays["w1"].shape[1]))
            for i in range(3):
                driver.weights[i][...] = arrays[f"w{i}"]
                driver.biases[i][...] = arrays[f"b{i}"]
            driver.updates = int(arrays["updates"])
        return driver


SCENARIOS = [
    {"name": "Start straight", "fraction": 0.0, "lateral": 0.0, "heading_error": 0.0, "speed": 12.0},
    {"name": "Corner entry", "fraction": 0.28, "lateral": 1.0, "heading_error": 0.04, "speed": 12.0},
    {"name": "Recovery start", "fraction": 0.63, "lateral": -2.0, "heading_error": -0.08, "speed": 10.0},
]
CHECKPOINTS = [0, 1, 5, 10, 20, 40, 60, 80, 100, 120]


def simulate(track, config, driver, scenario, environment_type=Environment):
    environment = environment_type(track, config)
    environment.state = environment.make_state(
        scenario["fraction"], scenario["lateral"],
        scenario["heading_error"], scenario["speed"],
    )
    _, location = environment.observe()
    previous_distance = location[0]
    progress = 0.0
    trace = []
    reason = "timeout"
    completed = False
    for step in range(config.max_steps):
        features, location = environment.observe()
        state = environment.state
        action = environment.teacher(location=location) if driver is None else driver.predict(features)
        trace.append([
            step * config.dt, state.x, state.y, state.heading, state.speed,
            state.steering, location[1], progress / track.length,
        ])
        location, failure, lateral_acceleration = environment.step(action)
        progress += ((location[0] - previous_distance + track.length / 2)
                     % track.length - track.length / 2)
        previous_distance = location[0]
        if failure or progress >= track.length:
            completed = not failure and progress >= track.length
            reason = "lap_complete" if completed else failure
            state = environment.state
            trace.append([
                (step + 1) * config.dt, state.x, state.y, state.heading,
                state.speed, state.steering, location[1], progress / track.length,
            ])
            break
    trajectory = np.asarray(trace)
    metrics = {
        "scenario": scenario["name"],
        "completed": int(completed),
        "progress": float(np.clip(progress / track.length, 0, 1)),
        "duration_s": float(trajectory[-1, 0]),
        "lap_time_s": float(trajectory[-1, 0]) if completed else None,
        "mean_abs_lateral_m": float(np.mean(np.abs(trajectory[:, 6]))),
        "max_abs_lateral_m": float(np.max(np.abs(trajectory[:, 6]))),
        "mean_speed_kmh": float(np.mean(trajectory[:, 4]) * 3.6),
        "steering_variation_rad_s": float(np.mean(np.abs(np.diff(trajectory[:, 5]))) / config.dt),
        "end_reason": reason,
    }
    return metrics, trajectory


def collect_initial_data(track, config, rng, environment_type=Environment):
    environment = environment_type(track, config)
    features, targets, sources = [], [], []
    for _ in range(config.initial_recovery_samples):
        state = environment.make_state(
            float(rng.uniform()), float(rng.uniform(-6, 6)),
            float(rng.uniform(-0.45, 0.45)), float(rng.uniform(6, 32)),
        )
        observation, location = environment.observe(state)
        target = environment.teacher(state, location)
        state.steering = float(target[0] * config.max_steer + rng.normal(0, 0.03))
        state.steering = float(np.clip(state.steering, -config.max_steer, config.max_steer))
        observation, _ = environment.observe(state)
        features.append(observation)
        targets.append(target)
        sources.append(0)
    for fraction in (0.11, 0.41, 0.79):
        environment.state = environment.make_state(fraction, speed=10.0)
        distance = environment.observe()[1][0]
        progress = 0.0
        for _ in range(config.max_steps):
            observation, location = environment.observe()
            action = environment.teacher(location=location)
            features.append(observation)
            targets.append(action)
            sources.append(1)
            location, failure, _ = environment.step(action)
            progress += ((location[0] - distance + track.length / 2)
                         % track.length - track.length / 2)
            distance = location[0]
            if failure:
                raise RuntimeError(f"Teacher demonstration failed: {failure}")
            if progress >= track.length:
                break
        else:
            raise RuntimeError("Teacher could not complete a demonstration lap")
    return np.asarray(features), np.asarray(targets), np.asarray(sources, dtype=np.uint8)


def collect_learner_data(track, config, driver, rng):
    environment = Environment(track, config)
    features, targets = [], []
    def reset():
        environment.state = environment.make_state(
            float(rng.uniform()), float(rng.uniform(-3, 3)),
            float(rng.uniform(-0.12, 0.12)), float(rng.uniform(8, 18)),
        )
    reset()
    for _ in range(config.rollout_steps):
        observation, location = environment.observe()
        features.append(observation)
        targets.append(environment.teacher(location=location))
        _, failure, _ = environment.step(driver.predict(observation))
        if failure:
            reset()
    return np.asarray(features), np.asarray(targets)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def config_dict(config):
    return asdict(config)
