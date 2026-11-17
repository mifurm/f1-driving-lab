"""Harder circuit variants and an explicit fictional automatic drivetrain."""

from dataclasses import dataclass

import numpy as np

from circuits import CIRCUITS
from driving import Config, State, Track, wrap_angle
from multitrack import MultiTrackEnvironment


RACE_CONFIG = Config(
    seed=2028, rounds=120, max_steps=2600, initial_recovery_samples=1800,
    rollout_steps=220, updates_per_round=64, batch_size=484,
)
TRACE_COLUMNS = [
    "time_s", "x_m", "y_m", "heading_rad", "speed_m_s", "steering_rad",
    "lateral_error_m", "lap_progress_fraction", "throttle_fraction",
    "brake_fraction", "gear", "engine_rpm", "acceleration_m_s2",
    "target_speed_m_s", "requested_steering_rad", "lateral_acceleration_m_s2",
    "shift_active",
]
GEAR_RATIOS = np.array([7.5, 5.0, 3.6, 2.7, 2.1, 1.65, 1.30, 1.05])
FINAL_DRIVE = 3.8
WHEEL_RADIUS = 0.33
IDLE_RPM = 3000.0
UPSHIFT_RPM = 9000.0
DOWNSHIFT_RPM = 4800.0


def reversal_count(track):
    meaningful = np.sign(track.curvature[np.abs(track.curvature) > 0.003])
    return int(np.count_nonzero(meaningful != np.roll(meaningful, 1)))


def complex_circuits(config=RACE_CONFIG):
    catalog, tracks = [], {}
    for circuit in CIRCUITS:
        base = circuit.build(config)
        scale = 1.15
        length = base.length * scale
        distances = np.linspace(0, length, 400, endpoint=False)
        indices = np.floor(distances / scale / base.ds).astype(int) % base.n
        points = base.point(distances / scale) * scale
        normals = base.normal[indices]
        curvature = np.abs(base.curvature[indices]) / scale
        displacement = np.zeros(len(points))
        windows = []
        for sector, target in enumerate((0.13, 0.37, 0.62, 0.86)):
            width = 135.0 + (circuit.id % 3) * 10
            candidates = np.linspace(target - 0.07, target + 0.07, 35) % 1
            scores = []
            for fraction in candidates:
                separation = (distances - fraction * length + length / 2) % length - length / 2
                maximum = float(np.max(curvature[np.abs(separation) <= width / 2]))
                score = maximum + abs(fraction - target) * 0.03
                if any(abs((fraction * length - c + length / 2) % length - length / 2) < width + 20
                       for c, _ in windows):
                    score += 10
                scores.append(score)
            center = float(candidates[int(np.argmin(scores))] * length)
            offset = (distances - center + length / 2) % length - length / 2
            maximum = float(np.max(curvature[np.abs(offset) <= width / 2]))
            amplitude = min(13.0 + circuit.id % 4, 0.26 / max(maximum, 0.005))
            u = offset / width + 0.5
            inside = (u >= 0) & (u <= 1)
            displacement[inside] += (
                (-1) ** sector * amplitude
                * np.sin(2 * np.pi * u[inside]) * np.sin(np.pi * u[inside]) ** 2
            )
            windows.append((center, width))
        knots = points + normals * displacement[:, None]
        knots = np.vstack([knots, knots[0]])
        track = Track(config, knots)
        tracks[circuit.id] = track
        catalog.append({
            "id": circuit.id, "name": f"{circuit.name} GP-X", "split": circuit.split,
            "style": f"{circuit.style}; four added left-right technical sectors",
            "knots": knots.tolist(), "length_m": track.length,
            "previous_length_m": base.length, "previous_reversals": reversal_count(base),
            "reversals": reversal_count(track),
            "added_sectors": [{"center_m": center, "window_m": width} for center, width in windows],
            "width_m": config.track_width,
        })
    return catalog, tracks


def engine_rpm(speed, gear):
    wheel_rpm = speed * 60 / (2 * np.pi * WHEEL_RADIUS)
    return float(np.clip(wheel_rpm * FINAL_DRIVE * GEAR_RATIOS[gear - 1], IDLE_RPM, 12000))


@dataclass
class RaceState(State):
    gear: int = 1
    rpm: float = IDLE_RPM
    throttle: float = 0.0
    brake: float = 0.0
    acceleration: float = 0.0
    target_speed: float = 0.0
    requested_steering: float = 0.0
    lateral_acceleration: float = 0.0
    shift_active: bool = False
    shift_remaining: float = 0.0
    shift_cooldown: float = 0.0


class RaceEnvironment(MultiTrackEnvironment):
    def make_state(self, fraction, lateral=0.0, heading_error=0.0, speed=12.0):
        state = super().make_state(fraction, lateral, heading_error, speed)
        gear = 1
        while gear < len(GEAR_RATIOS) and engine_rpm(speed, gear) > UPSHIFT_RPM:
            gear += 1
        return RaceState(
            state.x, state.y, state.heading, state.speed,
            gear=gear, rpm=engine_rpm(speed, gear), target_speed=speed,
        )

    def step(self, action):
        state, config = self.state, self.config
        gear = state.gear
        shift_remaining = max(0.0, state.shift_remaining - config.dt)
        cooldown = max(0.0, state.shift_cooldown - config.dt)
        rpm = engine_rpm(state.speed, gear)
        if cooldown == 0:
            if rpm > UPSHIFT_RPM and gear < len(GEAR_RATIOS):
                gear += 1
                shift_remaining, cooldown = 0.2, 0.6
            elif rpm < DOWNSHIFT_RPM and gear > 1:
                gear -= 1
                shift_remaining, cooldown = 0.2, 0.6
        shifting = shift_remaining > 0
        rpm = engine_rpm(state.speed, gear)
        requested_steer = float(np.clip(action[0], -1, 1) * config.max_steer)
        target_speed = float(np.clip((action[1] + 1) * config.max_speed / 2, 0, config.max_speed))
        requested_acceleration = 1.8 * (target_speed - state.speed)
        throttle = float(np.clip(requested_acceleration / config.max_acceleration, 0, 1))
        brake = float(np.clip(-requested_acceleration / config.max_braking, 0, 1))
        torque = 150 * (0.75 + 0.25 * np.exp(-((rpm - 7000) / 3000) ** 2))
        engine_acceleration = min(
            config.max_acceleration,
            torque * GEAR_RATIOS[gear - 1] * FINAL_DRIVE * 0.88 / (WHEEL_RADIUS * 950),
        )
        acceleration = throttle * engine_acceleration * (not shifting)
        acceleration -= brake * config.max_braking + 0.002 * state.speed ** 2
        speed = float(np.clip(state.speed + acceleration * config.dt, 0, config.max_speed))
        steering = state.steering + float(np.clip(
            requested_steer - state.steering,
            -config.steering_rate * config.dt, config.steering_rate * config.dt,
        ))
        mean_speed = (state.speed + speed) / 2
        heading_change = mean_speed / config.wheelbase * np.tan(steering) * config.dt
        midpoint = state.heading + heading_change / 2
        lateral_acceleration = float(speed ** 2 * abs(np.tan(steering)) / config.wheelbase)
        self.state = RaceState(
            state.x + mean_speed * np.cos(midpoint) * config.dt,
            state.y + mean_speed * np.sin(midpoint) * config.dt,
            float(wrap_angle(state.heading + heading_change)), speed, steering,
            gear, engine_rpm(speed, gear), throttle, brake, float(acceleration),
            target_speed, requested_steer, lateral_acceleration, shifting,
            shift_remaining, cooldown,
        )
        location = self.track.locate(self.state.x, self.state.y)
        reason = ""
        if abs(location[1]) + config.car_width / 2 > config.track_width / 2:
            reason = "off_track"
        elif lateral_acceleration > config.max_lateral_acceleration:
            reason = "lateral_acceleration_limit"
        return location, reason, lateral_acceleration


def telemetry_row(environment, time, location, progress):
    state = environment.state
    return [
        time, state.x, state.y, state.heading, state.speed, state.steering,
        location[1], progress / environment.track.length,
        state.throttle, state.brake, state.gear, state.rpm, state.acceleration,
        state.target_speed, state.requested_steering, state.lateral_acceleration,
        int(state.shift_active),
    ]


def race_simulate(track, config, driver, scenario):
    environment = RaceEnvironment(track, config)
    environment.state = environment.make_state(
        scenario["fraction"], scenario["lateral"], scenario["heading_error"], scenario["speed"],
    )
    _, location = environment.observe()
    previous = location[0]
    progress = 0.0
    trace = [telemetry_row(environment, 0, location, progress)]
    reason = "timeout"
    for step in range(config.max_steps):
        features, location = environment.observe()
        action = environment.teacher(location=location) if driver is None else driver.predict(features)
        location, failure, _ = environment.step(action)
        progress += (location[0] - previous + track.length / 2) % track.length - track.length / 2
        previous = location[0]
        trace.append(telemetry_row(environment, (step + 1) * config.dt, location, progress))
        if failure or progress >= track.length:
            reason = failure if failure else "lap_complete"
            break
    trace = np.asarray(trace)
    completed = reason == "lap_complete"
    metrics = {
        "scenario": scenario["name"], "completed": int(completed),
        "progress": float(np.clip(progress / track.length, 0, 1)),
        "end_reason": reason, "duration_s": float(trace[-1, 0]),
        "lap_time_s": float(trace[-1, 0]) if completed else None,
        "mean_abs_lateral_m": float(np.mean(np.abs(trace[:, 6]))),
        "max_abs_lateral_m": float(np.max(np.abs(trace[:, 6]))),
        "mean_speed_kmh": float(np.mean(trace[:, 4]) * 3.6),
        "peak_speed_kmh": float(np.max(trace[:, 4]) * 3.6),
        "braking_time_s": float(np.count_nonzero(trace[1:, 9] > 0.01) * config.dt),
        "brake_energy_proxy": float(np.sum(trace[1:, 9] * trace[1:, 4]) * config.dt),
        "gear_changes": int(np.count_nonzero(np.diff(trace[:, 10]))),
        "max_gear": int(np.max(trace[:, 10])),
        "steering_variation_rad_s": float(np.mean(np.abs(np.diff(trace[:, 5]))) / config.dt),
    }
    return metrics, trace
