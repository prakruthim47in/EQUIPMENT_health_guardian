"""
Synthetic industrial sensor data simulator.
Generates vibration, temperature, and RPM readings for multiple machines,
with a few injectable fault patterns so your demo has realistic scenarios.
"""
import numpy as np
import pandas as pd

MACHINES = ["Motor-A1", "Compressor-B2", "Pump-C3"]

FAULT_PROFILES = {
    "none": {"vib_shift": 0.0, "temp_shift": 0.0, "rpm_shift": 0.0, "noise_mult": 1.0},
    "bearing_wear": {"vib_shift": 3.5, "temp_shift": 4.0, "rpm_shift": -50, "noise_mult": 2.2},
    "overheating": {"vib_shift": 0.5, "temp_shift": 18.0, "rpm_shift": -20, "noise_mult": 1.3},
    "imbalance": {"vib_shift": 5.0, "temp_shift": 1.0, "rpm_shift": 30, "noise_mult": 3.0},
}


def generate_series(machine_name: str, n_points: int = 200, fault: str = "none",
                     fault_start_frac: float = 0.65, seed: int | None = None) -> pd.DataFrame:
    """Generate a time series of sensor readings for one machine.

    fault_start_frac controls where in the series the fault begins (0-1),
    so early readings look healthy and later ones drift into the fault.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_points)

    base_vibration = 2.0 + 0.3 * np.sin(t / 8)
    base_temperature = 55 + 3 * np.sin(t / 20)
    base_rpm = 1500 + 15 * np.sin(t / 15)

    profile = FAULT_PROFILES[fault]
    fault_start = int(n_points * fault_start_frac)
    ramp = np.clip((t - fault_start) / max(1, (n_points - fault_start)), 0, 1)

    vibration = base_vibration + ramp * profile["vib_shift"] + rng.normal(0, 0.15 * profile["noise_mult"], n_points)
    temperature = base_temperature + ramp * profile["temp_shift"] + rng.normal(0, 0.6 * profile["noise_mult"], n_points)
    rpm = base_rpm + ramp * profile["rpm_shift"] + rng.normal(0, 5 * profile["noise_mult"], n_points)

    df = pd.DataFrame({
        "t": t,
        "machine": machine_name,
        "vibration_mm_s": vibration.round(3),
        "temperature_c": temperature.round(2),
        "rpm": rpm.round(1),
        "true_fault": [fault if r > 0.05 else "none" for r in ramp],
    })
    return df


def generate_fleet(fault_map: dict[str, str] | None = None, n_points: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate data for the whole machine fleet.

    fault_map: e.g. {"Motor-A1": "bearing_wear"} to inject a fault into a specific machine.
    Machines not in fault_map run healthy ("none").
    """
    fault_map = fault_map or {}
    frames = []
    for i, m in enumerate(MACHINES):
        fault = fault_map.get(m, "none")
        frames.append(generate_series(m, n_points=n_points, fault=fault, seed=seed + i))
    return pd.concat(frames, ignore_index=True)
