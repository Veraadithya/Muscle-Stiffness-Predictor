import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

# ------------------------------------------------
# Design notes (why this version differs from v1)
# ------------------------------------------------
# 1. The original generator produced reference_stiffness values that
#    were ALWAYS >= 51 (min was 51.3 across 500 rows). That makes a
#    yes/no "is stiffness present" label useless, since almost every
#    row is "yes". This version widens the underlying sensor ranges
#    and rebalances the formula so stiffness spans close to the full
#    0-100 range, with a realistic distribution of relaxed, moderate,
#    and stiff sessions.
#
# 2. Public research on real muscle stiffness (shear-wave elastography
#    studies on plantar flexors, e.g. Belghith et al., Data in Brief
#    2024) consistently shows stiffness increasing with joint/posture
#    angle and with active muscle contraction (proxied here by EMG
#    activity). Those two relationships are given the strongest
#    weight below, which is more physiologically grounded than an
#    arbitrary flat linear combination.
#
# 3. This is STILL synthetic data. It is useful for building and
#    testing your ML pipeline (preprocessing, model comparison,
#    saving/loading models, predict.py), but it is NOT a substitute
#    for real sensor + real stiffness-label data from your own
#    device. Swap this file's output for real collected data as soon
#    as you have it - the pipeline downstream doesn't need to change.
# ------------------------------------------------

N = 1000  # more rows than v1, gives CV folds more to work with

start_time = datetime(2026, 8, 1, 9, 0, 0)
timestamps = [start_time + timedelta(minutes=i) for i in range(N)]

# ------------------------------------------------
# Body regions
# ------------------------------------------------
# Different baseline stiffness tendencies by region (lower back
# tends to carry more chronic tension in most posture-related
# stiffness literature than upper back).

body_regions = np.random.choice(
    ["Upper_Back", "Middle_Back", "Lower_Back"], N, p=[0.30, 0.30, 0.40]
)

region_baseline = np.where(
    body_regions == "Lower_Back", 8,
    np.where(body_regions == "Middle_Back", 3, 0)
)

# ------------------------------------------------
# Sensor data
# ------------------------------------------------
# Ranges widened vs v1 so both relaxed and stiff sessions are
# represented, rather than only the upper half of each range.

pressure_avg = np.random.uniform(10, 55, N)
pressure_variation = np.random.uniform(0.5, 7, N)
motor_current = np.random.uniform(0.2, 1.0, N)
skin_temperature = np.random.uniform(32.5, 35.5, N)
posture_angle = np.random.uniform(0, 45, N)          # wider range, includes near-neutral posture
emg_activity = np.random.uniform(0.05, 0.85, N)      # includes near-resting EMG
massage_intensity = np.random.uniform(20, 80, N)
session_duration = np.random.uniform(5, 20, N)
discomfort_level = np.random.randint(0, 8, N)

# ------------------------------------------------
# Synthetic reference stiffness
# ------------------------------------------------
# Still NOT a medical formula - for pipeline testing only. Relative
# WEIGHTS below reflect the literature's emphasis (posture_angle and
# emg_activity as the strongest drivers, per SWE studies), but the
# combined raw score is standardized (z-scored) before being mapped
# onto 0-100. This is what actually gives a realistic spread of low/
# medium/high stiffness sessions - weighting alone isn't enough,
# since summed weighted terms tend to cluster near one end of the
# range and get clipped there (that was the bug in the first pass).

raw_score = (
    region_baseline
    + 1.1 * posture_angle          # strongest driver
    + 35 * emg_activity            # second strongest driver
    + 0.5 * pressure_avg
    + 12 * motor_current
    + 1.3 * discomfort_level
    + 0.1 * massage_intensity
)

# Standardize to mean 0, std 1, then rescale to a target distribution
# centered near the middle of the 0-100 range with a spread wide
# enough to cover low/medium/high stiffness in realistic proportions.
raw_z = (raw_score - raw_score.mean()) / raw_score.std()

reference_stiffness = (
    50                              # target mean
    + raw_z * 18                   # target spread from the weighted drivers
    + np.random.normal(0, 10, N)   # measurement noise
)

reference_stiffness = np.clip(reference_stiffness, 0, 100)

# ------------------------------------------------
# Build DataFrame
# ------------------------------------------------

data = pd.DataFrame({
    "timestamp": timestamps,
    "body_region": body_regions,
    "pressure_avg": pressure_avg,
    "pressure_variation": pressure_variation,
    "motor_current": motor_current,
    "skin_temperature": skin_temperature,
    "posture_angle": posture_angle,
    "emg_activity": emg_activity,
    "massage_intensity": massage_intensity,
    "session_duration": session_duration,
    "discomfort_level": discomfort_level,
    "reference_stiffness": reference_stiffness
})

numeric_columns = data.select_dtypes(include=["float64"]).columns
data[numeric_columns] = data[numeric_columns].round(2)

os.makedirs("data/raw", exist_ok=True)
data.to_csv("data/raw/muscle_stiffness.csv", index=False)

print("Dataset created successfully!")
print("Number of records:", len(data))
print("\nStiffness distribution:")
print(data["reference_stiffness"].describe())
print("\nStiffness range check (should span most of 0-100, not just 51-100):")
print(f"  Min: {data['reference_stiffness'].min():.1f}")
print(f"  Max: {data['reference_stiffness'].max():.1f}")
print(f"  % below 30 (low):    {(data['reference_stiffness'] < 30).mean() * 100:.1f}%")
print(f"  % 30-60 (medium):    {((data['reference_stiffness'] >= 30) & (data['reference_stiffness'] < 60)).mean() * 100:.1f}%")
print(f"  % 60+ (high):        {(data['reference_stiffness'] >= 60).mean() * 100:.1f}%")
print("\nFirst 5 rows:")
print(data.head())