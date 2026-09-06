import pandas as pd
import joblib

from pathlib import Path


# ------------------------------------------------
# File paths
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "stiffness_model.pkl"


# ------------------------------------------------
# Stiffness "yes/no" threshold
# ------------------------------------------------
# This is the point above which the massager should treat the
# body region as stiff and respond (e.g. raise intensity, flag
# the region). Tune this based on real device behavior, not
# statistics from the synthetic training data.

STIFF_THRESHOLD = 70.0


# ------------------------------------------------
# Load trained model
# ------------------------------------------------

print("Loading stiffness model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!\n")


# ------------------------------------------------
# Get sensor input
# ------------------------------------------------

pressure_avg = float(input("Average Pressure: "))

pressure_variation = float(
    input("Pressure Variation: ")
)

motor_current = float(
    input("Motor Current: ")
)

skin_temperature = float(
    input("Skin Temperature: ")
)

posture_angle = float(
    input("Posture Angle: ")
)

emg_activity = float(
    input("EMG Activity: ")
)

massage_intensity = float(
    input("Massage Intensity: ")
)

session_duration = float(
    input("Session Duration (minutes): ")
)

discomfort_level = float(
    input("Discomfort Level (0-7): ")
)


# ------------------------------------------------
# Create input DataFrame
# ------------------------------------------------

input_data = pd.DataFrame({

    "pressure_avg": [pressure_avg],

    "pressure_variation": [pressure_variation],

    "motor_current": [motor_current],

    "skin_temperature": [skin_temperature],

    "posture_angle": [posture_angle],

    "emg_activity": [emg_activity],

    "massage_intensity": [massage_intensity],

    "session_duration": [session_duration],

    "discomfort_level": [discomfort_level]

})


# ------------------------------------------------
# Predict stiffness
# ------------------------------------------------

prediction = model.predict(input_data)

stiffness = float(prediction[0])

# Keep between 0 and 100
stiffness = max(0, min(100, stiffness))


# ------------------------------------------------
# Determine stiffness level (for display / logging)
# ------------------------------------------------

if stiffness < 30:

    level = "LOW"

    message = "Muscle stiffness is low."

elif stiffness < 60:

    level = "MEDIUM"

    message = "Moderate muscle stiffness detected."

else:

    level = "HIGH"

    message = "High muscle stiffness detected. Consider reducing massage intensity."


# ------------------------------------------------
# Determine yes/no stiffness verdict (for the massager's decision logic)
# ------------------------------------------------

is_stiff = stiffness >= STIFF_THRESHOLD

verdict = "YES" if is_stiff else "NO"


# ------------------------------------------------
# Display result
# ------------------------------------------------

print("\n" + "=" * 40)

print("       MUSCLE STIFFNESS RESULT")

print("=" * 40)

print(f"Stiffness Percentage : {stiffness:.2f}%")

print(f"Stiffness Level      : {level}")

print(f"Stiff? (threshold {STIFF_THRESHOLD:.0f}%) : {verdict}")

print(f"Message              : {message}")

print("=" * 40)