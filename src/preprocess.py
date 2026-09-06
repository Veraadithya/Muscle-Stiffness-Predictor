import pandas as pd
from pathlib import Path


# ------------------------------------------------
# File paths
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA = BASE_DIR / "data" / "raw" / "muscle_stiffness.csv"
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "training_data.csv"


# ------------------------------------------------
# Yes/No stiffness threshold
# ------------------------------------------------
# Same threshold used in predict.py. Kept here too so the processed
# dataset carries a ready-made is_stiff column, useful if a
# classification model is trained later. Tune based on the real
# device behavior, not statistics.

STIFF_THRESHOLD = 70.0


# ------------------------------------------------
# Expected sensor ranges
# ------------------------------------------------
# Rows outside these physically plausible ranges are almost
# certainly sensor glitches or bad readings, not real data.
# Ranges are based on the device's known operating limits
# (see create_dataset.py for the generating ranges).

VALID_RANGES = {
    "pressure_avg": (0, 100),
    "pressure_variation": (0, 20),
    "motor_current": (0, 2.0),
    "skin_temperature": (25, 40),
    "posture_angle": (0, 90),
    "emg_activity": (0, 1.0),
    "massage_intensity": (0, 100),
    "session_duration": (0, 60),
    "discomfort_level": (0, 10),
    "reference_stiffness": (0, 100),
}


def load_data():
    """Load the raw dataset."""
    data = pd.read_csv(RAW_DATA)

    print("Raw dataset loaded")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    return data


def clean_data(data):
    """Clean and prepare the dataset."""

    initial_rows = len(data)

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Convert timestamp
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    # Remove rows with invalid timestamps
    data = data.dropna(subset=["timestamp"])

    # Numerical columns
    numerical_columns = list(VALID_RANGES.keys())

    # Convert numerical columns to numbers
    for column in numerical_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove rows with missing values
    data = data.dropna()

    # Drop rows with sensor readings outside plausible ranges
    # (glitches / dropouts rather than genuine measurements)
    for column, (low, high) in VALID_RANGES.items():
        before = len(data)
        data = data[(data[column] >= low) & (data[column] <= high)]
        dropped = before - len(data)
        if dropped:
            print(f"Dropped {dropped} rows with out-of-range '{column}'")

    # Reset index
    data = data.reset_index(drop=True)

    print(f"\nTotal rows removed during cleaning: {initial_rows - len(data)}")

    return data


def add_derived_columns(data):
    """Add columns derived from the cleaned sensor data."""

    # Binary stiffness flag, useful if a classifier is trained later
    # instead of / alongside the regression model.
    data["is_stiff"] = (
        data["reference_stiffness"] >= STIFF_THRESHOLD
    ).astype(int)

    print(f"\nStiff/Not-stiff split at threshold {STIFF_THRESHOLD:.0f}%:")
    print(data["is_stiff"].value_counts().rename({1: "stiff (1)", 0: "not stiff (0)"}))

    return data


def save_data(data):
    """Save processed dataset."""

    PROCESSED_DATA.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        PROCESSED_DATA,
        index=False
    )

    print("\nProcessed dataset saved:")
    print(PROCESSED_DATA)


def main():

    # Load
    data = load_data()

    # Clean
    data = clean_data(data)

    # Add derived columns (e.g. is_stiff)
    data = add_derived_columns(data)

    # Save
    save_data(data)

    print("\nFinal dataset:")
    print(data.head())

    print("\nFinal shape:")
    print(data.shape)


if __name__ == "__main__":
    main()