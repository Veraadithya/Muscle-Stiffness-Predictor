import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ------------------------------------------------
# File paths
# ------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "training_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "stiffness_model.pkl"
RESULTS_PATH = MODEL_DIR / "model_comparison.csv"


# ------------------------------------------------
# Load dataset
# ------------------------------------------------

print("Loading dataset...")

data = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Rows: {data.shape[0]}")
print(f"Columns: {data.shape[1]}")


# ------------------------------------------------
# Prepare features and target
# ------------------------------------------------

FEATURES = [
    "pressure_avg",
    "pressure_variation",
    "motor_current",
    "skin_temperature",
    "posture_angle",
    "emg_activity",
    "massage_intensity",
    "session_duration",
    "discomfort_level",
]

TARGET = "reference_stiffness"

X = data[FEATURES]
y = data[TARGET]


# ------------------------------------------------
# Train-test split (held out for final evaluation)
# ------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------------
# Candidate models
# ------------------------------------------------
# Linear/SVM/KNN models are sensitive to feature scale, so they're
# wrapped in a Pipeline with a StandardScaler. Tree-based ensembles
# don't need scaling but it doesn't hurt them either.

candidates = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ]),
    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1, random_state=42)),
    ]),
    "ElasticNet": Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
    ]),
    "KNeighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=7)),
    ]),
    "SVR_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=10, epsilon=0.5)),
    ]),
    "DecisionTree": DecisionTreeRegressor(
        max_depth=8, random_state=42
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=200, learning_rate=0.05, random_state=42
    ),
}

if HAS_XGBOOST:
    candidates["XGBoost"] = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )
else:
    print("\n(xgboost not installed - skipping XGBoost. "
          "Run 'pip install xgboost --break-system-packages' to include it.)")


# ------------------------------------------------
# Compare models with 5-fold cross-validation on the training set
# ------------------------------------------------

print("\nRunning 5-fold cross-validation on each model...\n")

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results = []

for name, model in candidates.items():
    mae_scores = -cross_val_score(
        model, X_train, y_train, cv=kfold,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    r2_scores = cross_val_score(
        model, X_train, y_train, cv=kfold,
        scoring="r2", n_jobs=-1
    )

    cv_results.append({
        "model": name,
        "cv_mae_mean": mae_scores.mean(),
        "cv_mae_std": mae_scores.std(),
        "cv_r2_mean": r2_scores.mean(),
        "cv_r2_std": r2_scores.std(),
    })

    print(f"{name:18s} | MAE: {mae_scores.mean():6.3f} ± {mae_scores.std():.3f} "
          f"| R²: {r2_scores.mean():6.3f} ± {r2_scores.std():.3f}")

cv_df = pd.DataFrame(cv_results).sort_values("cv_mae_mean").reset_index(drop=True)


# ------------------------------------------------
# Refit every model on the full training set and score on the held-out test set
# ------------------------------------------------

print("\nEvaluating each model on the held-out test set...\n")

test_results = []

fitted_models = {}

for name, model in candidates.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    test_results.append({
        "model": name,
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
    })

    print(f"{name:18s} | MAE: {mae:6.3f} | RMSE: {rmse:6.3f} | R²: {r2:6.3f}")

test_df = pd.DataFrame(test_results)

summary = cv_df.merge(test_df, on="model").sort_values("test_mae").reset_index(drop=True)

print("\n" + "=" * 70)
print("MODEL COMPARISON SUMMARY (sorted by test MAE, lower is better)")
print("=" * 70)
print(summary.to_string(index=False))


# ------------------------------------------------
# Pick and save the best model
# ------------------------------------------------

best_name = summary.iloc[0]["model"]
best_model = fitted_models[best_name]

print(f"\nBest model: {best_name}")
print(f"Test MAE : {summary.iloc[0]['test_mae']:.3f}")
print(f"Test R²  : {summary.iloc[0]['test_r2']:.3f}")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, MODEL_PATH)
summary.to_csv(RESULTS_PATH, index=False)

print(f"\nBest model saved to: {MODEL_PATH}")
print(f"Full comparison saved to: {RESULTS_PATH}")