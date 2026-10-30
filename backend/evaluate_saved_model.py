
from pathlib import Path
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "car_price_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "car_price_pipeline.joblib"
OUTPUT_PATH = PROJECT_ROOT / "models" / "evaluation.json"

TARGET = "selling_price"
FEATURES = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

# IMPORTANT:
# These must match the test split settings used during training.
TEST_SIZE = 0.20
RANDOM_STATE = 42


def calculate_metrics(actual, predicted):
    """Calculate regression metrics in the original INR price scale."""
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 2
        ),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def main():
    print("=" * 60)
    print("          CAR PRICE MODEL EVALUATION")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # Use only the required columns and remove invalid target rows.
    df = df[required_columns].copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=required_columns)
    df = df[df[TARGET] > 0].reset_index(drop=True)

    X = df[FEATURES]
    y = df[TARGET]

    # Reproduce the training split exactly if these settings match training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = joblib.load(MODEL_PATH)

    # Predictions are returned in the original target units by the
    # TransformedTargetRegressor, if that is how the pipeline was saved.
    train_predictions = np.asarray(model.predict(X_train)).reshape(-1)
    test_predictions = np.asarray(model.predict(X_test)).reshape(-1)

    if not np.isfinite(train_predictions).all():
        raise ValueError("Training predictions contain invalid values.")

    if not np.isfinite(test_predictions).all():
        raise ValueError("Test predictions contain invalid values.")

    train_metrics = calculate_metrics(y_train, train_predictions)
    test_metrics = calculate_metrics(y_test, test_predictions)

    # Extract the fitted tree importances from the preprocessing pipeline.
    fitted_regressor = getattr(model, "regressor_", model)
    pipeline = fitted_regressor
    estimator = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        raise ValueError("The saved model does not expose feature importances.")
    feature_importance = [
        {"feature": str(name), "importance": round(float(value), 6)}
        for name, value in sorted(
            zip(feature_names, importances), key=lambda item: item[1], reverse=True
        )[:12]
    ]

    # Keep a small sample for the actual-versus-predicted chart.
    sample_count = min(200, len(y_test))
    sample = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": test_predictions,
    }).head(sample_count)

    actual_vs_predicted = [
        {
            "actual": round(float(row.actual), 2),
            "predicted": round(float(row.predicted), 2),
        }
        for row in sample.itertuples(index=False)
    ]

    # Summarize errors across price bands.
    evaluation_rows = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": test_predictions,
    })
    evaluation_rows["absolute_error"] = (
        evaluation_rows["actual"] - evaluation_rows["predicted"]
    ).abs()

    evaluation_rows["price_range"] = pd.cut(
        evaluation_rows["actual"],
        bins=[0, 100_000, 300_000, 500_000, 1_000_000, np.inf],
        labels=[
            "Under ₹1 lakh",
            "₹1–3 lakh",
            "₹3–5 lakh",
            "₹5–10 lakh",
            "Over ₹10 lakh",
        ],
        include_lowest=True,
    )

    price_range_metrics = []
    for price_range, group in evaluation_rows.groupby(
        "price_range", observed=False
    ):
        if group.empty:
            continue

        price_range_metrics.append({
            "range": str(price_range),
            "count": int(len(group)),
            "mae": round(float(group["absolute_error"].mean()), 2),
        })

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "currency": "INR",
        "model_name": "Log-target Extra Trees",
        "dataset": {
            "rows_evaluated": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features": FEATURES,
            "target": TARGET,
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
        "feature_importance": feature_importance,
        "price_range_metrics": price_range_metrics,
        "actual_vs_predicted": actual_vs_predicted,
        "notes": [
            "Metrics are calculated in the original INR price scale.",
            "Holdout results are valid only if the split matches training.",
            "Predictions are estimates, not guaranteed selling prices.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nDataset rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")

    print("\nTRAIN METRICS")
    for metric, value in train_metrics.items():
        print(f"{metric.upper():<8}: {value}")

    print("\nTEST METRICS")
    for metric, value in test_metrics.items():
        print(f"{metric.upper():<8}: {value}")

    print(f"\nEvaluation results saved to:\n{OUTPUT_PATH}")
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()
