"""Compare three regressors on the training notebook's fixed holdout split.

Run from the repository root with the project's environment:
    .venv/Scripts/python.exe backend/evaluate_models.py
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "car_price_cleaned.csv"
MODEL_PATH = ROOT / "models" / "car_price_pipeline.joblib"
OUTPUT_PATH = ROOT / "models" / "evaluation.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
FEATURES = ["name", "year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
NUMERIC = ["year", "km_driven"]
CATEGORICAL = ["name", "fuel", "seller_type", "transmission", "owner"]
TEST_SIZE = 0.20
RANDOM_STATE = 42
PRICE_EDGES = [0, 100_000, 300_000, 500_000, 1_000_000, np.inf]
PRICE_LABELS = ["Under ₹1 lakh", "₹1–3 lakh", "₹3–5 lakh", "₹5–10 lakh", "Over ₹10 lakh"]
MIN_MAE_GAIN = 0.05
MAX_RMSE_REGRESSION = 0.05


def make_preprocessor():
    """Match notebooks/10_final_model_selection.ipynb exactly."""
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])


def make_pipeline(estimator):
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", estimator),
    ])


def build_candidates():
    return {
        "Extra Trees": TransformedTargetRegressor(
            regressor=make_pipeline(ExtraTreesRegressor(
                n_estimators=250, max_depth=25, min_samples_leaf=2,
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
        "Random Forest": make_pipeline(RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE,
            n_jobs=-1, min_samples_leaf=2,
        )),
        "Gradient Boosting": make_pipeline(GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3,
            min_samples_leaf=2, loss="huber", random_state=RANDOM_STATE,
        )),
    }


def metrics(actual, predicted):
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 2),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def fitted_pipeline(model):
    return model.regressor_ if isinstance(model, TransformedTargetRegressor) else model


def main():
    df = pd.read_csv(DATA_PATH)
    missing = [column for column in FEATURES + ["selling_price"] if column not in df]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    # Match the notebook's row handling and order exactly: no additional dropna,
    # filtering, sorting, or feature-derived split is performed here.
    X = df[FEATURES]
    y = df["selling_price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    if len(df) != 3577 or (len(X_train), len(X_test)) != (2861, 716):
        raise ValueError("Dataset row counts differ from the verified notebook holdout split.")

    results = {}
    fitted = {}
    for name, candidate in build_candidates().items():
        candidate.fit(X_train, y_train)
        fitted[name] = candidate
        train_predictions = candidate.predict(X_train)
        test_predictions = candidate.predict(X_test)
        if not np.isfinite(test_predictions).all():
            raise ValueError(f"{name} produced invalid test predictions.")
        actual = y_test.to_numpy()
        residual = actual - test_predictions
        range_frame = pd.DataFrame({"actual": actual, "predicted": test_predictions})
        range_frame["residual"] = range_frame.actual - range_frame.predicted
        range_frame["price_range"] = pd.cut(
            range_frame.actual, bins=PRICE_EDGES, labels=PRICE_LABELS,
            include_lowest=True,
        )
        ranges = []
        for label, group in range_frame.groupby("price_range", observed=False):
            if group.empty:
                continue
            ranges.append({
                "range": str(label),
                "count": int(len(group)),
                "mae": round(float(mean_absolute_error(group.actual, group.predicted)), 2),
                "mean_actual_minus_predicted": round(float(group.residual.mean()), 2),
                "underprediction_rate": round(float((group.predicted < group.actual).mean()), 4),
            })
        results[name] = {
            "training": metrics(y_train, train_predictions),
            "test": metrics(y_test, test_predictions),
            "price_ranges": ranges,
            "residual_diagnostics": {
                "mean_actual_minus_predicted": round(float(np.mean(residual)), 2),
                "median_actual_minus_predicted": round(float(np.median(residual)), 2),
                "underprediction_rate": round(float(np.mean(test_predictions < actual)), 4),
            },
            "_test_predictions": test_predictions,
        }

    # Choose the lowest holdout MAE for the comparison display. Since this
    # holdout is used for candidate selection, these comparisons are exploratory
    # and should not be presented as an untouched final test after selection.
    candidate_winner = min(
        results,
        key=lambda name: (
            results[name]["test"]["mae"],
            results[name]["test"]["rmse"],
            -results[name]["test"]["r2"],
        ),
    )
    incumbent_name = "Extra Trees"
    incumbent = results[incumbent_name]["test"]
    challenger = results[candidate_winner]["test"]
    meaningful_gain = (
        candidate_winner != incumbent_name
        and challenger["mae"] <= incumbent["mae"] * (1 - MIN_MAE_GAIN)
        and challenger["rmse"] <= incumbent["rmse"] * (1 + MAX_RMSE_REGRESSION)
    )
    if meaningful_gain:
        joblib.dump(fitted[candidate_winner], MODEL_PATH)
        pipeline_updated = True
        active_name = candidate_winner
    else:
        pipeline_updated = False
        active_name = incumbent_name

    active_model = fitted[active_name]
    production_model_name = (
        "Log-target Extra Trees" if active_name == "Extra Trees" else active_name
    )
    chosen_pipeline = fitted_pipeline(active_model)
    estimator = chosen_pipeline.named_steps["model"]
    feature_names = chosen_pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_importance = [
        {"feature": str(feature), "importance": round(float(value), 6)}
        for feature, value in sorted(
            zip(feature_names, estimator.feature_importances_),
            key=lambda item: item[1], reverse=True,
        )[:12]
    ]
    selected_predictions = results[active_name].pop("_test_predictions")
    for result in results.values():
        result.pop("_test_predictions", None)

    actual_vs_predicted = [
        {"actual": round(float(actual), 2), "predicted": round(float(predicted), 2)}
        for actual, predicted in zip(y_test.to_numpy(), selected_predictions)
    ]
    residual = y_test.to_numpy() - selected_predictions

    error_rows = X_test[["name", "year", "km_driven"]].reset_index(drop=True).copy()
    error_rows["actual"] = y_test.to_numpy()
    error_rows["predicted"] = selected_predictions
    error_rows["residual"] = residual
    error_rows["absolute_error"] = np.abs(residual)
    error_rows["direction"] = np.where(residual > 0, "underprediction", "overprediction")

    def serialize_errors(frame):
        return [
            {
                "name": str(row.name),
                "year": int(row.year),
                "km_driven": int(row.km_driven),
                "actual": round(float(row.actual), 2),
                "predicted": round(float(row.predicted), 2),
                "residual": round(float(row.residual), 2),
                "absolute_error": round(float(row.absolute_error), 2),
                "direction": row.direction,
            }
            for row in frame.itertuples(index=False)
        ]

    high_price_train = y_train >= 1_000_000
    overall_name_counts = X_train["name"].value_counts()
    high_price_name_counts = X_train.loc[high_price_train, "name"].value_counts()
    high_price_variants = []
    for variant, count in high_price_name_counts[high_price_name_counts <= 2].items():
        prices = y_train.loc[high_price_train & X_train["name"].eq(variant)]
        high_price_variants.append({
            "name": str(variant),
            "training_count": int(count),
            "median_price": round(float(prices.median()), 2),
        })
    high_price_variants.sort(key=lambda item: (item["training_count"], -item["median_price"]))
    rare_names = [
        {"name": str(name), "training_count": int(count)}
        for name, count in overall_name_counts[overall_name_counts <= 2].head(15).items()
    ]

    error_rows["price_range"] = pd.cut(
        error_rows.actual, bins=PRICE_EDGES, labels=PRICE_LABELS, include_lowest=True
    )
    final_ranges = []
    for label, group in error_rows.groupby("price_range", observed=False):
        if group.empty:
            continue
        final_ranges.append({
            "range": str(label),
            "count": int(len(group)),
            "mae": round(float(group.absolute_error.mean()), 2),
            "mean_actual_minus_predicted": round(float(group.residual.mean()), 2),
            "underprediction_rate": round(float((group.residual > 0).mean()), 4),
        })

    model_metadata = {
        "model_name": production_model_name,
        "candidate_name": active_name,
        "selection_metric_priority": ["test MAE (primary)", "test RMSE (tie-break)", "test R² (tie-break)"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    METADATA_PATH.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": active_name,
        "production_model_name": production_model_name,
        "candidate_winner": candidate_winner,
        "selection_metric_priority": model_metadata["selection_metric_priority"],
        "model_updated": pipeline_updated,
        "selection_note": (
            "Production priority: lowest shared-holdout MAE; RMSE and then R² break ties. "
            "A challenger must improve MAE by at least 5% without worsening RMSE by more than 5%. "
            "Since the holdout selects the candidate, these scores are exploratory, not an untouched final test."
        ),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "matches_notebook_10": True,
        },
        "models": results,
        "price_range_metrics": final_ranges,
        "feature_importance": feature_importance,
        "actual_vs_predicted": actual_vs_predicted,
        "largest_overpredictions": serialize_errors(error_rows.sort_values("residual").head(8)),
        "largest_underpredictions": serialize_errors(error_rows.sort_values("residual", ascending=False).head(8)),
        "largest_high_price_errors": serialize_errors(
            error_rows[error_rows.actual > 1_000_000]
            .sort_values("absolute_error", ascending=False).head(12)
        ),
        "residual_diagnostics": {
            "mean_actual_minus_predicted": round(float(np.mean(residual)), 2),
            "median_actual_minus_predicted": round(float(np.median(residual)), 2),
            "underprediction_rate": round(float(np.mean(selected_predictions < y_test.to_numpy())), 4),
        },
        "dataset": {
            "rows": len(df),
            "features": FEATURES,
            "median_price": round(float(y.median()), 2),
            "mean_price": round(float(y.mean()), 2),
            "price_min": round(float(y.min()), 2),
            "price_max": round(float(y.max()), 2),
            "missing_values": int(df[FEATURES + ["selling_price"]].isna().sum().sum()),
            "top_fuels": df.fuel.value_counts().head(5).to_dict(),
            "unique_names_training": int(overall_name_counts.size),
            "names_seen_once_training": int((overall_name_counts == 1).sum()),
            "names_seen_at_most_twice_training": int((overall_name_counts <= 2).sum()),
            "high_price_rows_training": int(high_price_train.sum()),
            "high_price_unique_names_training": int(high_price_name_counts.size),
            "high_price_names_seen_at_most_twice_training": int((high_price_name_counts <= 2).sum()),
            "rare_name_examples": rare_names,
            "rare_high_price_name_examples": high_price_variants[:15],
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook split verified: {len(X_train)} train / {len(X_test)} test rows (random_state=42)")
    for name, result in results.items():
        print(f"{name:18} test MAE INR {result['test']['mae']:,.2f} | RMSE INR {result['test']['rmse']:,.2f} | R2 {result['test']['r2']:.4f}")
    print(f"Lowest holdout MAE: {candidate_winner}")
    print(f"Dashboard model: {active_name}")
    print("Pipeline updated:", pipeline_updated)
    print("Evaluation saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
