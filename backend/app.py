
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Resolve the model relative to this file, not the terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "car_price_pipeline.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

try:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
except Exception:
    app.logger.exception("Could not load model at %s", MODEL_PATH)
    model = None


def resolve_model_name(loaded_model):
    if loaded_model is None:
        return None
    try:
        metadata = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        if metadata.get("model_name"):
            return metadata["model_name"]
    except (OSError, json.JSONDecodeError):
        pass

    is_transformed = hasattr(loaded_model, "regressor_")
    pipeline = getattr(loaded_model, "regressor_", loaded_model)
    estimator = pipeline.named_steps.get("model")
    estimator_names = {
        "ExtraTreesRegressor": "Extra Trees",
        "RandomForestRegressor": "Random Forest",
        "GradientBoostingRegressor": "Gradient Boosting",
    }
    name = estimator_names.get(type(estimator).__name__, type(estimator).__name__)
    return f"Log-target {name}" if is_transformed else name


MODEL_NAME = resolve_model_name(model)

REQUIRED_FIELDS = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

CATEGORICAL_FIELDS = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]


@app.get("/")
def home():
    return jsonify({
        "message": "Car Price Predictor API",
        "status": "running",
        "endpoints": ["/health", "/predict"],
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
    })


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({
            "error": "Prediction model is unavailable. Train or restore the model file."
        }), 503
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    missing = [
        field for field in REQUIRED_FIELDS
        if field not in data
        or data[field] is None
        or (isinstance(data[field], str) and not data[field].strip())
    ]

    if missing:
        return jsonify({
            "error": "Missing required fields.",
            "fields": missing,
        }), 400

    try:
        year_value = data["year"]
        km_value = data["km_driven"]

        # Reject booleans, strings and non-finite numeric values.
        if isinstance(year_value, bool) or not isinstance(
            year_value, (int, float)
        ):
            raise ValueError("'year' must be a number.")

        if isinstance(km_value, bool) or not isinstance(
            km_value, (int, float)
        ):
            raise ValueError("'km_driven' must be a number.")

        if not np.isfinite(year_value) or not float(year_value).is_integer():
            raise ValueError("'year' must be a finite integer.")

        if not np.isfinite(km_value) or not float(km_value).is_integer():
            raise ValueError("'km_driven' must be a finite integer.")

        year = int(year_value)
        km_driven = int(km_value)

        if not 1992 <= year <= 2026:
            raise ValueError("'year' must be between 1992 and 2026.")

        if km_driven < 0:
            raise ValueError("'km_driven' cannot be negative.")

        cleaned = {}

        for field in CATEGORICAL_FIELDS:
            value = data[field]

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"'{field}' must be a non-empty string.")

            cleaned[field] = value.strip()

        input_df = pd.DataFrame([{
            "name": cleaned["name"],
            "year": year,
            "km_driven": km_driven,
            "fuel": cleaned["fuel"],
            "seller_type": cleaned["seller_type"],
            "transmission": cleaned["transmission"],
            "owner": cleaned["owner"],
        }])

        prediction = float(model.predict(input_df)[0])

        if not np.isfinite(prediction) or prediction <= 0:
            return jsonify({
                "error": "The model returned an invalid price estimate."
            }), 500

        return jsonify({
            "predicted_price": round(prediction, 2),
            "currency": "INR",
            "model": MODEL_NAME,
            "note": (
                "Estimate only; actual selling prices depend on "
                "vehicle condition, location and market conditions."
            ),
        })

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the backend logs."
        }), 500


@app.get("/evaluation")
def evaluation():
    """Serve the saved holdout evaluation and dataset summary."""
    path = PROJECT_ROOT / "models" / "evaluation.json"
    if not path.exists():
        return jsonify({
            "error": "Evaluation is unavailable. Run backend/evaluate_models.py first."
        }), 503
    try:
        return jsonify(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        app.logger.exception("Could not load evaluation artifact")
        return jsonify({"error": "Evaluation data could not be loaded."}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
