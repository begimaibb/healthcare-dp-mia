import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

from differential_privacy import apply_laplace_noise, privacy_budget_report

MODELS_DIR = os.environ.get("MODELS_DIR", "models/")
EPSILON = float(os.environ.get("DP_EPSILON", "0.5"))

FEATURE_RANGES = {
    "age":          (0,   120),
    "gender_male":  (0,   1),
    "gender_female":(0,   1),
    "latest_bmi":   (10,  60),
    "latest_a1c":   (3,   15),
}

app = Flask(__name__)

model = joblib.load(os.path.join(MODELS_DIR, "disease_risk_model.pkl"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

print(f"Model loaded. Features: {feature_names}")
print(f"Privacy budget: {privacy_budget_report(EPSILON)}")


def validate_input(data: dict):
    features = []
    for feat in feature_names:
        if feat not in data:
            return None, f"Missing required field: '{feat}'"
        val = data[feat]
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None, f"Field '{feat}' must be numeric"
        lo, hi = FEATURE_RANGES.get(feat, (-np.inf, np.inf))
        if not (lo <= val <= hi):
            return None, f"Field '{feat}' out of range [{lo}, {hi}], got {val}"
        features.append(val)
    return np.array(features).reshape(1, -1), None


@app.route("/predict_risk", methods=["POST"])
def predict_risk():
    data = request.get_json(force=True, silent=True)
    if data is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    features, error = validate_input(data)
    if error:
        return jsonify({"error": error}), 422

    raw_prob = model.predict_proba(features)[:, 1][0]
    protected_prob = apply_laplace_noise(raw_prob, epsilon=EPSILON)

    return jsonify({
        "risk_score": round(protected_prob, 4),
        "epsilon": EPSILON,
        "privacy_level": privacy_budget_report(EPSILON),
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "features": feature_names,
        "epsilon": EPSILON,
        "privacy_level": privacy_budget_report(EPSILON),
        "dp_mechanism": "Laplace",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)