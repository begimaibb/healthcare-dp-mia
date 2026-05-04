import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from differential_privacy import apply_laplace_noise


def mia_objective(candidate, model, target_prob, feature_names):
    df = pd.DataFrame([candidate], columns=feature_names)
    prob = model.predict_proba(df)[:, 1][0]
    return (prob - target_prob) ** 2


def run_inversion_attack(model, target_features, target_prob, feature_names, n_restarts=5, bounds=None):
    best_result = None
    best_loss = float("inf")
    n_features = len(target_features)

    for _ in range(n_restarts):
        x0 = np.random.uniform(0, 1, size=n_features)
        result = minimize(
            mia_objective, x0,
            args=(model, target_prob, feature_names),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_result = result.x

    return best_result


def compute_rmse(true, reconstructed):
    return float(np.sqrt(np.mean((true - reconstructed) ** 2)))


def evaluate(models_dir: str, epsilon: float, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    model = joblib.load(os.path.join(models_dir, "disease_risk_model.pkl"))
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
    X_test, y_test = joblib.load(os.path.join(models_dir, "test_data.pkl"))

    positive_indices = np.where(y_test == 1)[0]
    target_idx = positive_indices[0]
    target = np.array(X_test.iloc[target_idx])

    bounds = [(0, 120), (0, 1), (0, 1), (10, 60), (3, 15)]

    print("=" * 55)
    print("  Model Inversion Attack — Security Analysis")
    print("=" * 55)

    raw_prob = model.predict_proba(X_test.iloc[[target_idx]])[:, 1][0]
    recon_unprotected = run_inversion_attack(model, target, raw_prob, feature_names, bounds=bounds)
    rmse_unprotected = compute_rmse(target, recon_unprotected)

    print(f"\n[Unprotected] RMSE: {rmse_unprotected:.4f}")

    protected_prob = apply_laplace_noise(raw_prob, epsilon=epsilon)
    recon_protected = run_inversion_attack(model, target, protected_prob, feature_names, bounds=bounds)
    rmse_protected = compute_rmse(target, recon_protected)

    print(f"[Protected ε={epsilon}] RMSE: {rmse_protected:.4f}")

    improvement = ((rmse_protected - rmse_unprotected) / rmse_unprotected) * 100
    print(f"\n✅ DP increased reconstruction error by {improvement:.1f}%")

    print("\n--- Per-feature reconstruction ---")
    print(f"{'Feature':<18} {'True':>8} {'Unprotected':>12} {'Protected':>10}")
    print("-" * 52)
    for i, feat in enumerate(feature_names):
        print(f"{feat:<18} {target[i]:>8.2f} {recon_unprotected[i]:>12.2f} {recon_protected[i]:>10.2f}")

    joblib.dump(recon_unprotected, os.path.join(models_dir, "reconstructed_unprotected.pkl"))
    joblib.dump(recon_protected, os.path.join(models_dir, "reconstructed_protected.pkl"))

    errors_unp = recon_unprotected - target
    errors_prot = recon_protected - target
    x = np.arange(len(feature_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, errors_unp, width, label="Unprotected", color="#4C72B0")
    ax.bar(x + width / 2, errors_prot, width, label="DP-Protected", color="#DD8452")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Feature-Level Reconstruction Error: Unprotected vs DP-Protected")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mitigation_effectiveness_kpi.png"), dpi=150)
    plt.close()

    return {"rmse_unprotected": rmse_unprotected, "rmse_protected": rmse_protected, "improvement_pct": improvement}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", default="models/")
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()
    evaluate(args.models_dir, args.epsilon, args.results_dir)