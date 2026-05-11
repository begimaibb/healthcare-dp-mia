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


def run_inversion_attack(model, target_prob, feature_names, initial_guess, bounds):
    result = minimize(
        mia_objective, initial_guess,
        args=(model, target_prob, feature_names),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 500},
    )
    return result.x


def compute_rmse(true, reconstructed):
    return float(np.sqrt(np.mean((true - reconstructed) ** 2)))


def evaluate(models_dir: str, epsilon: float, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    model         = joblib.load(os.path.join(models_dir, "disease_risk_model.pkl"))
    feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
    X_test, y_test = joblib.load(os.path.join(models_dir, "test_data.pkl"))

    initial_guess = X_test.mean().values
    bounds = [(18,100),(0,1),(0,1),(15,60),(4,15)]

    print("=" * 55)
    print("  Model Inversion Attack — Security Analysis")
    print("  Evaluated across ALL diabetic patients")
    print("=" * 55)

    improvements = []
    all_rmse_u   = []
    all_rmse_p   = []

    for i in range(len(y_test)):
        if y_test.iloc[i] != 1:
            continue
        target   = np.array(X_test.iloc[i])
        raw_prob = model.predict_proba(X_test.iloc[[i]])[:, 1][0]
        prot_prob = float(np.clip(raw_prob + np.random.laplace(0, 1/epsilon), 0, 1))

        recon_u = run_inversion_attack(model, raw_prob,  feature_names, initial_guess, bounds)
        recon_p = run_inversion_attack(model, prot_prob, feature_names, initial_guess, bounds)

        rmse_u = compute_rmse(target, recon_u)
        rmse_p = compute_rmse(target, recon_p)
        imp    = (rmse_p - rmse_u) / rmse_u * 100

        all_rmse_u.append(rmse_u)
        all_rmse_p.append(rmse_p)
        improvements.append(imp)

    avg_u   = np.mean(all_rmse_u)
    avg_p   = np.mean(all_rmse_p)
    avg_imp = np.mean(improvements)
    med_imp = np.median(improvements)

    print(f"\n  Patients evaluated : {len(improvements)}")
    print(f"  Avg RMSE unprotected: {avg_u:.4f}")
    print(f"  Avg RMSE protected  : {avg_p:.4f}")
    print(f"\n  Average improvement : {avg_imp:.1f}%")
    print(f"  Median improvement  : {med_imp:.1f}%")
    print(f"  Min / Max           : {np.min(improvements):.1f}% / {np.max(improvements):.1f}%")

    if avg_imp > 0:
        print(f"\n✅ DP increased reconstruction error by {avg_imp:.1f}% on average")
    else:
        print(f"\n❌ Mitigation had limited average effect ({avg_imp:.1f}%)")

    # Plot distribution of improvements
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart: average RMSE
    ax1.bar(["Unprotected", f"DP-Protected\n(ε={epsilon})"],
            [avg_u, avg_p], color=["#4C72B0", "#DD8452"])
    ax1.set_ylabel("Average Reconstruction RMSE")
    ax1.set_title("Average Attack RMSE Across All Patients")
    ax1.grid(axis="y", alpha=0.3)

    # Histogram: improvement distribution
    ax2.hist(improvements, bins=20, color="#4C72B0", edgecolor="white")
    ax2.axvline(avg_imp, color="#DD8452", linestyle="--", label=f"Mean: {avg_imp:.1f}%")
    ax2.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("RMSE Improvement (%)")
    ax2.set_ylabel("Number of Patients")
    ax2.set_title("Distribution of DP Improvement Across Patients")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "mitigation_effectiveness_kpi.png"), dpi=150)
    plt.close()
    print(f"\nPlot saved → {results_dir}/mitigation_effectiveness_kpi.png")

    return {"avg_rmse_unprotected": avg_u, "avg_rmse_protected": avg_p,
            "avg_improvement_pct": avg_imp, "median_improvement_pct": med_imp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir",  default="models/")
    parser.add_argument("--epsilon",     type=float, default=0.5)
    parser.add_argument("--results-dir", default="results/")
    args = parser.parse_args()
    evaluate(args.models_dir, args.epsilon, args.results_dir)