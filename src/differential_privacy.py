import numpy as np


def apply_laplace_noise(probability: float, epsilon: float = 0.5) -> float:
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    noise = np.random.laplace(loc=0.0, scale=1.0 / epsilon)
    return float(np.clip(probability + noise, 0.0, 1.0))


def apply_gaussian_noise(probability: float, epsilon: float = 0.5, delta: float = 1e-5) -> float:
    if epsilon <= 0 or delta <= 0:
        raise ValueError("epsilon and delta must be positive")
    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(loc=0.0, scale=sigma)
    return float(np.clip(probability + noise, 0.0, 1.0))


def privacy_budget_report(epsilon: float) -> str:
    if epsilon < 0.3:
        level = "Strong  — high noise, lower utility"
    elif epsilon < 1.0:
        level = "Moderate — balanced noise and utility"
    elif epsilon < 5.0:
        level = "Weak    — low noise, higher utility"
    else:
        level = "Minimal  — negligible privacy protection"
    return f"ε={epsilon:.2f} → {level}"
