import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from differential_privacy import apply_laplace_noise, apply_gaussian_noise, privacy_budget_report


class TestDifferentialPrivacy:

    def test_laplace_output_in_range(self):
        for _ in range(200):
            result = apply_laplace_noise(0.5, epsilon=0.5)
            assert 0.0 <= result <= 1.0, f"Out of range: {result}"

    def test_laplace_extreme_probabilities(self):
        assert 0.0 <= apply_laplace_noise(0.0, epsilon=0.5) <= 1.0
        assert 0.0 <= apply_laplace_noise(1.0, epsilon=0.5) <= 1.0

    def test_laplace_invalid_epsilon(self):
        with pytest.raises(ValueError):
            apply_laplace_noise(0.5, epsilon=0.0)
        with pytest.raises(ValueError):
            apply_laplace_noise(0.5, epsilon=-1.0)

    def test_higher_epsilon_less_noise(self):
        n = 2000
        probs = [0.5] * n
        low_eps  = [abs(apply_laplace_noise(p, epsilon=0.1) - p) for p in probs]
        high_eps = [abs(apply_laplace_noise(p, epsilon=5.0) - p) for p in probs]
        assert np.mean(low_eps) > np.mean(high_eps)

    def test_gaussian_output_in_range(self):
        for _ in range(200):
            result = apply_gaussian_noise(0.5, epsilon=0.5)
            assert 0.0 <= result <= 1.0

    def test_privacy_budget_report_labels(self):
        assert "Strong" in privacy_budget_report(0.1)
        assert "Moderate" in privacy_budget_report(0.5)
        assert "Weak" in privacy_budget_report(2.0)
        assert "Minimal" in privacy_budget_report(10.0)
