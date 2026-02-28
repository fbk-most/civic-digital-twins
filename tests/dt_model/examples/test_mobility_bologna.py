"""Regression tests for the Bologna mobility example.

With a fixed random seed the KPI outputs must remain stable across changes.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from mobility_bologna.mobility_bologna import BolognaModel, compute_kpis, evaluate

# Ensemble size used in all tests. Small enough to be fast, large enough to
# exercise the ensemble path (distribution-backed indexes are sampled).
_ENSEMBLE_SIZE = 5


@pytest.fixture(scope="module")
def model():
    """Shared BolognaModel instance (graph construction is the expensive part)."""
    return BolognaModel()


@pytest.fixture
def evaluation(model):
    """Evaluate the model with a fixed seed and return the subs dict."""
    np.random.seed(42)
    return evaluate(model, size=_ENSEMBLE_SIZE)


def test_base_inflow_is_deterministic(model):
    """Base inflow depends only on fixed data — must be identical across seeds."""
    np.random.seed(0)
    subs_a = evaluate(model, size=_ENSEMBLE_SIZE)
    np.random.seed(99)
    subs_b = evaluate(model, size=_ENSEMBLE_SIZE)
    kpis_a = compute_kpis(model, subs_a)
    kpis_b = compute_kpis(model, subs_b)
    assert kpis_a["Base inflow [veh/day]"] == kpis_b["Base inflow [veh/day]"]


def test_kpis_stable_with_fixed_seed(model, evaluation):
    """All KPIs must reproduce exactly when the random seed is fixed."""
    kpis = compute_kpis(model, evaluation)

    expected = {
        "Base inflow [veh/day]": 168139,
        "Modified inflow [veh/day]": 141044,
        "Shifted inflow [veh/day]": 5357,
        "Paying inflow [veh/day]": 77118,
        "Collected fees [€/day]": 294670,
        "Emissions [NOx gr/day]": 508105,
        "Modified emissions [NOx gr/day]": 104427,
    }

    assert kpis == expected
