# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Bologna mobility example.

With a fixed random seed the KPI outputs must remain stable across changes.
"""

from __future__ import annotations

import numpy as np
import pytest
from mobility_bologna.bologna_model import BolognaEvaluator, BolognaModel

from civic_digital_twins.dt_model import Scenario
from civic_digital_twins.dt_model.simulation.runner import EvaluationConfig

_ENSEMBLE_SIZE = 5


@pytest.fixture(scope="module")
def model() -> BolognaModel:
    """Shared BolognaModel instance (graph construction is the expensive part)."""
    return BolognaModel(**BolognaModel.default_inputs())


@pytest.fixture(scope="module")
def evaluator(model: BolognaModel) -> BolognaEvaluator:
    """Shared evaluator built on the shared model."""
    return BolognaEvaluator(model)


def test_base_inflow_is_deterministic(evaluator: BolognaEvaluator, model: BolognaModel) -> None:
    """Base inflow depends only on fixed data — must be identical across seeds."""
    config = EvaluationConfig(ensemble_size=_ENSEMBLE_SIZE)
    np.random.seed(0)
    out_a = evaluator.evaluate(Scenario(model), config)
    np.random.seed(99)
    out_b = evaluator.evaluate(Scenario(model), config)
    assert out_a.kpis["Base inflow [veh/day]"] == out_b.kpis["Base inflow [veh/day]"]


def test_kpis_stable_with_fixed_seed(evaluator: BolognaEvaluator, model: BolognaModel) -> None:
    """All KPIs must reproduce exactly when the random seed is fixed."""
    config = EvaluationConfig(ensemble_size=_ENSEMBLE_SIZE)
    np.random.seed(42)
    out = evaluator.evaluate(Scenario(model), config)
    expected = {
        "Base inflow [veh/day]": 168139,
        "Modified inflow [veh/day]": 141044,
        "Shifted inflow [veh/day]": 5357,
        "Paying inflow [veh/day]": 77118,
        "Collected fees [€/day]": 294670,
        "Emissions [NOx gr/day]": 508105,
        "Modified emissions [NOx gr/day]": 104427,
    }
    assert out.kpis == expected
