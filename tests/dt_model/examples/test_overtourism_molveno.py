"""Tests for the Molveno overtourism example using the new Evaluation.evaluate(axes=...) API."""

# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import pytest

from civic_digital_twins.dt_model import Evaluation
from civic_digital_twins.dt_model.model.index import Distribution
from overtourism_molveno.constraint import Constraint
from overtourism_molveno.ensemble import OvertourismEnsemble
from overtourism_molveno.overtourism import (
    CV_season,
    CV_weather,
    CV_weekday,
    M_Base,
    PV_excursionists,
    PV_tourists,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_field(model, scenarios, tt, ee):
    """Evaluate the sustainability field using Evaluation.evaluate(axes=...).

    Returns ``(field, field_elements, state)`` where:
    - ``field`` has shape ``(tt.size, ee.size)``
    - ``field_elements`` maps each Constraint to a ``(tt.size, ee.size)`` array
    - ``state`` is the raw executor State
    """
    weights = np.array([w for w, _ in scenarios])

    nodes = [c.usage for c in model.constraints]

    state = Evaluation(model).evaluate(
        scenarios, nodes, axes={PV_tourists: tt, PV_excursionists: ee}
    )

    S = len(scenarios)
    full_shape = (tt.size, ee.size, S)

    field = np.ones((tt.size, ee.size))
    field_elements = {}
    for c in model.constraints:
        # Broadcast to full shape in case the formula doesn't depend on all axes.
        usage = np.broadcast_to(state.values[c.usage], full_shape)
        if isinstance(c.capacity.value, Distribution):
            mask = (1.0 - c.capacity.value.cdf(usage)).astype(float)
        else:
            cap = np.broadcast_to(state.values[c.capacity.node], full_shape)
            mask = (usage <= cap).astype(float)
        field_elem = np.tensordot(mask, weights, axes=([-1], [0]))
        field_elements[c] = field_elem
        field *= field_elem

    return field, field_elements, state


def compare_constraint_results(
    got: dict[Constraint, np.ndarray],
    expect: dict[str, np.ndarray],
) -> list[str]:
    """Compare constraint results and return any failures."""
    if len(got) != len(expect):
        return [f"Constraint count mismatch: expected {len(expect)}, got {len(got)}"]

    failures: list[str] = []
    got_by_name = {c.name: result for c, result in got.items()}

    for name, expected_result in expect.items():
        if name not in got_by_name:
            failures.append(f"Constraint '{name}' not found in results")
            continue

        actual_result = got_by_name[name]

        if expected_result.shape != actual_result.shape:
            failures.append(
                f"Shape mismatch for constraint '{name}': "
                f"{expected_result.shape} vs {actual_result.shape}"
            )
            continue

        if not np.allclose(expected_result, actual_result, rtol=1e-5, atol=1e-8):
            diff_info = f"\n--- expected/{name}\n+++ got/{name}\n"
            for j in range(expected_result.shape[0]):
                row_expect = [f"{x:.8f}" for x in expected_result[j]]
                row_got = [f"{x:.8f}" for x in actual_result[j]]
                if not np.allclose(expected_result[j], actual_result[j], rtol=1e-5, atol=1e-8):
                    diff_info += f"-{row_expect}\n"
                    diff_info += f"+{row_got}\n"
                else:
                    diff_info += f" {row_expect}\n"
            failures.append(diff_info)

    for name in got_by_name:
        if name not in expect:
            failures.append(f"Unexpected constraint found: '{name}'")

    return failures


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tourists():
    return np.array([1000, 5000, 10000])


@pytest.fixture
def excursionists():
    return np.array([1000, 5000, 10000])


@pytest.fixture
def good_weather_scenarios():
    """Single-member scenario list: good weather, monday, high season."""
    np.random.seed(0)
    random.seed(0)
    return list(OvertourismEnsemble(M_Base, {
        CV_weekday: ["monday"],
        CV_season: ["high"],
        CV_weather: ["good"],
    }))


# ---------------------------------------------------------------------------
# Shape / range tests (replaces test_evaluation.py)
# ---------------------------------------------------------------------------

def test_evaluate_axes_returns_correct_shape(good_weather_scenarios, tourists, excursionists):
    """evaluate(axes=...) produces field with shape (tt.size, ee.size)."""
    field, _, _ = compute_field(M_Base, good_weather_scenarios, tourists, excursionists)
    assert field.shape == (tourists.size, excursionists.size)


def test_evaluate_axes_field_values_in_range(good_weather_scenarios, tourists, excursionists):
    """Sustainability field values are in [0, 1]."""
    field, _, _ = compute_field(M_Base, good_weather_scenarios, tourists, excursionists)
    assert np.all(field >= 0.0)
    assert np.all(field <= 1.0)


def test_evaluate_axes_field_elements_match_constraints(good_weather_scenarios, tourists, excursionists):
    """field_elements has one entry per constraint."""
    _, field_elements, _ = compute_field(M_Base, good_weather_scenarios, tourists, excursionists)
    assert len(field_elements) == len(M_Base.constraints)


def test_evaluate_axes_low_presence_is_sustainable(good_weather_scenarios):
    """Very low presence values should be fully sustainable (field ≈ 1)."""
    tt = np.array([1, 2])
    ee = np.array([1, 2])
    field, _, _ = compute_field(M_Base, good_weather_scenarios, tt, ee)
    assert np.allclose(field, 1.0)


def test_evaluate_axes_high_presence_is_unsustainable(good_weather_scenarios):
    """Very high presence values should be unsustainable (field ≈ 0)."""
    tt = np.array([500000])
    ee = np.array([500000])
    field, _, _ = compute_field(M_Base, good_weather_scenarios, tt, ee)
    assert np.allclose(field, 0.0)


def test_ensemble_based_evaluation(tourists, excursionists):
    """OvertourismEnsemble-based evaluation produces a valid sustainability field."""
    np.random.seed(42)
    random.seed(42)
    scenario = {CV_weather: ["good", "bad"]}
    ensemble = OvertourismEnsemble(M_Base, scenario, cv_ensemble_size=5)
    scenarios = list(ensemble)

    field, field_elements, _ = compute_field(M_Base, scenarios, tourists, excursionists)

    assert field.shape == (tourists.size, excursionists.size)
    assert np.all(field >= 0.0)
    assert np.all(field <= 1.0)
    assert len(field_elements) == len(M_Base.constraints)


# ---------------------------------------------------------------------------
# Regression test (replaces test_molveno.py test_fixed_ensemble)
# Field now has shape (N_t, N_e) — tourists on axis 0, excursionists on axis 1.
# Expected values are the transposes of the original test_molveno.py values.
# ---------------------------------------------------------------------------

def test_fixed_ensemble():
    """Evaluate the model using a fixed single-member ensemble (seed regression)."""
    tourists = np.array([1000, 2000, 5000, 10000, 20000, 50000])
    excursionists = np.array([1000, 2000, 5000, 10000, 20000, 50000])

    np.random.seed(4)
    random.seed(4)

    # Build single-member scenarios with distribution-backed index samples.
    ensemble = OvertourismEnsemble(M_Base, {
        CV_weekday: ["monday"],
        CV_season: ["high"],
        CV_weather: ["good"],
    })
    scenarios = list(ensemble)

    _, got, _ = compute_field(M_Base, scenarios, tourists, excursionists)

    # Expected: field_elements[t_idx, e_idx] — tourists on axis 0, excursionists on axis 1.
    # Parking: mostly excursionist-dominated; parking violated when excursionists > ~2000.
    expect: dict[str, np.ndarray] = {
        "parking": np.array(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "beach": np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "accommodation": np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [8.91250437e-01, 8.91250437e-01, 8.91250437e-01, 8.91250437e-01, 8.91250437e-01, 8.91250437e-01],
                [8.09024620e-06, 8.09024620e-06, 8.09024620e-06, 8.09024620e-06, 8.09024620e-06, 8.09024620e-06],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "food": np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.77777778, 0.0, 0.0],
                [1.0, 1.0, 0.77777778, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    }

    failures = compare_constraint_results(got, expect)
    if failures:
        assert False, "Model comparison failed:\n" + "\n".join(failures)

    assert M_Base.name == "base model"


def test_multiple_ensemble_members():
    """Test with multiple ensemble members to catch shape issues."""
    np.random.seed(0)
    random.seed(0)
    scenario = {CV_weather: ["good", "bad"]}
    ens = OvertourismEnsemble(M_Base, scenario, cv_ensemble_size=10)
    scenarios = list(ens)

    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])

    field, field_elements, _ = compute_field(M_Base, scenarios, tourists, excursionists)

    assert field is not None
    assert field_elements is not None
    assert field.shape == (tourists.size, excursionists.size)


def test_bug_37():
    """Regression for https://github.com/fbk-most/dt-model/issues/37."""
    np.random.seed(0)
    random.seed(0)
    situation = {CV_weather: ["good", "unsettled", "bad"]}
    ensemble = OvertourismEnsemble(M_Base, situation, cv_ensemble_size=20)
    scenarios = list(ensemble)

    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])

    field, field_elements, _ = compute_field(M_Base, scenarios, tourists, excursionists)

    assert field is not None
    assert field_elements is not None
    assert field.shape == (tourists.size, excursionists.size)
