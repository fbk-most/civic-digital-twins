"""Tests for the Molveno overtourism example using the new Evaluation.evaluate(axes=...) API."""

# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest
from overtourism_molveno.molveno_model import (
    Constraint,
    MolvenoModel,
)

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    ConditionalDistributionIndex,
    CrossProductEnsemble,
    Evaluation,
    ModelContractWarning,
    Scenario,
)
from civic_digital_twins.dt_model.model.index import Distribution, DistributionIndex, GenericIndex, Index

model = MolvenoModel(inputs=MolvenoModel.default_inputs())
_pvs = [model.inputs.pv_tourists, model.inputs.pv_excursionists]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_field(model, ensemble, tt, ee):
    """Evaluate the sustainability field using Evaluation.evaluate(parameters=...).

    Returns ``(field, field_elements, result)`` where:
    - ``field`` has shape ``(tt.size, ee.size)``
    - ``field_elements`` maps each Constraint to a ``(tt.size, ee.size)`` array
    - ``result`` is the :class:`~dt_model.simulation.evaluation.EvaluationResult`
    """
    result = Evaluation(Scenario(model, parameter_axes=_pvs)).evaluate(
        ensemble=ensemble, parameters={model.inputs.pv_tourists: tt, model.inputs.pv_excursionists: ee}
    )

    field = np.ones((tt.size, ee.size))
    field_elements = {}
    for c in model.constraints:
        # Broadcast to full shape in case the formula doesn't depend on all axes.
        usage = np.broadcast_to(result[c.usage], result.full_shape)
        if isinstance(c.capacity.value, Distribution):
            mask = (1.0 - c.capacity.value.cdf(usage)).astype(float)
        else:
            cap = np.broadcast_to(result[c.capacity], result.full_shape)
            mask = (usage <= cap).astype(float)
        field_elem = np.tensordot(mask, result.weights, axes=([-1], [0]))
        field_elements[c] = field_elem
        field *= field_elem

    return field, field_elements, result


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
            failures.append(f"Shape mismatch for constraint '{name}': {expected_result.shape} vs {actual_result.shape}")
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
    """Tourist presence axis: low, medium, high."""
    return np.array([1000, 5000, 10000])


@pytest.fixture
def excursionists():
    """Excursionist presence axis: low, medium, high."""
    return np.array([1000, 5000, 10000])


@pytest.fixture
def good_weather_scenarios():
    """Single-member ensemble: good weather, monday, high season."""
    return CrossProductEnsemble(
        Scenario(
            model,
            overrides={
                model.inputs.cv_weekday: ["monday"],
                model.inputs.cv_season: ["high"],
                model.inputs.cv_weather: ["good"],
            },
            parameter_axes=_pvs,
        ),
    )


# ---------------------------------------------------------------------------
# Shape / range tests (replaces test_evaluation.py)
# ---------------------------------------------------------------------------


def test_evaluate_axes_returns_correct_shape(good_weather_scenarios, tourists, excursionists):
    """evaluate(axes=...) produces field with shape (tt.size, ee.size)."""
    field, _, _ = compute_field(model, good_weather_scenarios, tourists, excursionists)
    assert field.shape == (tourists.size, excursionists.size)


def test_evaluate_axes_field_values_in_range(good_weather_scenarios, tourists, excursionists):
    """Sustainability field values are in [0, 1]."""
    field, _, _ = compute_field(model, good_weather_scenarios, tourists, excursionists)
    assert np.all(field >= 0.0)
    assert np.all(field <= 1.0)


def test_evaluate_axes_field_elements_match_constraints(good_weather_scenarios, tourists, excursionists):
    """field_elements has one entry per constraint."""
    _, field_elements, _ = compute_field(model, good_weather_scenarios, tourists, excursionists)
    assert len(field_elements) == len(model.constraints)


def test_evaluate_axes_low_presence_is_sustainable(good_weather_scenarios):
    """Very low presence values should be fully sustainable (field ≈ 1)."""
    tt = np.array([1, 2])
    ee = np.array([1, 2])
    field, _, _ = compute_field(model, good_weather_scenarios, tt, ee)
    assert np.allclose(field, 1.0)


def test_evaluate_axes_high_presence_is_unsustainable(good_weather_scenarios):
    """Very high presence values should be unsustainable (field ≈ 0)."""
    tt = np.array([500000])
    ee = np.array([500000])
    field, _, _ = compute_field(model, good_weather_scenarios, tt, ee)
    assert np.allclose(field, 0.0)


def test_ensemble_based_evaluation(tourists, excursionists):
    """CrossProductEnsemble-based evaluation produces a valid sustainability field."""
    ensemble = CrossProductEnsemble(
        Scenario(model, overrides={model.inputs.cv_weather: ["good", "bad"]}, parameter_axes=_pvs),
        max_categorical_size=5,
    )

    field, field_elements, _ = compute_field(model, ensemble, tourists, excursionists)

    assert field.shape == (tourists.size, excursionists.size)
    assert np.all(field >= 0.0)
    assert np.all(field <= 1.0)
    assert len(field_elements) == len(model.constraints)


# ---------------------------------------------------------------------------
# Regression test (replaces test_molveno.py test_fixed_ensemble)
# Field now has shape (N_t, N_e) — tourists on axis 0, excursionists on axis 1.
# Expected values are the transposes of the original test_molveno.py values.
# ---------------------------------------------------------------------------


def test_fixed_ensemble():
    """Evaluate the model using a fixed single-member ensemble (seed regression)."""
    tourists = np.array([1000, 2000, 5000, 10000, 20000, 50000])
    excursionists = np.array([1000, 2000, 5000, 10000, 20000, 50000])

    # Build single-member scenarios with distribution-backed index samples.
    ensemble = CrossProductEnsemble(
        Scenario(
            model,
            overrides={
                model.inputs.cv_weekday: ["monday"],
                model.inputs.cv_season: ["high"],
                model.inputs.cv_weather: ["good"],
            },
            parameter_axes=_pvs,
        ),
        rng=np.random.default_rng(4),
    )
    _, got, _ = compute_field(model, ensemble, tourists, excursionists)

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
                [1.0, 1.0, 0.82080114, 0.0, 0.0, 0.0],
                [1.0, 0.91611209, 0.0, 0.0, 0.0, 0.0],
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

    assert model.name == "base model"


def test_multiple_ensemble_members():
    """Test with multiple ensemble members to catch shape issues."""
    ens = CrossProductEnsemble(
        Scenario(model, overrides={model.inputs.cv_weather: ["good", "bad"]}, parameter_axes=_pvs),
        max_categorical_size=10,
    )
    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])

    field, field_elements, _ = compute_field(model, ens, tourists, excursionists)

    assert field is not None
    assert field_elements is not None
    assert field.shape == (tourists.size, excursionists.size)


# ---------------------------------------------------------------------------
# MolvenoModel structure — CVs, PVs, sub-models
# ---------------------------------------------------------------------------


def test_molveno_model_expose_has_four_sub_model_proxies():
    """MolvenoModel.expose holds output proxies for all four concern sub-models."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    assert m.expose.parking is not None
    assert m.expose.beach is not None
    assert m.expose.accommodation is not None
    assert m.expose.food is not None


def test_molveno_model_exposes_pvs():
    """MolvenoModel exposes pv_tourists and pv_excursionists as attributes."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    assert isinstance(m.inputs.pv_tourists, ConditionalDistributionIndex)
    assert isinstance(m.inputs.pv_excursionists, ConditionalDistributionIndex)
    assert m.inputs.pv_tourists.name == "tourists"
    assert m.inputs.pv_excursionists.name == "excursionists"


def test_molveno_model_exposes_context_variables():
    """MolvenoModel exposes the three context variables as attributes."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    assert isinstance(m.inputs.cv_weekday, CategoricalIndex)
    assert isinstance(m.inputs.cv_season, CategoricalIndex)
    assert isinstance(m.inputs.cv_weather, CategoricalIndex)
    assert m.inputs.cv_weekday.name == "weekday"
    assert m.inputs.cv_season.name == "season"
    assert m.inputs.cv_weather.name == "weather"


def test_cv_probabilities_sum_to_one():
    """Each CategoricalIndex CV has outcomes summing to 1.0."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    for cv in (m.inputs.cv_weekday, m.inputs.cv_season, m.inputs.cv_weather):
        total = sum(cv.outcomes.values())
        assert abs(total - 1.0) < 1e-9, f"{cv.name}: outcomes sum to {total}"


def test_concern_model_outputs_are_generic_indexes_only():
    """Expose proxies yield only GenericIndex instances (no Constraint)."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    for proxy in (m.expose.parking, m.expose.beach, m.expose.accommodation, m.expose.food):
        for idx in proxy:
            assert isinstance(idx, GenericIndex), f"expose proxy yielded a non-GenericIndex: {type(idx)}"


def test_concern_model_inputs_include_all_i_parameters():
    """All i_* parameters are declared on MolvenoModel.Inputs."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())

    # Parking: 7 i_* params
    assert isinstance(m.inputs.i_u_tourists_parking, Index)
    assert isinstance(m.inputs.i_u_excursionists_parking, Index)
    assert isinstance(m.inputs.i_xa_tourists_per_vehicle, Index)
    assert isinstance(m.inputs.i_xa_excursionists_per_vehicle, Index)
    assert isinstance(m.inputs.i_xo_tourists_parking, Index)
    assert isinstance(m.inputs.i_xo_excursionists_parking, Index)
    assert isinstance(m.inputs.i_c_parking, DistributionIndex)

    # Beach: 5 i_* params
    assert isinstance(m.inputs.i_u_tourists_beach, Index)
    assert isinstance(m.inputs.i_u_excursionists_beach, Index)
    assert isinstance(m.inputs.i_xo_tourists_beach, DistributionIndex)
    assert isinstance(m.inputs.i_xo_excursionists_beach, Index)
    assert isinstance(m.inputs.i_c_beach, DistributionIndex)

    # Accommodation: 3 i_* params
    assert isinstance(m.inputs.i_u_tourists_accommodation, Index)
    assert isinstance(m.inputs.i_xa_tourists_accommodation, Index)
    assert isinstance(m.inputs.i_c_accommodation, DistributionIndex)

    # Food: 5 i_* params
    assert isinstance(m.inputs.i_u_tourists_food, Index)
    assert isinstance(m.inputs.i_u_excursionists_food, Index)
    assert isinstance(m.inputs.i_xa_visitors_food, Index)
    assert isinstance(m.inputs.i_xo_visitors_food, Index)
    assert isinstance(m.inputs.i_c_food, DistributionIndex)


def test_parking_outputs_index_types():
    """ParkingModel outputs are accessible via expose and contain the usage index."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    assert isinstance(m.expose.parking.i_u_parking, Index)
    assert len(m.expose.parking) == 1


def test_beach_outputs_index_types():
    """BeachModel outputs are accessible via expose and contain the usage index."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    assert isinstance(m.expose.beach.i_u_beach, Index)
    assert len(m.expose.beach) == 1


def test_accommodation_outputs_index_types():
    """AccommodationModel outputs are accessible via expose and contain the usage index."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    assert isinstance(m.expose.accommodation.i_u_accommodation, Index)
    assert len(m.expose.accommodation) == 1


def test_food_outputs_index_types():
    """FoodModel outputs are accessible via expose and contain the usage index."""
    m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    assert isinstance(m.expose.food.i_u_food, Index)
    assert len(m.expose.food) == 1


# ---------------------------------------------------------------------------
# No contract warnings at construction
# ---------------------------------------------------------------------------


def test_molveno_model_construction_is_warning_free():
    """Constructing MolvenoModel emits no ModelContractWarning or DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelContractWarning)
        warnings.simplefilter("error", DeprecationWarning)
        MolvenoModel(inputs=MolvenoModel.default_inputs())


# ---------------------------------------------------------------------------
# Root-model indexes coverage
# ---------------------------------------------------------------------------


def test_presence_cvs_in_root_indexes():
    """All three context variables appear in model.indexes."""
    cv_ids = {id(model.inputs.cv_weekday), id(model.inputs.cv_season), id(model.inputs.cv_weather)}
    root_ids = {id(idx) for idx in model.indexes}
    assert cv_ids <= root_ids


def test_presence_pvs_in_root_indexes():
    """Both presence variables appear in model.indexes."""
    pv_ids = {id(model.inputs.pv_tourists), id(model.inputs.pv_excursionists)}
    root_ids = {id(idx) for idx in model.indexes}
    assert pv_ids <= root_ids


def test_all_capacity_indexes_in_root_indexes():
    """All four capacity DistributionIndexes appear in model.indexes."""
    cap_ids = {
        id(model.inputs.i_c_parking),
        id(model.inputs.i_c_beach),
        id(model.inputs.i_c_accommodation),
        id(model.inputs.i_c_food),
    }
    root_ids = {id(idx) for idx in model.indexes}
    assert cap_ids <= root_ids


def test_beach_rotation_factor_in_root_indexes():
    """i_xo_tourists_beach (DistributionIndex in MolvenoModel.Inputs) is in model.indexes."""
    rotation_id = id(model.inputs.i_xo_tourists_beach)
    root_ids = {id(idx) for idx in model.indexes}
    assert rotation_id in root_ids


def test_usage_formula_indexes_in_root_indexes():
    """All four usage formula indexes appear in model.indexes via expose."""
    usage_ids = {
        id(model.expose.parking.i_u_parking),
        id(model.expose.beach.i_u_beach),
        id(model.expose.accommodation.i_u_accommodation),
        id(model.expose.food.i_u_food),
    }
    root_ids = {id(idx) for idx in model.indexes}
    assert usage_ids <= root_ids


def test_root_indexes_has_no_duplicates():
    """model.indexes contains no duplicate objects (identity check)."""
    ids = [id(idx) for idx in model.indexes]
    assert len(ids) == len(set(ids))


def test_beach_rotation_factor_is_abstract():
    """i_xo_tourists_beach is distribution-backed and therefore abstract."""
    assert any(idx is model.inputs.i_xo_tourists_beach for idx in model.abstract_indexes())


# ---------------------------------------------------------------------------
# MolvenoModel domain attributes (cvs, pvs, constraints)
# ---------------------------------------------------------------------------


def test_molveno_model_cvs_list():
    """model.Inputs declares the three named context variables."""
    assert isinstance(model.inputs.cv_weekday, CategoricalIndex)
    assert isinstance(model.inputs.cv_season, CategoricalIndex)
    assert isinstance(model.inputs.cv_weather, CategoricalIndex)
    assert model.inputs.cv_weekday.name == "weekday"
    assert model.inputs.cv_season.name == "season"
    assert model.inputs.cv_weather.name == "weather"


def test_molveno_model_pvs_list():
    """model.Inputs declares the two named presence variables."""
    assert isinstance(model.inputs.pv_tourists, ConditionalDistributionIndex)
    assert isinstance(model.inputs.pv_excursionists, ConditionalDistributionIndex)
    assert model.inputs.pv_tourists.name == "tourists"
    assert model.inputs.pv_excursionists.name == "excursionists"


def test_molveno_model_constraints_list():
    """model.constraints contains exactly four Constraint objects, one per concern."""
    assert len(model.constraints) == 4
    names = {c.name for c in model.constraints}
    assert names == {"parking", "beach", "accommodation", "food"}


def test_presence_transformation_indexes_in_root_indexes():
    """The four presence-transformation indexes appear in model.indexes."""
    pt_ids = {
        id(model.inputs.i_p_tourists_reduction_factor),
        id(model.inputs.i_p_excursionists_reduction_factor),
        id(model.inputs.i_p_tourists_saturation_level),
        id(model.inputs.i_p_excursionists_saturation_level),
    }
    root_ids = {id(idx) for idx in model.indexes}
    assert pt_ids <= root_ids


# ---------------------------------------------------------------------------


def test_bug_37():
    """Regression for https://github.com/fbk-most/dt-model/issues/37."""
    ensemble = CrossProductEnsemble(
        Scenario(model, overrides={model.inputs.cv_weather: ["good", "unsettled", "bad"]}, parameter_axes=_pvs),
        max_categorical_size=20,
    )

    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])

    field, field_elements, _ = compute_field(model, ensemble, tourists, excursionists)

    assert field is not None
    assert field_elements is not None
    assert field.shape == (tourists.size, excursionists.size)
