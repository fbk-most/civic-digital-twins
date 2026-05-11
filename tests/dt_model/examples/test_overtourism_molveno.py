"""Tests for the Molveno overtourism example using the new Evaluation.evaluate(axes=...) API."""

# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pytest
from overtourism_molveno.molveno_model import (
    AccommodationModel,
    BeachModel,
    Constraint,
    FoodModel,
    MolvenoModel,
    ParkingModel,
)

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    ConditionalDistributionIndex,
    CrossProductEnsemble,
    Evaluation,
    ModelContractWarning,
)
from civic_digital_twins.dt_model.model.index import Distribution, DistributionIndex, GenericIndex, Index

model = MolvenoModel()

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
    result = Evaluation(model).evaluate(
        ensemble=ensemble, parameters={model.pv_tourists: tt, model.pv_excursionists: ee}
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
        model,
        restrictions={
            model.cv_weekday: ["monday"],
            model.cv_season: ["high"],
            model.cv_weather: ["good"],
        },
        exclude=model.pvs,
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
    scenario: dict[CategoricalIndex, list[str]] = {model.cv_weather: ["good", "bad"]}
    ensemble = CrossProductEnsemble(model, restrictions=scenario, max_categorical_size=5, exclude=model.pvs)

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
        model,
        restrictions={
            model.cv_weekday: ["monday"],
            model.cv_season: ["high"],
            model.cv_weather: ["good"],
        },
        exclude=model.pvs,
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

    assert model.name == "base model"


def test_multiple_ensemble_members():
    """Test with multiple ensemble members to catch shape issues."""
    scenario: dict[CategoricalIndex, list[str]] = {model.cv_weather: ["good", "bad"]}
    ens = CrossProductEnsemble(model, restrictions=scenario, max_categorical_size=10, exclude=model.pvs)
    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])

    field, field_elements, _ = compute_field(model, ens, tourists, excursionists)

    assert field is not None
    assert field_elements is not None
    assert field.shape == (tourists.size, excursionists.size)


# ---------------------------------------------------------------------------
# MolvenoModel structure — CVs, PVs, sub-models
# ---------------------------------------------------------------------------


def test_molveno_model_has_four_sub_models():
    """MolvenoModel exposes all four concern sub-models as named attributes."""
    m = MolvenoModel()
    assert isinstance(m.parking, ParkingModel)
    assert isinstance(m.beach, BeachModel)
    assert isinstance(m.accommodation, AccommodationModel)
    assert isinstance(m.food, FoodModel)


def test_molveno_model_exposes_pvs():
    """MolvenoModel exposes pv_tourists and pv_excursionists as attributes."""
    m = MolvenoModel()
    assert isinstance(m.pv_tourists, ConditionalDistributionIndex)
    assert isinstance(m.pv_excursionists, ConditionalDistributionIndex)
    assert m.pv_tourists.name == "tourists"
    assert m.pv_excursionists.name == "excursionists"


def test_molveno_model_exposes_context_variables():
    """MolvenoModel exposes the three context variables as attributes."""
    m = MolvenoModel()
    assert isinstance(m.cv_weekday, CategoricalIndex)
    assert isinstance(m.cv_season, CategoricalIndex)
    assert isinstance(m.cv_weather, CategoricalIndex)
    assert m.cv_weekday.name == "weekday"
    assert m.cv_season.name == "season"
    assert m.cv_weather.name == "weather"


def test_cv_probabilities_sum_to_one():
    """Each CategoricalIndex CV has outcomes summing to 1.0."""
    m = MolvenoModel()
    for cv in (m.cv_weekday, m.cv_season, m.cv_weather):
        total = sum(cv.outcomes.values())
        assert abs(total - 1.0) < 1e-9, f"{cv.name}: outcomes sum to {total}"


def test_parking_model_inputs_wired_from_root():
    """ParkingModel presence inputs are the same objects as MolvenoModel attributes."""
    m = MolvenoModel()
    assert m.parking.inputs.pv_tourists is m.pv_tourists
    assert m.parking.inputs.pv_excursionists is m.pv_excursionists
    assert m.parking.inputs.cv_weather is m.cv_weather


def test_beach_model_inputs_wired_from_root():
    """BeachModel presence inputs are the same objects as MolvenoModel attributes."""
    m = MolvenoModel()
    assert m.beach.inputs.pv_tourists is m.pv_tourists
    assert m.beach.inputs.pv_excursionists is m.pv_excursionists
    assert m.beach.inputs.cv_weather is m.cv_weather


def test_accommodation_model_inputs_wired_from_root():
    """AccommodationModel.inputs.pv_tourists is the same object as MolvenoModel attribute."""
    m = MolvenoModel()
    assert m.accommodation.inputs.pv_tourists is m.pv_tourists


def test_food_model_inputs_wired_from_root():
    """FoodModel presence inputs are the same objects as MolvenoModel attributes."""
    m = MolvenoModel()
    assert m.food.inputs.pv_tourists is m.pv_tourists
    assert m.food.inputs.pv_excursionists is m.pv_excursionists
    assert m.food.inputs.cv_weather is m.cv_weather


def test_concern_model_outputs_are_generic_indexes_only():
    """Outputs dataclasses contain only GenericIndex instances (no Constraint)."""
    m = MolvenoModel()
    for submodel in (m.parking, m.beach, m.accommodation, m.food):
        for idx in submodel.outputs:
            assert isinstance(idx, GenericIndex), (
                f"{type(submodel).__name__}.outputs yielded a non-GenericIndex: {type(idx)}"
            )


def test_concern_model_inputs_include_all_i_parameters():
    """All i_* parameters are Inputs to the concern sub-model that uses them."""
    m = MolvenoModel()

    # Parking: 7 i_* params + 3 presence/cv inputs
    assert isinstance(m.parking.inputs.i_u_tourists_parking, Index)
    assert isinstance(m.parking.inputs.i_u_excursionists_parking, Index)
    assert isinstance(m.parking.inputs.i_xa_tourists_per_vehicle, Index)
    assert isinstance(m.parking.inputs.i_xa_excursionists_per_vehicle, Index)
    assert isinstance(m.parking.inputs.i_xo_tourists_parking, Index)
    assert isinstance(m.parking.inputs.i_xo_excursionists_parking, Index)
    assert isinstance(m.parking.inputs.i_c_parking, DistributionIndex)

    # Beach: 5 i_* params + 3 presence/cv inputs
    assert isinstance(m.beach.inputs.i_u_tourists_beach, Index)
    assert isinstance(m.beach.inputs.i_u_excursionists_beach, Index)
    assert isinstance(m.beach.inputs.i_xo_tourists_beach, DistributionIndex)
    assert isinstance(m.beach.inputs.i_xo_excursionists_beach, Index)
    assert isinstance(m.beach.inputs.i_c_beach, DistributionIndex)

    # Accommodation: 3 i_* params + 1 presence input
    assert isinstance(m.accommodation.inputs.i_u_tourists_accommodation, Index)
    assert isinstance(m.accommodation.inputs.i_xa_tourists_accommodation, Index)
    assert isinstance(m.accommodation.inputs.i_c_accommodation, DistributionIndex)

    # Food: 5 i_* params + 3 presence/cv inputs
    assert isinstance(m.food.inputs.i_u_tourists_food, Index)
    assert isinstance(m.food.inputs.i_u_excursionists_food, Index)
    assert isinstance(m.food.inputs.i_xa_visitors_food, Index)
    assert isinstance(m.food.inputs.i_xo_visitors_food, Index)
    assert isinstance(m.food.inputs.i_c_food, DistributionIndex)


def test_concern_models_have_no_expose():
    """Concern sub-models have no Expose — no internal uncertain parameters."""
    m = MolvenoModel()
    for submodel in (m.parking, m.beach, m.accommodation, m.food):
        assert len(submodel.expose) == 0, f"{type(submodel).__name__}.expose is not empty: {submodel.expose}"


def test_concern_model_constraint_is_plain_attribute():
    """Each concern sub-model exposes its Constraint as a plain instance attribute."""
    m = MolvenoModel()
    for submodel in (m.parking, m.beach, m.accommodation, m.food):
        assert isinstance(submodel.constraint, Constraint), f"{type(submodel).__name__}.constraint is not a Constraint"


def test_parking_outputs_index_types():
    """ParkingModel.outputs contains only the usage formula index."""
    m = MolvenoModel()
    assert isinstance(m.parking.outputs.i_u_parking, Index)
    assert len(m.parking.outputs) == 1
    assert m.parking.constraint.name == "parking"


def test_beach_outputs_index_types():
    """BeachModel.outputs contains only the usage formula index."""
    m = MolvenoModel()
    assert isinstance(m.beach.outputs.i_u_beach, Index)
    assert len(m.beach.outputs) == 1
    assert m.beach.constraint.name == "beach"


def test_accommodation_outputs_index_types():
    """AccommodationModel.outputs contains only the usage formula index."""
    m = MolvenoModel()
    assert isinstance(m.accommodation.outputs.i_u_accommodation, Index)
    assert len(m.accommodation.outputs) == 1
    assert m.accommodation.constraint.name == "accommodation"


def test_food_outputs_index_types():
    """FoodModel.outputs contains only the usage formula index."""
    m = MolvenoModel()
    assert isinstance(m.food.outputs.i_u_food, Index)
    assert len(m.food.outputs) == 1
    assert m.food.constraint.name == "food"


# ---------------------------------------------------------------------------
# No contract warnings at construction
# ---------------------------------------------------------------------------


def test_molveno_model_construction_is_warning_free():
    """Constructing MolvenoModel emits no ModelContractWarning or DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ModelContractWarning)
        warnings.simplefilter("error", DeprecationWarning)
        MolvenoModel()


# ---------------------------------------------------------------------------
# Root-model indexes coverage
# ---------------------------------------------------------------------------


def test_presence_cvs_in_root_indexes():
    """All three context variables appear in model.indexes."""
    cv_ids = {id(model.cv_weekday), id(model.cv_season), id(model.cv_weather)}
    root_ids = {id(idx) for idx in model.indexes}
    assert cv_ids <= root_ids


def test_presence_pvs_in_root_indexes():
    """Both presence variables appear in model.indexes."""
    pv_ids = {id(model.pv_tourists), id(model.pv_excursionists)}
    root_ids = {id(idx) for idx in model.indexes}
    assert pv_ids <= root_ids


def test_all_capacity_indexes_in_root_indexes():
    """All four capacity DistributionIndexes appear in model.indexes."""
    cap_ids = {
        id(model.parking.inputs.i_c_parking),
        id(model.beach.inputs.i_c_beach),
        id(model.accommodation.inputs.i_c_accommodation),
        id(model.food.inputs.i_c_food),
    }
    root_ids = {id(idx) for idx in model.indexes}
    assert cap_ids <= root_ids


def test_beach_rotation_factor_in_root_indexes():
    """i_xo_tourists_beach (DistributionIndex in BeachModel.Inputs) is in model.indexes."""
    rotation_id = id(model.beach.inputs.i_xo_tourists_beach)
    root_ids = {id(idx) for idx in model.indexes}
    assert rotation_id in root_ids


def test_usage_formula_indexes_in_root_indexes():
    """All four usage formula indexes appear in model.indexes."""
    usage_ids = {
        id(model.parking.outputs.i_u_parking),
        id(model.beach.outputs.i_u_beach),
        id(model.accommodation.outputs.i_u_accommodation),
        id(model.food.outputs.i_u_food),
    }
    root_ids = {id(idx) for idx in model.indexes}
    assert usage_ids <= root_ids


def test_root_indexes_has_no_duplicates():
    """model.indexes contains no duplicate objects (identity check)."""
    ids = [id(idx) for idx in model.indexes]
    assert len(ids) == len(set(ids))


def test_beach_rotation_factor_is_abstract():
    """i_xo_tourists_beach is distribution-backed and therefore abstract."""
    assert any(idx is model.beach.inputs.i_xo_tourists_beach for idx in model.abstract_indexes())


# ---------------------------------------------------------------------------
# MolvenoModel domain attributes (cvs, pvs, constraints)
# ---------------------------------------------------------------------------


def test_molveno_model_cvs_list():
    """model.cvs contains exactly the three context variables."""
    assert len(model.cvs) == 3
    cv_ids = {id(cv) for cv in model.cvs}
    assert id(model.cv_weekday) in cv_ids
    assert id(model.cv_season) in cv_ids
    assert id(model.cv_weather) in cv_ids


def test_molveno_model_pvs_list():
    """model.pvs contains exactly the two presence variables."""
    assert len(model.pvs) == 2
    pv_ids = {id(pv) for pv in model.pvs}
    assert id(model.pv_tourists) in pv_ids
    assert id(model.pv_excursionists) in pv_ids


def test_molveno_model_constraints_list():
    """model.constraints contains exactly four Constraint objects, one per concern."""
    assert len(model.constraints) == 4
    names = {c.name for c in model.constraints}
    assert names == {"parking", "beach", "accommodation", "food"}


def test_molveno_model_constraints_match_sub_model_attributes():
    """model.constraints entries are the same objects as sub-model .constraint attributes."""
    sub_constraints = {
        model.parking.constraint,
        model.beach.constraint,
        model.accommodation.constraint,
        model.food.constraint,
    }
    root_constraints = set(model.constraints)
    # Identity check via id()
    assert {id(c) for c in sub_constraints} == {id(c) for c in root_constraints}


def test_presence_transformation_indexes_in_root_indexes():
    """The four presence-transformation indexes appear in model.indexes."""
    pt_ids = {
        id(model.i_p_tourists_reduction_factor),
        id(model.i_p_excursionists_reduction_factor),
        id(model.i_p_tourists_saturation_level),
        id(model.i_p_excursionists_saturation_level),
    }
    root_ids = {id(idx) for idx in model.indexes}
    assert pt_ids <= root_ids


# ---------------------------------------------------------------------------


def test_bug_37():
    """Regression for https://github.com/fbk-most/dt-model/issues/37."""
    situation: dict[CategoricalIndex, list[str]] = {model.cv_weather: ["good", "unsettled", "bad"]}
    ensemble = CrossProductEnsemble(model, restrictions=situation, max_categorical_size=20, exclude=model.pvs)

    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])

    field, field_elements, _ = compute_field(model, ensemble, tourists, excursionists)

    assert field is not None
    assert field_elements is not None
    assert field.shape == (tourists.size, excursionists.size)
