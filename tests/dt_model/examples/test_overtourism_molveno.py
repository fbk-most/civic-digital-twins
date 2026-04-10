"""Tests for the Molveno overtourism example using the new Evaluation.evaluate(axes=...) API."""

# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import pytest
from overtourism_molveno.molveno_model import (
    AccommodationModel,
    BeachModel,
    CV_season,
    CV_weather,
    CV_weekday,
    FoodModel,
    I_P_excursionists_reduction_factor,
    I_P_excursionists_saturation_level,
    I_P_tourists_reduction_factor,
    I_P_tourists_saturation_level,
    M_Base,
    MolvenoModel,
    ParkingModel,
    PresenceModel,
    PV_excursionists,
    PV_tourists,
)
from overtourism_molveno.overtourism_metamodel import (
    CategoricalContextVariable,
    Constraint,
    ContextVariable,
    OvertourismEnsemble,
    PresenceVariable,
    UniformCategoricalContextVariable,
)

from civic_digital_twins.dt_model import Evaluation
from civic_digital_twins.dt_model.model.index import Distribution, DistributionIndex, GenericIndex, Index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_field(model, scenarios, tt, ee):
    """Evaluate the sustainability field using Evaluation.evaluate(axes=...).

    Returns ``(field, field_elements, result)`` where:
    - ``field`` has shape ``(tt.size, ee.size)``
    - ``field_elements`` maps each Constraint to a ``(tt.size, ee.size)`` array
    - ``result`` is the :class:`~dt_model.simulation.evaluation.EvaluationResult`
    """
    result = Evaluation(model).evaluate(scenarios, parameters={PV_tourists: tt, PV_excursionists: ee})

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
    """Single-member scenario list: good weather, monday, high season."""
    np.random.seed(0)
    random.seed(0)
    return list(
        OvertourismEnsemble(
            M_Base,
            {
                CV_weekday: ["monday"],
                CV_season: ["high"],
                CV_weather: ["good"],
            },
        )
    )


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
    scenario: dict[ContextVariable, list] = {CV_weather: ["good", "bad"]}
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
    ensemble = OvertourismEnsemble(
        M_Base,
        {
            CV_weekday: ["monday"],
            CV_season: ["high"],
            CV_weather: ["good"],
        },
    )
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
    scenario: dict[ContextVariable, list] = {CV_weather: ["good", "bad"]}
    ens = OvertourismEnsemble(M_Base, scenario, cv_ensemble_size=10)
    scenarios = list(ens)

    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])

    field, field_elements, _ = compute_field(M_Base, scenarios, tourists, excursionists)

    assert field is not None
    assert field_elements is not None
    assert field.shape == (tourists.size, excursionists.size)


# ---------------------------------------------------------------------------
# Sub-model hierarchy — structure and typing
# ---------------------------------------------------------------------------


def test_molveno_model_has_five_sub_models():
    """MolvenoModel exposes all five concern sub-models as named attributes."""
    m = MolvenoModel()
    assert isinstance(m.presence, PresenceModel)
    assert isinstance(m.parking, ParkingModel)
    assert isinstance(m.beach, BeachModel)
    assert isinstance(m.accommodation, AccommodationModel)
    assert isinstance(m.food, FoodModel)


def test_presence_model_outputs_pvs():
    """PresenceModel.outputs exposes pv_tourists and pv_excursionists."""
    m = MolvenoModel()
    assert isinstance(m.presence.outputs.pv_tourists, PresenceVariable)
    assert isinstance(m.presence.outputs.pv_excursionists, PresenceVariable)
    assert m.presence.outputs.pv_tourists.name == "tourists"
    assert m.presence.outputs.pv_excursionists.name == "excursionists"


def test_presence_model_outputs_context_variables():
    """PresenceModel.outputs holds all three context variables."""
    m = MolvenoModel()
    assert isinstance(m.presence.outputs.cv_weekday, UniformCategoricalContextVariable)
    assert isinstance(m.presence.outputs.cv_season, CategoricalContextVariable)
    assert isinstance(m.presence.outputs.cv_weather, CategoricalContextVariable)
    assert m.presence.outputs.cv_weekday.name == "weekday"
    assert m.presence.outputs.cv_season.name == "season"
    assert m.presence.outputs.cv_weather.name == "weather"


def test_presence_model_has_no_expose():
    """PresenceModel has no Expose — all outputs are contractual."""
    m = MolvenoModel()
    assert len(m.presence.expose) == 0


def test_parking_model_inputs_wired_from_presence():
    """ParkingModel presence inputs are the same objects as PresenceModel outputs."""
    m = MolvenoModel()
    assert m.parking.inputs.pv_tourists is m.presence.outputs.pv_tourists
    assert m.parking.inputs.pv_excursionists is m.presence.outputs.pv_excursionists
    assert m.parking.inputs.cv_weather is m.presence.outputs.cv_weather


def test_beach_model_inputs_wired_from_presence():
    """BeachModel presence inputs are the same objects as PresenceModel outputs."""
    m = MolvenoModel()
    assert m.beach.inputs.pv_tourists is m.presence.outputs.pv_tourists
    assert m.beach.inputs.pv_excursionists is m.presence.outputs.pv_excursionists
    assert m.beach.inputs.cv_weather is m.presence.outputs.cv_weather


def test_accommodation_model_inputs_wired_from_presence():
    """AccommodationModel.inputs.pv_tourists is the same object as PresenceModel outputs."""
    m = MolvenoModel()
    assert m.accommodation.inputs.pv_tourists is m.presence.outputs.pv_tourists


def test_food_model_inputs_wired_from_presence():
    """FoodModel presence inputs are the same objects as PresenceModel outputs."""
    m = MolvenoModel()
    assert m.food.inputs.pv_tourists is m.presence.outputs.pv_tourists
    assert m.food.inputs.pv_excursionists is m.presence.outputs.pv_excursionists
    assert m.food.inputs.cv_weather is m.presence.outputs.cv_weather


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
# Sub-model hierarchy — indexes coverage
# ---------------------------------------------------------------------------


def test_presence_cvs_in_root_indexes():
    """All three context variables from PresenceModel appear in M_Base.indexes."""
    cv_ids = {
        id(M_Base.presence.outputs.cv_weekday),
        id(M_Base.presence.outputs.cv_season),
        id(M_Base.presence.outputs.cv_weather),
    }
    root_ids = {id(idx) for idx in M_Base.indexes}
    assert cv_ids <= root_ids


def test_presence_pvs_in_root_indexes():
    """Both presence variables from PresenceModel appear in M_Base.indexes."""
    pv_ids = {
        id(M_Base.presence.outputs.pv_tourists),
        id(M_Base.presence.outputs.pv_excursionists),
    }
    root_ids = {id(idx) for idx in M_Base.indexes}
    assert pv_ids <= root_ids


def test_all_capacity_indexes_in_root_indexes():
    """All four capacity DistributionIndexes appear in M_Base.indexes."""
    cap_ids = {
        id(M_Base.parking.inputs.i_c_parking),
        id(M_Base.beach.inputs.i_c_beach),
        id(M_Base.accommodation.inputs.i_c_accommodation),
        id(M_Base.food.inputs.i_c_food),
    }
    root_ids = {id(idx) for idx in M_Base.indexes}
    assert cap_ids <= root_ids


def test_beach_rotation_factor_in_root_indexes():
    """i_xo_tourists_beach (DistributionIndex in BeachModel.Inputs) is in M_Base.indexes."""
    rotation_id = id(M_Base.beach.inputs.i_xo_tourists_beach)
    root_ids = {id(idx) for idx in M_Base.indexes}
    assert rotation_id in root_ids


def test_usage_formula_indexes_in_root_indexes():
    """All four usage formula indexes appear in M_Base.indexes."""
    usage_ids = {
        id(M_Base.parking.outputs.i_u_parking),
        id(M_Base.beach.outputs.i_u_beach),
        id(M_Base.accommodation.outputs.i_u_accommodation),
        id(M_Base.food.outputs.i_u_food),
    }
    root_ids = {id(idx) for idx in M_Base.indexes}
    assert usage_ids <= root_ids


def test_root_indexes_has_no_duplicates():
    """M_Base.indexes contains no duplicate objects (identity check)."""
    ids = [id(idx) for idx in M_Base.indexes]
    assert len(ids) == len(set(ids))


def test_beach_rotation_factor_is_abstract():
    """i_xo_tourists_beach is distribution-backed and therefore abstract."""
    assert M_Base.beach.inputs.i_xo_tourists_beach in M_Base.abstract_indexes()


# ---------------------------------------------------------------------------
# OvertourismModel domain attributes preserved on MolvenoModel
# ---------------------------------------------------------------------------


def test_molveno_model_cvs_list():
    """M_Base.cvs contains exactly the three context variables."""
    assert len(M_Base.cvs) == 3
    cv_ids = {id(cv) for cv in M_Base.cvs}
    assert id(M_Base.presence.outputs.cv_weekday) in cv_ids
    assert id(M_Base.presence.outputs.cv_season) in cv_ids
    assert id(M_Base.presence.outputs.cv_weather) in cv_ids


def test_molveno_model_pvs_list():
    """M_Base.pvs contains exactly the two presence variables."""
    assert len(M_Base.pvs) == 2
    pv_ids = {id(pv) for pv in M_Base.pvs}
    assert id(M_Base.presence.outputs.pv_tourists) in pv_ids
    assert id(M_Base.presence.outputs.pv_excursionists) in pv_ids


def test_molveno_model_constraints_list():
    """M_Base.constraints contains exactly four Constraint objects, one per concern."""
    assert len(M_Base.constraints) == 4
    names = {c.name for c in M_Base.constraints}
    assert names == {"parking", "beach", "accommodation", "food"}


def test_molveno_model_constraints_match_sub_model_attributes():
    """M_Base.constraints entries are the same objects as sub-model .constraint attributes."""
    sub_constraints = {
        M_Base.parking.constraint,
        M_Base.beach.constraint,
        M_Base.accommodation.constraint,
        M_Base.food.constraint,
    }
    root_constraints = set(M_Base.constraints)
    # Identity check via id()
    assert {id(c) for c in sub_constraints} == {id(c) for c in root_constraints}


# ---------------------------------------------------------------------------
# Backward-compat module-level aliases
# ---------------------------------------------------------------------------


def test_module_aliases_cv_identity():
    """Module-level CV_* aliases are identical to PresenceModel.outputs attributes."""
    assert CV_weekday is M_Base.presence.outputs.cv_weekday
    assert CV_season is M_Base.presence.outputs.cv_season
    assert CV_weather is M_Base.presence.outputs.cv_weather


def test_module_aliases_pv_identity():
    """Module-level PV_* aliases are identical to the presence sub-model outputs."""
    assert PV_tourists is M_Base.presence.outputs.pv_tourists
    assert PV_excursionists is M_Base.presence.outputs.pv_excursionists


def test_module_aliases_presence_transformation_identity():
    """Module-level I_P_* aliases are identical to the root model attributes."""
    assert I_P_tourists_reduction_factor is M_Base.I_P_tourists_reduction_factor
    assert I_P_excursionists_reduction_factor is M_Base.I_P_excursionists_reduction_factor
    assert I_P_tourists_saturation_level is M_Base.I_P_tourists_saturation_level
    assert I_P_excursionists_saturation_level is M_Base.I_P_excursionists_saturation_level


def test_presence_transformation_indexes_in_root_indexes():
    """The four presence-transformation indexes appear in M_Base.indexes."""
    pt_ids = {
        id(I_P_tourists_reduction_factor),
        id(I_P_excursionists_reduction_factor),
        id(I_P_tourists_saturation_level),
        id(I_P_excursionists_saturation_level),
    }
    root_ids = {id(idx) for idx in M_Base.indexes}
    assert pt_ids <= root_ids


# ---------------------------------------------------------------------------


def test_bug_37():
    """Regression for https://github.com/fbk-most/dt-model/issues/37."""
    np.random.seed(0)
    random.seed(0)
    situation: dict[ContextVariable, list] = {CV_weather: ["good", "unsettled", "bad"]}
    ensemble = OvertourismEnsemble(M_Base, situation, cv_ensemble_size=20)
    scenarios = list(ensemble)

    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])

    field, field_elements, _ = compute_field(M_Base, scenarios, tourists, excursionists)

    assert field is not None
    assert field_elements is not None
    assert field.shape == (tourists.size, excursionists.size)
