# SPDX-License-Identifier: Apache-2.0
"""Integration tests for MolvenoOutput and MolvenoEvaluator.

These tests exercise the full evaluation lifecycle:
  evaluate → to_dict / from_dict → resume → run_async
"""

from __future__ import annotations

import pytest
from overtourism_molveno.molveno_model import MolvenoEvaluator, MolvenoModel, MolvenoOutput

from civic_digital_twins.dt_model import Scenario
from civic_digital_twins.dt_model.simulation.handle import EvaluationHandle
from civic_digital_twins.dt_model.simulation.runner import EvaluationConfig, ModelRunHandle

# ---------------------------------------------------------------------------
# Small grid so tests stay fast.
# ---------------------------------------------------------------------------
_T_MAX = 100
_E_MAX = 100
_T_SAMPLE = 5
_E_SAMPLE = 5
_ENSEMBLE_SIZE = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> MolvenoModel:
    """Shared MolvenoModel instance (graph construction is the expensive part)."""
    return MolvenoModel()


@pytest.fixture(scope="module")
def evaluator(model: MolvenoModel) -> MolvenoEvaluator:
    """Return an evaluator configured with a small grid."""
    return MolvenoEvaluator(model, t_max=_T_MAX, e_max=_E_MAX, t_sample=_T_SAMPLE, e_sample=_E_SAMPLE)


@pytest.fixture(scope="module")
def scenario(model: MolvenoModel) -> Scenario:
    """Return a base scenario with no overrides."""
    return Scenario(model)


@pytest.fixture(scope="module")
def config() -> EvaluationConfig:
    """Return an evaluation config with a small ensemble size."""
    return EvaluationConfig(ensemble_size=_ENSEMBLE_SIZE)


@pytest.fixture(scope="module")
def output(evaluator: MolvenoEvaluator, scenario: Scenario, config: EvaluationConfig) -> MolvenoOutput:
    """One fully evaluated MolvenoOutput (reused across tests in this module)."""
    return evaluator.evaluate(scenario, config)


# ---------------------------------------------------------------------------
# Tests: evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_returns_molveno_output(output: MolvenoOutput) -> None:
    """evaluate() must return a MolvenoOutput instance."""
    assert isinstance(output, MolvenoOutput)


def test_evaluate_field_shape(output: MolvenoOutput) -> None:
    """Assert field has shape (t_sample+1, e_sample+1)."""
    expected_shape = (_T_SAMPLE + 1, _E_SAMPLE + 1)
    assert output.field.shape == expected_shape, f"Expected shape {expected_shape}, got {output.field.shape}"


def test_evaluate_tt_ee_shapes(output: MolvenoOutput) -> None:
    """Assert tt and ee are 1-D arrays with t_sample+1 and e_sample+1 elements."""
    assert output.tt.shape == (_T_SAMPLE + 1,)
    assert output.ee.shape == (_E_SAMPLE + 1,)


def test_evaluate_field_values_in_range(output: MolvenoOutput) -> None:
    """All field values must be in [0, 1]."""
    assert float(output.field.min()) >= 0.0, "field has values below 0"
    assert float(output.field.max()) <= 1.0, "field has values above 1"


def test_evaluate_field_elements_keys(output: MolvenoOutput, model: MolvenoModel) -> None:
    """field_elements must have one entry per model constraint."""
    assert len(output.field_elements) == len(model.constraints)


def test_evaluate_field_elements_shapes(output: MolvenoOutput) -> None:
    """Every field element must have the same shape as field."""
    for key, elem in output.field_elements.items():
        assert elem.shape == output.field.shape, f"field_elements[{key!r}] shape mismatch: {elem.shape}"


def test_evaluate_output_is_resumable(output: MolvenoOutput) -> None:
    """An output produced by evaluate() must be immediately resumable."""
    assert output.is_resumable


# ---------------------------------------------------------------------------
# Tests: to_dict() / from_dict()
# ---------------------------------------------------------------------------


def test_to_dict_contains_required_keys(output: MolvenoOutput) -> None:
    """to_dict() must contain the expected top-level keys."""
    d = output.to_dict()
    for key in (
        "dt_model_version",
        "field",
        "field_elements",
        "tt",
        "ee",
        "sample_tourists",
        "sample_excursionists",
        "_resume",
    ):
        assert key in d, f"Missing key: {key!r}"


def test_roundtrip_is_resumable(output: MolvenoOutput) -> None:
    """from_dict(output.to_dict()) must set is_resumable=True."""
    loaded = MolvenoOutput.from_dict(output.to_dict())
    assert loaded.is_resumable


def test_roundtrip_field_shape(output: MolvenoOutput) -> None:
    """Round-tripped field must have the same shape as the original."""
    loaded = MolvenoOutput.from_dict(output.to_dict())
    assert loaded.field.shape == output.field.shape


def test_roundtrip_field_values(output: MolvenoOutput) -> None:
    """Round-tripped field must be numerically identical to the original."""
    import numpy as np

    loaded = MolvenoOutput.from_dict(output.to_dict())
    assert np.array_equal(loaded.field, output.field)


def test_roundtrip_tt_ee(output: MolvenoOutput) -> None:
    """Round-tripped tt and ee must be numerically identical to the originals."""
    import numpy as np

    loaded = MolvenoOutput.from_dict(output.to_dict())
    assert np.array_equal(loaded.tt, output.tt)
    assert np.array_equal(loaded.ee, output.ee)


def test_roundtrip_presence_samples(output: MolvenoOutput) -> None:
    """Round-tripped sample_tourists and sample_excursionists must equal the originals."""
    loaded = MolvenoOutput.from_dict(output.to_dict())
    assert loaded.sample_tourists == output.sample_tourists
    assert loaded.sample_excursionists == output.sample_excursionists


def test_evaluate_presence_samples_length(output: MolvenoOutput) -> None:
    """sample_tourists and sample_excursionists must be non-empty lists of floats."""
    assert isinstance(output.sample_tourists, list)
    assert isinstance(output.sample_excursionists, list)
    assert len(output.sample_tourists) > 0
    assert len(output.sample_excursionists) > 0


def test_roundtrip_field_elements_count(output: MolvenoOutput) -> None:
    """Round-tripped field_elements must have the same number of entries."""
    loaded = MolvenoOutput.from_dict(output.to_dict())
    assert len(loaded.field_elements) == len(output.field_elements)


def test_roundtrip_without_resume(output: MolvenoOutput) -> None:
    """from_dict on a dict without '_resume' must produce is_resumable=False."""
    d = output.to_dict()
    d.pop("_resume", None)
    loaded = MolvenoOutput.from_dict(d)
    assert not loaded.is_resumable


# ---------------------------------------------------------------------------
# Tests: resume()
# ---------------------------------------------------------------------------


def test_resume_returns_evaluation_handle(
    evaluator: MolvenoEvaluator,
    scenario: Scenario,
    output: MolvenoOutput,
    config: EvaluationConfig,
) -> None:
    """resume() on a resumable output must return an EvaluationHandle."""
    loaded = MolvenoOutput.from_dict(output.to_dict())
    handle = evaluator.resume(scenario, loaded, config)
    assert isinstance(handle, EvaluationHandle)


# ---------------------------------------------------------------------------
# Tests: run_async()
# ---------------------------------------------------------------------------


def test_run_async_returns_model_run_handle(
    evaluator: MolvenoEvaluator,
    scenario: Scenario,
    config: EvaluationConfig,
) -> None:
    """run_async() must return a ModelRunHandle."""
    handle = evaluator.run_async(scenario, config)
    assert isinstance(handle, ModelRunHandle)


def test_run_async_get_returns_molveno_output(
    evaluator: MolvenoEvaluator,
    scenario: Scenario,
    config: EvaluationConfig,
) -> None:
    """ModelRunHandle.get() must return a MolvenoOutput."""
    handle = evaluator.run_async(scenario, config)
    result = handle.get()
    assert isinstance(result, MolvenoOutput)


def test_run_async_output_field_shape(
    evaluator: MolvenoEvaluator,
    scenario: Scenario,
    config: EvaluationConfig,
) -> None:
    """ModelRunHandle.get().field must have the expected shape."""
    handle = evaluator.run_async(scenario, config)
    result = handle.get()
    assert result.field.shape == (_T_SAMPLE + 1, _E_SAMPLE + 1)


# ---------------------------------------------------------------------------
# Tests: structure()
# ---------------------------------------------------------------------------


def test_structure_returns_non_empty_dict(evaluator: MolvenoEvaluator) -> None:
    """structure() must return a non-empty dict."""
    schema = evaluator.structure()
    assert isinstance(schema, dict)
    assert len(schema) > 0


def test_structure_contains_categorical_cvs(evaluator: MolvenoEvaluator, model: MolvenoModel) -> None:
    """structure() must include all three categorical context variables."""
    schema = evaluator.structure()
    for cv in model.cvs:
        assert cv.name in schema, f"Missing CV {cv.name!r} in structure()"
        assert schema[cv.name]["type"] == "categorical"
        assert "support" in schema[cv.name]


def test_structure_contains_capacity_parameters(evaluator: MolvenoEvaluator, model: MolvenoModel) -> None:
    """structure() must include all capacity parameters."""
    schema = evaluator.structure()
    for cap in model.capacities:
        assert cap.name in schema, f"Missing capacity {cap.name!r} in structure()"
        assert schema[cap.name]["type"] == "distribution"


# ---------------------------------------------------------------------------
# Tests: scenario with overrides
# ---------------------------------------------------------------------------


def test_evaluate_with_str_override(
    evaluator: MolvenoEvaluator,
    model: MolvenoModel,
    config: EvaluationConfig,
) -> None:
    """evaluate() must work with a str override on a CategoricalIndex."""
    scenario = Scenario(model, overrides={model.cv_weather: "good"})
    out = evaluator.evaluate(scenario, config)
    assert isinstance(out, MolvenoOutput)
    assert out.field.shape == (_T_SAMPLE + 1, _E_SAMPLE + 1)


def test_evaluate_with_dict_override(
    evaluator: MolvenoEvaluator,
    model: MolvenoModel,
    config: EvaluationConfig,
) -> None:
    """evaluate() must work with a dict override on a CategoricalIndex."""
    scenario = Scenario(model, overrides={model.cv_weather: {"good": 0.7, "unsettled": 0.3}})
    out = evaluator.evaluate(scenario, config)
    assert isinstance(out, MolvenoOutput)
    assert out.field.shape == (_T_SAMPLE + 1, _E_SAMPLE + 1)


# ---------------------------------------------------------------------------
# Tests: MolvenoOutput lazy cached properties
# ---------------------------------------------------------------------------


def test_sustainable_area_is_positive_float(output: MolvenoOutput) -> None:
    """sustainable_area must be a positive float."""
    assert isinstance(output.sustainable_area, float)
    assert output.sustainable_area > 0.0


def test_sustainability_index_is_float_pair(output: MolvenoOutput) -> None:
    """sustainability_index must be a (float, float) tuple."""
    si = output.sustainability_index
    assert isinstance(si, tuple)
    assert len(si) == 2
    assert all(isinstance(v, float) for v in si)


def test_sustainability_by_constraint_keys(output: MolvenoOutput, model: MolvenoModel) -> None:
    """sustainability_by_constraint must have one entry per model constraint."""
    sbc = output.sustainability_by_constraint
    assert isinstance(sbc, dict)
    assert len(sbc) == len(model.constraints)
    constraint_names = {c.name for c in model.constraints}
    assert set(sbc.keys()) == constraint_names


def test_sustainability_by_constraint_values_are_float_pairs(output: MolvenoOutput) -> None:
    """Every value in sustainability_by_constraint must be a (float, float) tuple."""
    for name, val in output.sustainability_by_constraint.items():
        assert isinstance(val, tuple), f"Expected tuple for {name!r}, got {type(val)}"
        assert len(val) == 2, f"Expected 2-tuple for {name!r}, got length {len(val)}"
        assert all(isinstance(v, float) for v in val), f"Expected floats for {name!r}, got {val}"


def test_modal_lines_is_dict(output: MolvenoOutput) -> None:
    """modal_lines must be a dict (may be empty for small grids)."""
    assert isinstance(output.modal_lines, dict)


def test_modal_lines_values_are_coordinate_pairs(output: MolvenoOutput) -> None:
    """Each modal line must be a ((t0, t1), (e0, e1)) pair."""
    for name, line in output.modal_lines.items():
        assert isinstance(line, tuple), f"Expected tuple for {name!r}"
        assert len(line) == 2, f"Expected 2-tuple for {name!r}"
        (ts, es) = line
        assert len(ts) == 2, f"Tourist endpoints for {name!r} should be a 2-tuple"
        assert len(es) == 2, f"Excursionist endpoints for {name!r} should be a 2-tuple"


def test_lazy_property_caching(output: MolvenoOutput) -> None:
    """Calling a lazy property twice must return the same object (caching works)."""
    assert output.sustainable_area is output.sustainable_area
    assert output.sustainability_index is output.sustainability_index
    assert output.sustainability_by_constraint is output.sustainability_by_constraint
    assert output.modal_lines is output.modal_lines


def test_lazy_properties_work_after_roundtrip(output: MolvenoOutput, model: MolvenoModel) -> None:
    """Lazy properties must work correctly on a round-tripped (deserialized) output."""
    loaded = MolvenoOutput.from_dict(output.to_dict())
    assert isinstance(loaded.sustainable_area, float)
    assert loaded.sustainable_area > 0.0
    si = loaded.sustainability_index
    assert len(si) == 2 and all(isinstance(v, float) for v in si)
    sbc = loaded.sustainability_by_constraint
    assert len(sbc) == len(model.constraints)
    assert isinstance(loaded.modal_lines, dict)
