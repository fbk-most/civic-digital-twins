# SPDX-License-Identifier: Apache-2.0
"""Integration tests for BolognaOutput and BolognaEvaluator."""

import numpy as np
import pytest
from mobility_bologna.mobility_bologna import BolognaEvaluator, BolognaModel, BolognaOutput

from civic_digital_twins.dt_model.simulation.handle import EvaluationHandle
from civic_digital_twins.dt_model.simulation.runner import EvaluationConfig, ModelRunHandle
from civic_digital_twins.dt_model.simulation.scenario import Scenario

# Use a tiny ensemble to keep tests fast.
_ENSEMBLE_SIZE = 1

_EXPOSE_NAMES = (
    "ts_inflow",
    "modified_inflow",
    "traffic",
    "modified_traffic",
    "emissions",
    "modified_emissions",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> BolognaModel:
    """Shared BolognaModel — graph construction is the expensive part."""
    return BolognaModel(**BolognaModel.default_inputs())


@pytest.fixture(scope="module")
def evaluator(model: BolognaModel) -> BolognaEvaluator:
    """Shared evaluator built on the shared model."""
    return BolognaEvaluator(model)


@pytest.fixture(scope="module")
def scenario(model: BolognaModel) -> Scenario:
    """Baseline scenario with no overrides."""
    return Scenario(model)


@pytest.fixture(scope="module")
def config() -> EvaluationConfig:
    """Tiny evaluation config for fast tests."""
    return EvaluationConfig(ensemble_size=_ENSEMBLE_SIZE)


@pytest.fixture(scope="module")
def output(evaluator: BolognaEvaluator, scenario: Scenario, config: EvaluationConfig) -> BolognaOutput:
    """Single blocking evaluation result shared across tests."""
    return evaluator.evaluate(scenario, config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_evaluate_returns_bologna_output(output: BolognaOutput) -> None:
    """evaluate() must return a BolognaOutput instance."""
    assert isinstance(output, BolognaOutput)


def test_kpis_has_expected_keys(output: BolognaOutput) -> None:
    """Kpis dict must contain at least the 'Base inflow [veh/day]' key."""
    assert "Base inflow [veh/day]" in output.kpis


def test_kpis_values_are_int(output: BolognaOutput) -> None:
    """All KPI values must be integers (as returned by compute_kpis)."""
    for key, value in output.kpis.items():
        assert isinstance(value, int), f"KPI '{key}' has non-int value {value!r}"


def test_timeseries_has_expected_keys(output: BolognaOutput) -> None:
    """Timeseries dict must contain all six expose index names."""
    for name in _EXPOSE_NAMES:
        assert name in output.timeseries, f"timeseries missing key '{name}'"


def test_timeseries_values_are_arrays(output: BolognaOutput) -> None:
    """All timeseries values must be 1-D numpy arrays."""
    for name in _EXPOSE_NAMES:
        arr = output.timeseries[name]
        assert isinstance(arr, np.ndarray), f"timeseries['{name}'] is not an ndarray"


def test_to_dict_contains_required_keys(output: BolognaOutput) -> None:
    """to_dict() must include 'dt_model_version', 'kpis', 'timeseries', 'fields', and '_resume'."""
    data = output.to_dict()
    assert "dt_model_version" in data
    assert "kpis" in data
    assert "timeseries" in data
    assert "fields" in data
    assert "_resume" in data


def test_from_dict_round_trip_kpis(output: BolognaOutput) -> None:
    """from_dict(output.to_dict()) must preserve kpis exactly."""
    restored = BolognaOutput.from_dict(output.to_dict())
    assert restored.kpis == output.kpis


def test_from_dict_is_resumable(output: BolognaOutput) -> None:
    """from_dict() on a dict with '_resume' must produce is_resumable=True."""
    restored = BolognaOutput.from_dict(output.to_dict())
    assert restored.is_resumable is True


def test_from_dict_timeseries_round_trip(output: BolognaOutput) -> None:
    """from_dict() must round-trip all timeseries arrays exactly."""
    restored = BolognaOutput.from_dict(output.to_dict())
    for name in _EXPOSE_NAMES:
        np.testing.assert_array_equal(
            restored.timeseries[name],
            output.timeseries[name],
            err_msg=f"timeseries['{name}'] did not round-trip correctly",
        )


_FIELD_NAMES = ("modified_inflow", "modified_traffic", "modified_emissions")


def test_fields_has_expected_keys(output: BolognaOutput) -> None:
    """Fields dict must contain the three modified-quantity array keys."""
    for name in _FIELD_NAMES:
        assert name in output.fields, f"fields missing key '{name}'"


def test_fields_values_are_2d_arrays(output: BolognaOutput) -> None:
    """Each entry in fields must be a 2-D numpy array of shape (S, T)."""
    for name in _FIELD_NAMES:
        arr = output.fields[name]
        assert isinstance(arr, np.ndarray), f"fields['{name}'] is not an ndarray"
        assert arr.ndim == 2, f"fields['{name}'] has ndim={arr.ndim}, expected 2"


def test_to_dict_and_from_dict_round_trip_fields(output: BolognaOutput) -> None:
    """to_dict() / from_dict() must round-trip all field arrays exactly."""
    restored = BolognaOutput.from_dict(output.to_dict())
    for name in _FIELD_NAMES:
        np.testing.assert_array_equal(
            restored.fields[name],
            output.fields[name],
            err_msg=f"fields['{name}'] did not round-trip correctly",
        )


def test_from_dict_without_resume_not_resumable() -> None:
    """from_dict() on a dict without '_resume' must produce is_resumable=False."""
    data = {
        "dt_model_version": "0.0.0",
        "kpis": {"Base inflow [veh/day]": 42},
        "timeseries": {},
    }
    restored = BolognaOutput.from_dict(data)
    assert restored.is_resumable is False


def test_resume_returns_evaluation_handle(
    evaluator: BolognaEvaluator,
    scenario: Scenario,
    output: BolognaOutput,
    config: EvaluationConfig,
) -> None:
    """resume() must return an EvaluationHandle when output is resumable."""
    handle = evaluator.resume(scenario, output, config)
    assert isinstance(handle, EvaluationHandle)


def test_resume_after_round_trip(
    evaluator: BolognaEvaluator,
    scenario: Scenario,
) -> None:
    """Full save-and-restore cycle: evaluate → to_dict → from_dict → resume → extend."""
    config = EvaluationConfig(ensemble_size=2)
    fresh = evaluator.evaluate(scenario, config)
    restored = BolognaOutput.from_dict(fresh.to_dict())
    assert restored.is_resumable
    handle = evaluator.resume(scenario, restored, config)
    assert isinstance(handle, EvaluationHandle)
    extended = handle.extend(ensemble_size=2)
    assert extended is handle.result


def test_run_async_returns_model_run_handle(
    evaluator: BolognaEvaluator,
    scenario: Scenario,
    config: EvaluationConfig,
) -> None:
    """run_async() must return a ModelRunHandle."""
    handle = evaluator.run_async(scenario, config)
    assert isinstance(handle, ModelRunHandle)


def test_run_async_get_returns_bologna_output(
    evaluator: BolognaEvaluator,
    scenario: Scenario,
    config: EvaluationConfig,
) -> None:
    """ModelRunHandle.get() must return a BolognaOutput."""
    handle = evaluator.run_async(scenario, config)
    result = handle.get()
    assert isinstance(result, BolognaOutput)


def test_run_async_get_has_kpis(
    evaluator: BolognaEvaluator,
    scenario: Scenario,
    config: EvaluationConfig,
) -> None:
    """BolognaOutput from run_async().get() must have expected KPI keys."""
    handle = evaluator.run_async(scenario, config)
    result = handle.get()
    assert "Base inflow [veh/day]" in result.kpis


def test_structure_returns_non_empty_dict(evaluator: BolognaEvaluator) -> None:
    """structure() must return a non-empty dict."""
    schema = evaluator.structure()
    assert isinstance(schema, dict)
    assert len(schema) >= 1


def test_structure_type_values(evaluator: BolognaEvaluator) -> None:
    """Every entry in structure() must have a 'type' key with 'scalar' or 'distribution'."""
    schema = evaluator.structure()
    valid_types = {"scalar", "distribution"}
    for name, meta in schema.items():
        assert "type" in meta, f"structure()['{name}'] missing 'type' key"
        assert meta["type"] in valid_types, f"structure()['{name}']['type'] = {meta['type']!r} is invalid"


def test_structure_contains_distribution_index(evaluator: BolognaEvaluator) -> None:
    """structure() must contain at least one 'distribution' entry for i_b_p50_cost."""
    schema = evaluator.structure()
    distribution_entries = {name: meta for name, meta in schema.items() if meta["type"] == "distribution"}
    assert len(distribution_entries) >= 1, "Expected at least one distribution entry in structure()"
