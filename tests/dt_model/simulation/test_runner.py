# SPDX-License-Identifier: Apache-2.0
"""Tests for Step 1 of scenario-runner-protocol.md.

Covers:
- ``EvaluationConfig`` dataclass
- ``ResumeState`` dataclass
- ``IncompatibleResultError`` exception
- ``ModelOutput`` ABC: ``is_resumable`` flag mechanics and round-trip via a
  concrete stub subclass
- ``ModelRunHandle``: ``get()``, ``poll()``, ``cancel()``
- ``AsyncEvaluationHandle.future`` property (precondition for Step 1)
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
from typing import Any, Self

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import (
    AsyncEvaluationHandle,
    DistributionEnsemble,
    Evaluation,
    EvaluationConfig,
    EvaluationHandle,
    EvaluationResult,
    IncompatibleResultError,
    ModelEvaluator,
    ModelOutput,
    ModelRunHandle,
    ResumeState,
    Scenario,
)
from civic_digital_twins.dt_model.engine.numpybackend.executor import NumpyBackend
from civic_digital_twins.dt_model.model.index import DistributionIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.model.model_variant import ModelVariant
from civic_digital_twins.dt_model.simulation.runner import _encode_result, _format_value, _get_dt_model_version

# ---------------------------------------------------------------------------
# Minimal model fixture (reused from other simulation tests)
# ---------------------------------------------------------------------------


class _SimpleModel(Model):
    """Single distribution-backed input, one derived output."""

    @dataclasses.dataclass
    class Inputs:
        x: Index

    @dataclasses.dataclass
    class Outputs:
        y: Index

    def __init__(self, x: Index) -> None:
        y = Index("y", x.node * 2.0)
        super().__init__(
            "SimpleModel",
            inputs=_SimpleModel.Inputs(x=x),
            outputs=_SimpleModel.Outputs(y=y),
        )


def _make_simple_model() -> tuple[Index, _SimpleModel]:
    """Return a (x_index, model) pair ready for evaluation."""
    x = DistributionIndex("x", stats.norm, {"loc": 5.0, "scale": 1.0})
    return x, _SimpleModel(x)


def _make_result_from(model: Model | ModelVariant, size: int = 10) -> EvaluationResult:
    """Return a minimal EvaluationResult from a real evaluation via Scenario."""
    scenario = Scenario(model)
    ensemble = DistributionEnsemble(scenario, size)
    return Evaluation(scenario).evaluate(ensemble=ensemble)


# A second minimal model whose sole input is a plain scalar Index so that
# float overrides are valid (DistributionIndex requires a Distribution override).


class _ScalarModel(Model):
    """Model with one concrete scalar input — overridable with a plain float."""

    @dataclasses.dataclass
    class Inputs:
        cost: Index

    @dataclasses.dataclass
    class Outputs:
        out: Index

    def __init__(self, cost: Index) -> None:
        out = Index("out", cost.node * 2.0)
        super().__init__(
            "ScalarModel",
            inputs=_ScalarModel.Inputs(cost=cost),
            outputs=_ScalarModel.Outputs(out=out),
        )


def _make_scalar_model() -> tuple[Index, _ScalarModel]:
    """Return (cost_index, model) where cost is a plain scalar Index."""
    cost = Index("cost", 8.0)
    return cost, _ScalarModel(cost)


# ---------------------------------------------------------------------------
# Minimal concrete ModelOutput stub
# ---------------------------------------------------------------------------


class _StubOutput(ModelOutput):
    """Minimal concrete subclass used to test ModelOutput mechanics.

    Summary layer  : ``value`` (int)
    Resume payload : present when ``include_resume=True`` was passed to
                     ``__init__``; absent otherwise.
    """

    def __init__(self, value: int, *, include_resume: bool = True) -> None:
        super().__init__()
        self._value = value
        self._include_resume = include_resume

    @property
    def value(self) -> int:
        """The stored summary integer."""
        return self._value

    def to_dict(self) -> dict[str, Any]:
        """Serialise summary + optional resume payload."""
        d: dict[str, Any] = {
            "dt_model_version": _get_dt_model_version(),
            "value": self._value,
        }
        if self._include_resume:
            d["_resume"] = {"value": self._value}
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct; set ``is_resumable`` only when resume payload present."""
        obj = cls.__new__(cls)
        ModelOutput.__init__(obj)
        obj._value = data["value"]
        obj._include_resume = "_resume" in data
        if "_resume" in data:
            obj._is_resumable = True
        return obj


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


class TestEvaluationConfig:
    """Unit tests for EvaluationConfig dataclass."""

    def test_creation(self) -> None:
        """EvaluationConfig stores ensemble_size."""
        cfg = EvaluationConfig(ensemble_size=100)
        assert cfg.ensemble_size == 100

    def test_is_dataclass(self) -> None:
        """EvaluationConfig is a dataclass."""
        assert dataclasses.is_dataclass(EvaluationConfig)

    def test_field_names(self) -> None:
        """EvaluationConfig has exactly one field: ensemble_size."""
        fields = {f.name for f in dataclasses.fields(EvaluationConfig)}
        assert fields == {"ensemble_size"}

    def test_equality(self) -> None:
        """Two configs with equal ensemble_size compare equal."""
        assert EvaluationConfig(50) == EvaluationConfig(50)
        assert EvaluationConfig(50) != EvaluationConfig(99)


# ---------------------------------------------------------------------------
# ResumeState
# ---------------------------------------------------------------------------


class TestResumeState:
    """Unit tests for ResumeState dataclass."""

    def _make_result(self) -> EvaluationResult:
        """Return a minimal EvaluationResult from a real evaluation."""
        _, model = _make_simple_model()
        return _make_result_from(model)

    def test_required_fields(self) -> None:
        """ResumeState stores result and parameters."""
        result = self._make_result()
        rs = ResumeState(result=result, parameters={})
        assert rs.result is result
        assert rs.parameters == {}

    def test_optional_defaults(self) -> None:
        """Optional fields default to None / NumpyBackend."""
        result = self._make_result()
        rs = ResumeState(result=result, parameters={})
        assert rs.parameter_axes is None
        assert rs.functions is None
        assert rs.backend is NumpyBackend

    def test_optional_fields_set(self) -> None:
        """Optional fields can be set explicitly."""
        result = self._make_result()
        axes = {"t": np.linspace(0, 1, 5)}
        rs = ResumeState(result=result, parameters={}, parameter_axes=axes, functions=None, backend=NumpyBackend)
        assert rs.parameter_axes is axes

    def test_is_dataclass(self) -> None:
        """ResumeState is a dataclass."""
        assert dataclasses.is_dataclass(ResumeState)


# ---------------------------------------------------------------------------
# IncompatibleResultError
# ---------------------------------------------------------------------------


class TestIncompatibleResultError:
    """Unit tests for IncompatibleResultError."""

    def test_is_exception_subclass(self) -> None:
        """IncompatibleResultError is a subclass of Exception."""
        assert issubclass(IncompatibleResultError, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """IncompatibleResultError can be raised and caught."""
        with pytest.raises(IncompatibleResultError, match="version"):
            raise IncompatibleResultError("version mismatch")

    def test_caught_as_exception(self) -> None:
        """IncompatibleResultError is caught by a bare ``except Exception``."""
        with pytest.raises(Exception):
            raise IncompatibleResultError("test")


# ---------------------------------------------------------------------------
# ModelOutput
# ---------------------------------------------------------------------------


class TestModelOutput:
    """Unit tests for ModelOutput ABC mechanics via _StubOutput."""

    def test_cannot_instantiate_abc(self) -> None:
        """ModelOutput cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelOutput()  # type: ignore[abstract]

    def test_is_resumable_false_on_construction(self) -> None:
        """is_resumable is False immediately after __init__."""
        out = _StubOutput(42)
        assert out.is_resumable is False

    def test_is_resumable_false_when_no_resume_payload(self) -> None:
        """from_dict without resume payload leaves is_resumable False."""
        out = _StubOutput(42, include_resume=False)
        loaded = _StubOutput.from_dict(out.to_dict())
        assert loaded.is_resumable is False

    def test_is_resumable_true_when_resume_payload_present(self) -> None:
        """from_dict with resume payload sets is_resumable to True."""
        out = _StubOutput(42, include_resume=True)
        loaded = _StubOutput.from_dict(out.to_dict())
        assert loaded.is_resumable is True

    def test_roundtrip_preserves_summary(self) -> None:
        """from_dict(to_dict()) reconstructs the summary value."""
        out = _StubOutput(99)
        loaded = _StubOutput.from_dict(out.to_dict())
        assert loaded.value == 99

    def test_to_dict_contains_version_key(self) -> None:
        """to_dict includes 'dt_model_version' in the returned dict."""
        out = _StubOutput(1)
        d = out.to_dict()
        assert "dt_model_version" in d
        assert isinstance(d["dt_model_version"], str)

    def test_is_resumable_is_not_abstract(self) -> None:
        """is_resumable is a concrete property — not listed as abstract."""
        assert "is_resumable" not in getattr(ModelOutput, "__abstractmethods__", set())

    def test_to_dict_and_from_dict_are_abstract(self) -> None:
        """to_dict and from_dict are abstract methods."""
        abstract = ModelOutput.__abstractmethods__
        assert "to_dict" in abstract
        assert "from_dict" in abstract


# ---------------------------------------------------------------------------
# ModelRunHandle
# ---------------------------------------------------------------------------


class TestModelRunHandle:
    """Unit tests for ModelRunHandle generic wrapper."""

    def _resolved_future(self, value: EvaluationResult) -> concurrent.futures.Future[EvaluationResult]:
        """Return an already-resolved Future holding *value*."""
        f: concurrent.futures.Future[EvaluationResult] = concurrent.futures.Future()
        f.set_result(value)
        return f

    def _make_result(self) -> EvaluationResult:
        _, model = _make_simple_model()
        return _make_result_from(model)

    def test_get_applies_post_process(self) -> None:
        """get() applies the post-processor to the resolved EvaluationResult."""
        result = self._make_result()
        output = _StubOutput(7)
        handle: ModelRunHandle[_StubOutput] = ModelRunHandle(
            self._resolved_future(result),
            lambda _r: output,
        )
        assert handle.get() is output

    def test_poll_done_returns_true_and_output(self) -> None:
        """poll() returns (True, output) when the future is resolved."""
        result = self._make_result()
        output = _StubOutput(7)
        handle: ModelRunHandle[_StubOutput] = ModelRunHandle(
            self._resolved_future(result),
            lambda _r: output,
        )
        done, got = handle.poll()
        assert done is True
        assert got is output

    def test_poll_pending_returns_false_none(self) -> None:
        """poll() returns (False, None) when the future is still pending."""
        pending: concurrent.futures.Future[EvaluationResult] = concurrent.futures.Future()
        handle: ModelRunHandle[_StubOutput] = ModelRunHandle(
            pending,
            lambda _r: _StubOutput(0),
        )
        done, got = handle.poll()
        assert done is False
        assert got is None

    def test_cancel_pending_future(self) -> None:
        """cancel() returns True for a not-yet-started future."""
        pending: concurrent.futures.Future[EvaluationResult] = concurrent.futures.Future()
        handle: ModelRunHandle[_StubOutput] = ModelRunHandle(pending, lambda _r: _StubOutput(0))
        assert handle.cancel() is True

    def test_cancel_resolved_future(self) -> None:
        """cancel() returns False for an already-resolved future."""
        result = self._make_result()
        handle: ModelRunHandle[_StubOutput] = ModelRunHandle(
            self._resolved_future(result),
            lambda _r: _StubOutput(0),
        )
        assert handle.cancel() is False

    def test_get_with_thread_pool(self) -> None:
        """get() works when the future is backed by a real ThreadPoolExecutor."""
        _, model = _make_simple_model()
        scenario = Scenario(model)

        def _run() -> EvaluationResult:
            return Evaluation(scenario).evaluate(ensemble=DistributionEnsemble(scenario, 20))

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            handle: ModelRunHandle[_StubOutput] = ModelRunHandle(future, lambda _r: _StubOutput(5))
            output = handle.get()
        assert output.value == 5


# ---------------------------------------------------------------------------
# AsyncEvaluationHandle.future property (Step 1 precondition)
# ---------------------------------------------------------------------------


class TestAsyncEvaluationHandleFutureProperty:
    """Verify the public ``future`` property added as a Step 1 precondition."""

    def test_future_property_returns_future(self) -> None:
        """AsyncEvaluationHandle.future returns a concurrent.futures.Future."""
        _, model = _make_simple_model()
        handle = Evaluation(Scenario(model)).submit_evaluate(10)
        assert isinstance(handle, AsyncEvaluationHandle)
        assert isinstance(handle.future, concurrent.futures.Future)

    def test_future_property_is_same_object_used_by_get(self) -> None:
        """The future returned by .future resolves to the same EvaluationResult as .get()."""
        _, model = _make_simple_model()
        handle = Evaluation(Scenario(model)).submit_evaluate(10)
        result_via_future = handle.future.result()
        result_via_get = handle.get()
        assert result_via_future is result_via_get

    def test_future_property_is_done_after_get(self) -> None:
        """After .get() resolves, handle.future.done() is True."""
        _, model = _make_simple_model()
        handle = Evaluation(Scenario(model)).submit_evaluate(10)
        handle.get()
        assert handle.future.done() is True


# ---------------------------------------------------------------------------
# Scenario.overrides property (Step 2 precondition)
# ---------------------------------------------------------------------------


class TestScenarioOverrides:
    """Verify the public overrides property added as a Step 2 precondition."""

    def test_overrides_empty_for_base_scenario(self) -> None:
        """Overrides is empty when no overrides were given."""
        _, model = _make_simple_model()
        scenario = Scenario(model)
        assert scenario.overrides == {}

    def test_overrides_contains_provided_values(self) -> None:
        """Overrides contains every entry passed at construction."""
        cost, model = _make_scalar_model()
        scenario = Scenario(model, overrides={cost: 99.0})
        assert scenario.overrides == {cost: 99.0}

    def test_overrides_returns_copy(self) -> None:
        """Mutating the returned dict does not affect the Scenario."""
        cost, model = _make_scalar_model()
        scenario = Scenario(model, overrides={cost: 1.0})
        copy = scenario.overrides
        copy.clear()
        assert scenario.overrides == {cost: 1.0}


# ---------------------------------------------------------------------------
# Concrete stubs for ModelEvaluator tests
# ---------------------------------------------------------------------------


class _ResumableOutput(ModelOutput):
    """ModelOutput stub that always carries a real EvaluationResult resume payload."""

    def __init__(self, result: EvaluationResult) -> None:
        super().__init__()
        self._result = result
        self._is_resumable = True

    @property
    def raw_result(self) -> EvaluationResult:
        """The stored EvaluationResult."""
        return self._result

    def to_dict(self) -> dict:
        """Minimal stub serialisation."""
        return {}

    @classmethod
    def from_dict(cls, data: dict) -> "_ResumableOutput":
        """Not needed for these tests."""
        raise NotImplementedError


class _MinimalEvaluator(ModelEvaluator[_StubOutput]):
    """Minimal concrete evaluator for structural/introspection tests."""

    def evaluate(self, scenario: Scenario, config: EvaluationConfig) -> _StubOutput:
        """Return a stub output whose value equals ensemble_size."""
        return _StubOutput(config.ensemble_size)

    def structure(self) -> dict:
        """Return a minimal schema."""
        return {"y": {"type": "scalar"}}

    def _extract_resume_state(self, output: _StubOutput) -> ResumeState:
        """Unused in these tests."""
        raise NotImplementedError


class _ResumableEvaluator(ModelEvaluator[_ResumableOutput]):
    """Evaluator that produces resumable outputs backed by real EvaluationResults."""

    def evaluate(self, scenario: Scenario, config: EvaluationConfig) -> _ResumableOutput:
        """Run a real evaluation and wrap the result in a resumable output."""
        result = _make_result_from(scenario.model, config.ensemble_size)
        return _ResumableOutput(result)

    def structure(self) -> dict:
        """Return a minimal schema."""
        return {}

    def _extract_resume_state(self, output: _ResumableOutput) -> ResumeState:
        """Extract the stored EvaluationResult."""
        return ResumeState(result=output.raw_result, parameters={})


# ---------------------------------------------------------------------------
# ModelEvaluator — structural tests
# ---------------------------------------------------------------------------


class TestModelEvaluatorStructure:
    """Verify abstract method enforcement and default run_async behaviour."""

    def test_cannot_instantiate_without_abstract_methods(self) -> None:
        """Subclass missing evaluate() or structure() cannot be instantiated."""

        class _Incomplete(ModelEvaluator[_StubOutput]):  # type: ignore[abstract]
            pass

        with pytest.raises(TypeError):
            _Incomplete(_make_simple_model()[1])  # type: ignore[abstract]

    def test_run_async_raises_not_implemented_by_default(self) -> None:
        """run_async() raises NotImplementedError when not overridden."""
        _, model = _make_simple_model()
        evaluator = _MinimalEvaluator(model)
        with pytest.raises(NotImplementedError, match="run_async"):
            evaluator.run_async(Scenario(model), EvaluationConfig(ensemble_size=10))

    def test_evaluate_is_abstract(self) -> None:
        """Evaluate is listed as an abstract method."""
        assert "evaluate" in ModelEvaluator.__abstractmethods__

    def test_structure_is_abstract(self) -> None:
        """Structure is listed as an abstract method."""
        assert "structure" in ModelEvaluator.__abstractmethods__

    def test_extract_resume_state_is_abstract(self) -> None:
        """_extract_resume_state is listed as an abstract method."""
        assert "_extract_resume_state" in ModelEvaluator.__abstractmethods__


# ---------------------------------------------------------------------------
# ModelEvaluator — evaluate()
# ---------------------------------------------------------------------------


class TestModelEvaluatorEvaluate:
    """Verify evaluate() delegates correctly to the concrete implementation."""

    def test_evaluate_returns_model_output(self) -> None:
        """evaluate() returns an instance of ModelOutput."""
        _, model = _make_simple_model()
        evaluator = _MinimalEvaluator(model)
        output = evaluator.evaluate(Scenario(model), EvaluationConfig(ensemble_size=5))
        assert isinstance(output, ModelOutput)

    def test_evaluate_uses_ensemble_size_from_config(self) -> None:
        """_MinimalEvaluator.evaluate() propagates ensemble_size."""
        _, model = _make_simple_model()
        evaluator = _MinimalEvaluator(model)
        output = evaluator.evaluate(Scenario(model), EvaluationConfig(ensemble_size=42))
        assert output.value == 42


# ---------------------------------------------------------------------------
# _format_value helper
# ---------------------------------------------------------------------------


class TestFormatValue:
    """Unit tests for the _format_value private helper."""

    def test_none_returns_none_marker(self) -> None:
        """None produces the '(none)' marker string."""
        assert _format_value(None) == "(none)"

    def test_array_returns_shape_string(self) -> None:
        """A numpy array produces 'array<shape>'."""
        arr = np.ones((3, 4))
        assert _format_value(arr) == "array(3, 4)"

    def test_scalar_returns_str(self) -> None:
        """A plain scalar is converted via str()."""
        assert _format_value(42.0) == "42.0"


# ---------------------------------------------------------------------------
# _encode_result — KeyError path
# ---------------------------------------------------------------------------


class TestEncodeResultSkipsMissingIndex:
    """_encode_result silently skips indexes not present in the result."""

    def test_index_not_in_result_is_skipped(self) -> None:
        """An index whose node is absent from the result is omitted without error."""
        x, model = _make_simple_model()
        result = _make_result_from(model)
        extra = Index("extra", 42.0)  # node not in the evaluated state
        encoded = _encode_result(result, [x, extra])
        assert "extra" not in encoded["nodes"]


# ---------------------------------------------------------------------------
# ModelEvaluator — get_index_diffs()
# ---------------------------------------------------------------------------


class TestGetIndexDiffs:
    """Verify get_index_diffs() default implementation."""

    def test_empty_for_base_scenario(self) -> None:
        """No diffs when no overrides are active."""
        _, model = _make_simple_model()
        evaluator = _MinimalEvaluator(model)
        diffs = evaluator.get_index_diffs(Scenario(model))
        assert diffs == {}

    def test_diff_string_for_scalar_override(self) -> None:
        """Override of a scalar Index produces a 'was X \u2192 now Y' string."""
        cost, model = _make_scalar_model()
        scenario = Scenario(model, overrides={cost: 99.0})
        evaluator = _MinimalEvaluator(model)
        diffs = evaluator.get_index_diffs(scenario)
        assert cost.name in diffs
        assert "was" in diffs[cost.name]
        assert "now" in diffs[cost.name]
        assert "99.0" in diffs[cost.name]

    def test_diff_keys_match_overridden_indexes(self) -> None:
        """Diff dict has exactly one key per overridden index."""
        cost, model = _make_scalar_model()
        scenario = Scenario(model, overrides={cost: 1.0})
        evaluator = _MinimalEvaluator(model)
        diffs = evaluator.get_index_diffs(scenario)
        assert set(diffs.keys()) == {cost.name}


# ---------------------------------------------------------------------------
# ModelEvaluator — get_model_values()
# ---------------------------------------------------------------------------


class TestGetModelValues:
    """Verify get_model_values() default implementation."""

    def test_returns_model_defaults_for_base_scenario(self) -> None:
        """All values match model defaults when there are no overrides."""
        x, model = _make_simple_model()
        evaluator = _MinimalEvaluator(model)
        values = evaluator.get_model_values(Scenario(model))
        # x is abstract (DistributionIndex) → value is the frozen distribution
        # y is concrete (Index computed from x) → value is a graph node, idx.value is None
        assert x.name in values
        assert "y" in values

    def test_returns_override_for_overridden_index(self) -> None:
        """Override value is returned instead of the model default."""
        cost, model = _make_scalar_model()
        scenario = Scenario(model, overrides={cost: 7.0})
        evaluator = _MinimalEvaluator(model)
        values = evaluator.get_model_values(scenario)
        assert values[cost.name] == 7.0

    def test_keys_cover_all_model_indexes(self) -> None:
        """Every index in model.indexes has a corresponding key."""
        _, model = _make_simple_model()
        evaluator = _MinimalEvaluator(model)
        values = evaluator.get_model_values(Scenario(model))
        expected_names = {idx.name for idx in model.indexes}
        assert set(values.keys()) == expected_names


# ---------------------------------------------------------------------------
# ModelEvaluator — resume()
# ---------------------------------------------------------------------------


class TestResume:
    """Verify the resume() template method."""

    def test_resume_raises_when_not_resumable(self) -> None:
        """Resume raises IncompatibleResultError when is_resumable is False."""
        _, model = _make_simple_model()
        evaluator = _ResumableEvaluator(model)
        non_resumable = _StubOutput(0, include_resume=False)
        with pytest.raises(IncompatibleResultError):
            evaluator.resume(Scenario(model), non_resumable, EvaluationConfig(ensemble_size=10))  # type: ignore[arg-type]

    def test_resume_returns_evaluation_handle(self) -> None:
        """resume() returns an EvaluationHandle when the output is resumable."""
        _, model = _make_simple_model()
        scenario = Scenario(model)
        evaluator = _ResumableEvaluator(model)
        output = evaluator.evaluate(scenario, EvaluationConfig(ensemble_size=10))
        handle = evaluator.resume(scenario, output, EvaluationConfig(ensemble_size=5))
        assert isinstance(handle, EvaluationHandle)

    def test_resume_handle_can_be_extended(self) -> None:
        """The returned EvaluationHandle can be extended with more samples."""
        _, model = _make_simple_model()
        scenario = Scenario(model)
        evaluator = _ResumableEvaluator(model)
        output = evaluator.evaluate(scenario, EvaluationConfig(ensemble_size=10))
        handle = evaluator.resume(scenario, output, EvaluationConfig(ensemble_size=5))
        extended = handle.extend(ensemble_size=5)
        assert extended is handle.result
