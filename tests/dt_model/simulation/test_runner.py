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
    EvaluationResult,
    IncompatibleResultError,
    ModelOutput,
    ModelRunHandle,
    ResumeState,
    Scenario,
)
from civic_digital_twins.dt_model.engine.numpybackend.executor import NumpyBackend
from civic_digital_twins.dt_model.model.index import DistributionIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.runner import _get_dt_model_version

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


def _make_result_from(model: _SimpleModel, size: int = 10) -> EvaluationResult:
    """Return a minimal EvaluationResult from a real evaluation via Scenario."""
    scenario = Scenario(model)
    ensemble = DistributionEnsemble(scenario, size)
    return Evaluation(scenario).evaluate(ensemble=ensemble)


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
