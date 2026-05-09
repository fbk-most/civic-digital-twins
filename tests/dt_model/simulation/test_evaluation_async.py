# SPDX-License-Identifier: Apache-2.0
"""Tests for AsyncEvaluationHandle and submit_evaluate() — Step 4 of engine-control.md."""

import concurrent.futures
import dataclasses
import time

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model.model.index import DistributionIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.evaluation import _DEFAULT_EXECUTOR, Evaluation
from civic_digital_twins.dt_model.simulation.handle import AsyncEvaluationHandle, EvaluationHandle

# ---------------------------------------------------------------------------
# Shared model fixture (same pattern as test_evaluation_handle.py)
# ---------------------------------------------------------------------------


class _SimpleModel(Model):
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


def _make_simple() -> tuple[Index, _SimpleModel]:
    x = DistributionIndex("x", stats.norm, {"loc": 5.0, "scale": 1.0})
    return x, _SimpleModel(x)


# ---------------------------------------------------------------------------
# submit_evaluate — basic creation and type
# ---------------------------------------------------------------------------


def test_submit_evaluate_returns_async_handle() -> None:
    """submit_evaluate returns an AsyncEvaluationHandle."""
    _, model = _make_simple()
    handle = Evaluation(model).submit_evaluate(50)
    assert isinstance(handle, AsyncEvaluationHandle)


def test_async_handle_is_subclass_of_evaluation_handle() -> None:
    """AsyncEvaluationHandle is a subclass of EvaluationHandle."""
    assert issubclass(AsyncEvaluationHandle, EvaluationHandle)


# ---------------------------------------------------------------------------
# get() — blocking retrieval
# ---------------------------------------------------------------------------


def test_get_returns_evaluation_result_with_correct_shape() -> None:
    """get() blocks until done and returns a result with the expected shape."""
    _, model = _make_simple()
    handle = Evaluation(model).submit_evaluate(40)
    result = handle.get()
    assert result[model.outputs.y].shape == (40,)


def test_get_idempotent() -> None:
    """Repeated get() calls return the same cached result object."""
    _, model = _make_simple()
    handle = Evaluation(model).submit_evaluate(30)
    r1 = handle.get()
    r2 = handle.get()
    assert r1 is r2


def test_result_property_after_get() -> None:
    """handle.result is accessible after get() has been called."""
    _, model = _make_simple()
    handle = Evaluation(model).submit_evaluate(30)
    handle.get()
    arr = handle.result[model.outputs.y]
    assert arr.shape == (30,)


# ---------------------------------------------------------------------------
# result property — RuntimeError before done
# ---------------------------------------------------------------------------


def test_result_raises_before_done() -> None:
    """handle.result raises RuntimeError if the future has not yet completed."""
    _, model = _make_simple()
    # Use a single-thread executor and submit a slow job first to block
    # the worker, ensuring our handle's future is still pending.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as slow_exec:
        # Block the sole worker thread.
        blocker = slow_exec.submit(time.sleep, 5)
        handle = Evaluation(model).submit_evaluate(20, exec=slow_exec)
        try:
            with pytest.raises(RuntimeError, match="still running"):
                _ = handle.result
        finally:
            blocker.cancel()


# ---------------------------------------------------------------------------
# poll() — non-blocking status check
# ---------------------------------------------------------------------------


def test_poll_returns_true_after_get() -> None:
    """poll() returns (True, result) after the future resolves."""
    _, model = _make_simple()
    handle = Evaluation(model).submit_evaluate(30)
    handle.get()  # ensure resolved
    done, result = handle.poll()
    assert done is True
    assert result is not None
    assert result[model.outputs.y].shape == (30,)


def test_poll_returns_false_while_running() -> None:
    """poll() returns (False, None) while the future is still pending."""
    _, model = _make_simple()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as slow_exec:
        blocker = slow_exec.submit(time.sleep, 5)
        handle = Evaluation(model).submit_evaluate(20, exec=slow_exec)
        try:
            done, result = handle.poll()
            assert done is False
            assert result is None
        finally:
            blocker.cancel()


# ---------------------------------------------------------------------------
# extend() after get()
# ---------------------------------------------------------------------------


def test_extend_after_get_grows_ensemble() -> None:
    """extend() works normally after get() has resolved the future."""
    _, model = _make_simple()
    handle = Evaluation(model).submit_evaluate(30, rng=np.random.default_rng(1))
    handle.get()
    handle.extend(20)
    assert handle.result[model.outputs.y].shape == (50,)


def test_extend_raises_before_done() -> None:
    """extend() raises RuntimeError if called before the future completes."""
    _, model = _make_simple()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as slow_exec:
        blocker = slow_exec.submit(time.sleep, 5)
        handle = Evaluation(model).submit_evaluate(20, exec=slow_exec)
        try:
            with pytest.raises(RuntimeError, match="cannot extend"):
                handle.extend(10)
        finally:
            blocker.cancel()


# ---------------------------------------------------------------------------
# submit_evaluate matches evaluate_incremental (same RNG seed)
# ---------------------------------------------------------------------------


def test_submit_evaluate_matches_evaluate_incremental() -> None:
    """submit_evaluate(N) + get() is numerically identical to evaluate_incremental(N)."""
    _, model = _make_simple()
    ev = Evaluation(model)

    sync_handle = ev.evaluate_incremental(50, rng=np.random.default_rng(11))
    async_handle = ev.submit_evaluate(50, rng=np.random.default_rng(11))
    async_handle.get()

    np.testing.assert_array_equal(
        sync_handle.result[model.outputs.y],
        async_handle.result[model.outputs.y],
    )


# ---------------------------------------------------------------------------
# _DEFAULT_EXECUTOR
# ---------------------------------------------------------------------------


def test_default_executor_is_thread_pool() -> None:
    """_DEFAULT_EXECUTOR is a ThreadPoolExecutor."""
    assert isinstance(_DEFAULT_EXECUTOR, concurrent.futures.ThreadPoolExecutor)


def test_custom_executor_is_used() -> None:
    """A caller-supplied executor is used instead of the default."""
    _, model = _make_simple()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as custom_exec:
        handle = Evaluation(model).submit_evaluate(30, exec=custom_exec)
        result = handle.get()
    assert result[model.outputs.y].shape == (30,)
