"""Unit tests for Evaluation.evaluate() — both 1-D and axes (grid) modes."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.model.index import Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(*indexes):
    """Wrap indexes in a named model."""
    return Model("test", list(indexes))


# ---------------------------------------------------------------------------
# 1-D batch mode
# ---------------------------------------------------------------------------


def test_1d_single_scenario_constant_model():
    """A fully-concrete model evaluates with an empty scenario."""
    I_a = Index("a", 3.0)
    I_b = Index("b", 4.0)
    model = _make_model(I_a, I_b)
    formula = I_a.node + I_b.node

    # No abstract indexes → scenarios list is empty; still evaluates constants.
    state = Evaluation(model).evaluate([], [formula])
    assert np.isclose(state.values[formula], 7.0)


def test_1d_single_scenario_placeholder():
    """Single scenario with one placeholder index."""
    I_x = Index("x", None)
    I_scale = Index("scale", 2.0)
    model = _make_model(I_x, I_scale)
    formula = I_scale.node * I_x.node

    state = Evaluation(model).evaluate([(1.0, {I_x: 5.0})], [formula])
    # Batch dim: shape (1,); value = 2 * 5 = 10
    assert state.values[formula].shape == (1,)
    assert np.isclose(state.values[formula][0], 10.0)


def test_1d_multiple_scenarios():
    """Multiple scenarios are stacked; result has one entry per scenario."""
    I_x = Index("x", None)
    model = _make_model(I_x)
    formula = I_x.node * I_x.node

    scenarios = [(0.5, {I_x: 2.0}), (0.5, {I_x: 3.0})]
    state = Evaluation(model).evaluate(scenarios, [formula])
    assert state.values[formula].shape == (2,)
    assert np.isclose(state.values[formula][0], 4.0)
    assert np.isclose(state.values[formula][1], 9.0)


def test_1d_raises_on_unresolved_abstract_index():
    """ValueError is raised when a scenario is missing an abstract index."""
    I_x = Index("x", None)
    I_y = Index("y", None)
    model = _make_model(I_x, I_y)

    with pytest.raises(ValueError, match="abstract index"):
        Evaluation(model).evaluate([(1.0, {I_x: 1.0})], [I_x.node])


# ---------------------------------------------------------------------------
# axes (grid) mode — shape checks
# ---------------------------------------------------------------------------


def test_axes_single_axis_result_shape():
    """Single axis index produces result shape (N, 1)."""
    I_x = Index("x", None)
    model = _make_model(I_x)
    xs = np.array([1.0, 2.0, 3.0])

    state = Evaluation(model).evaluate(
        [(1.0, {})], [I_x.node], axes={I_x: xs}
    )
    assert state.values[I_x.node].shape == (3, 1)


def test_axes_two_axes_result_shape():
    """Two axis indexes produce result shape (N0, N1, 1)."""
    I_x = Index("x", None)
    I_y = Index("y", None)
    model = _make_model(I_x, I_y)
    xs = np.array([1.0, 2.0])
    ys = np.array([10.0, 20.0, 30.0])

    state = Evaluation(model).evaluate(
        [(1.0, {})], [I_x.node, I_y.node], axes={I_x: xs, I_y: ys}
    )
    assert state.values[I_x.node].shape == (2, 1, 1)
    assert state.values[I_y.node].shape == (1, 3, 1)


def test_axes_non_axis_abstract_has_shape_1_1_s():
    """A non-axis abstract index has shape (1, …, 1, S)."""
    I_x = Index("x", None)
    I_factor = Index("factor", None)
    model = _make_model(I_x, I_factor)
    xs = np.array([1.0, 2.0, 3.0])
    scenarios = [(0.5, {I_factor: 1.0}), (0.5, {I_factor: 2.0})]

    state = Evaluation(model).evaluate(
        scenarios, [I_factor.node], axes={I_x: xs}
    )
    # Non-axis abstract: shape (1, S) = (1, 2)
    assert state.values[I_factor.node].shape == (1, 2)


# ---------------------------------------------------------------------------
# axes (grid) mode — values
# ---------------------------------------------------------------------------


def test_axes_single_axis_formula_values():
    """Formula with a single axis index evaluates on the full grid."""
    I_x = Index("x", None)
    I_scale = Index("scale", 3.0)
    model = _make_model(I_x, I_scale)
    formula = I_scale.node * I_x.node
    xs = np.array([1.0, 2.0, 4.0])

    state = Evaluation(model).evaluate(
        [(1.0, {})], [formula], axes={I_x: xs}
    )
    # shape (3, 1); marginalise: tensordot(..., [1.0], axes=([-1],[0])) → (3,)
    result = np.tensordot(state.values[formula], np.array([1.0]), axes=([-1], [0]))
    assert np.allclose(result, [3.0, 6.0, 12.0])


def test_axes_two_axes_additive_formula():
    """Sum formula over two axes produces the correct (N0, N1, S) array."""
    I_x = Index("x", None)
    I_y = Index("y", None)
    model = _make_model(I_x, I_y)
    formula = I_x.node + I_y.node
    xs = np.array([1.0, 2.0])
    ys = np.array([10.0, 20.0, 30.0])

    state = Evaluation(model).evaluate(
        [(1.0, {})], [formula], axes={I_x: xs, I_y: ys}
    )
    result = np.tensordot(state.values[formula], np.array([1.0]), axes=([-1], [0]))
    # result[i, j] = xs[i] + ys[j]
    expected = xs[:, None] + ys[None, :]
    assert np.allclose(result, expected)


def test_axes_non_axis_factor_marginalised_correctly():
    """Weighted marginalisation over a non-axis index gives the correct mean."""
    I_x = Index("x", None)
    I_factor = Index("factor", None)
    model = _make_model(I_x, I_factor)
    formula = I_x.node * I_factor.node
    xs = np.array([1.0, 2.0, 3.0])
    # Two equiprobable scenarios: factor=1 and factor=3 → mean factor=2
    scenarios = [(0.5, {I_factor: 1.0}), (0.5, {I_factor: 3.0})]

    state = Evaluation(model).evaluate(scenarios, [formula], axes={I_x: xs})
    weights = np.array([0.5, 0.5])
    result = np.tensordot(state.values[formula], weights, axes=([-1], [0]))
    # result[i] = xs[i] * mean_factor = xs[i] * 2
    assert np.allclose(result, [2.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# axes (grid) mode — error handling
# ---------------------------------------------------------------------------


def test_axes_raises_on_unresolved_non_axis_abstract():
    """ValueError when a non-axis abstract index is missing from a scenario."""
    I_x = Index("x", None)
    I_missing = Index("missing", None)
    model = _make_model(I_x, I_missing)

    with pytest.raises(ValueError, match="abstract index"):
        Evaluation(model).evaluate(
            [(1.0, {})], [I_x.node], axes={I_x: np.array([1.0])}
        )


def test_axes_axis_index_not_required_in_scenario():
    """Axis indexes do not need to appear in the scenario assignments."""
    I_x = Index("x", None)
    model = _make_model(I_x)

    # Should not raise — I_x is an axis, not required in scenario dict.
    state = Evaluation(model).evaluate(
        [(1.0, {})], [I_x.node], axes={I_x: np.array([5.0, 10.0])}
    )
    assert state.values[I_x.node].shape == (2, 1)
