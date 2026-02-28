"""Tests for GenericIndex, Index, and TimeseriesIndex."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model import ConstIndex, GenericIndex, Index, TimeseriesIndex
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor


def test_timeseries_index_construction():
    """Test basic construction of a TimeseriesIndex."""
    values = np.array([1.0, 2.0, 3.0])
    idx = TimeseriesIndex("cap", values)
    assert idx.name == "cap"
    assert isinstance(idx.node, graph.timeseries_constant)
    assert idx.values is not None
    assert np.array_equal(idx.values, values)


def test_timeseries_index_value_attribute():
    """Test that the value attribute holds the numpy array."""
    values = np.array([10.0, 20.0, 30.0])
    idx = TimeseriesIndex("cap", values)
    assert isinstance(idx.value, np.ndarray)
    assert np.array_equal(idx.value, values)


def test_timeseries_index_evaluation():
    """Test that the TimeseriesIndex node evaluates to its values."""
    values = np.array([10.0, 20.0, 30.0])
    idx = TimeseriesIndex("cap", values)
    plan = linearize.forest(idx.node)
    state = executor.State({})
    executor.evaluate_nodes(state, *plan)
    assert np.array_equal(state.values[idx.node], values)


def test_timeseries_index_values_setter():
    """Test that updating values refreshes the graph node."""
    idx = TimeseriesIndex("cap", np.array([1.0, 2.0]))
    old_node = idx.node

    new_values = np.array([3.0, 4.0])
    idx.values = new_values

    assert np.array_equal(idx.values, new_values)
    assert isinstance(idx.value, np.ndarray)
    assert np.array_equal(idx.value, new_values)
    # A new node is created
    assert idx.node is not old_node

    # The new node evaluates correctly
    plan = linearize.forest(idx.node)
    state = executor.State({})
    executor.evaluate_nodes(state, *plan)
    assert np.array_equal(state.values[idx.node], new_values)


def test_timeseries_index_values_setter_no_change():
    """Test that setting the same values does not replace the node."""
    values = np.array([1.0, 2.0])
    idx = TimeseriesIndex("cap", values)
    old_node = idx.node

    # Setting identical values should not replace the node
    idx.values = np.array([1.0, 2.0])
    assert idx.node is old_node


def test_timeseries_index_str():
    """Test the string representation of a TimeseriesIndex."""
    idx = TimeseriesIndex("cap", np.array([1.0, 2.0]))
    assert str(idx) == "timeseries_idx([1.0, 2.0])"


def test_timeseries_index_in_arithmetic():
    """Test that a TimeseriesIndex node participates correctly in formulas."""
    values = np.array([10.0, 20.0, 30.0])
    idx = TimeseriesIndex("cap", values)
    halved = idx.node * graph.constant(0.5)
    plan = linearize.forest(halved)
    state = executor.State({})
    executor.evaluate_nodes(state, *plan)
    assert np.allclose(state.values[halved], [5.0, 10.0, 15.0])


# ---------------------------------------------------------------------------
# TimeseriesIndex — placeholder (no values)
# ---------------------------------------------------------------------------


def test_timeseries_index_no_values():
    """Test construction of a TimeseriesIndex with no values (placeholder mode)."""
    idx = TimeseriesIndex("inflow")
    assert isinstance(idx.node, graph.timeseries_placeholder)
    assert idx.values is None
    assert idx.value is None


def test_timeseries_index_placeholder_raises_without_state():
    """Test that evaluating a value-less TimeseriesIndex without a state entry raises."""
    idx = TimeseriesIndex("inflow")
    plan = linearize.forest(idx.node)
    state = executor.State({})
    with pytest.raises(executor.PlaceholderValueNotProvided):
        executor.evaluate_nodes(state, *plan)


def test_timeseries_index_placeholder_evaluates_with_state():
    """Test that a value-less TimeseriesIndex evaluates correctly when state is provided."""
    idx = TimeseriesIndex("inflow")
    values = np.array([1.0, 2.0, 3.0])
    plan = linearize.forest(idx.node)
    state = executor.State({idx.node: values})
    executor.evaluate_nodes(state, *plan)
    assert np.array_equal(state.values[idx.node], values)


def test_timeseries_index_none_to_array():
    """Test that assigning values to a placeholder TimeseriesIndex switches it to constant."""
    idx = TimeseriesIndex("inflow")
    old_node = idx.node
    idx.values = np.array([1.0, 2.0])
    assert isinstance(idx.node, graph.timeseries_constant)
    assert idx.node is not old_node
    assert np.array_equal(idx.values, [1.0, 2.0])


def test_timeseries_index_array_to_none():
    """Test that setting values to None switches a constant TimeseriesIndex to placeholder."""
    idx = TimeseriesIndex("inflow", np.array([1.0, 2.0]))
    idx.values = None
    assert idx.values is None
    assert idx.value is None
    assert isinstance(idx.node, graph.timeseries_placeholder)


def test_timeseries_index_str_placeholder():
    """Test the string representation of a placeholder TimeseriesIndex."""
    idx = TimeseriesIndex("inflow")
    assert str(idx) == "timeseries_idx(placeholder)"


# ---------------------------------------------------------------------------
# TimeseriesIndex — formula mode (graph.Node as value)
# ---------------------------------------------------------------------------


def test_timeseries_index_formula_construction():
    """TimeseriesIndex accepts a graph.Node and stores it as the node."""
    ts = TimeseriesIndex("inflow")
    result = TimeseriesIndex("outflow", ts.node * ts.node)
    assert isinstance(result.node, graph.multiply)
    assert result.values is None


def test_timeseries_index_formula_str():
    """String representation in formula mode."""
    ts = TimeseriesIndex("inflow")
    result = TimeseriesIndex("outflow", ts.node * ts.node)
    assert str(result).startswith("timeseries_idx(")


def test_timeseries_index_formula_evaluation():
    """TimeseriesIndex in formula mode evaluates correctly."""
    ts = TimeseriesIndex("inflow")
    result = TimeseriesIndex("outflow", ts.node * ts.node)
    plan = linearize.forest(result.node)
    values = np.array([2.0, 3.0, 4.0])
    state = executor.State({ts.node: values})
    executor.evaluate_nodes(state, *plan)
    assert np.allclose(state.values[result.node], values**2)


def test_timeseries_index_formula_mode():
    """TimeseriesIndex in formula mode wraps the given graph node."""
    ts = TimeseriesIndex("inflow")
    result = TimeseriesIndex("outflow", ts.node * graph.constant(2.0))
    assert result.node is not None


def test_timeseries_index_formula_via_operators():
    """TimeseriesIndex formula mode works with GenericIndex operators."""
    ts = TimeseriesIndex("inflow")
    result = TimeseriesIndex("outflow", ts * ts)
    plan = linearize.forest(result.node)
    values = np.array([2.0, 3.0, 4.0])
    state = executor.State({ts.node: values})
    executor.evaluate_nodes(state, *plan)
    assert np.allclose(state.values[result.node], values**2)


# ---------------------------------------------------------------------------
# GenericIndex arithmetic operators
# ---------------------------------------------------------------------------


def _eval(node: graph.Node) -> np.ndarray:
    """Evaluate a graph node with no external placeholder values."""
    state = executor.State({})
    executor.evaluate_nodes(state, *linearize.forest(node))
    return state.values[node]


def test_generic_index_is_abstract():
    """GenericIndex cannot be instantiated directly."""
    with pytest.raises(TypeError):
        GenericIndex()  # type: ignore[abstract]


def test_index_is_generic_index():
    """Index is a subclass of GenericIndex."""
    assert issubclass(Index, GenericIndex)
    assert isinstance(Index("x", 1.0), GenericIndex)


def test_timeseries_index_is_generic_index():
    """TimeseriesIndex is a subclass of GenericIndex."""
    assert isinstance(TimeseriesIndex("ts", np.array([1.0])), GenericIndex)


def test_index_add_scalar():
    """Index + scalar produces an add node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = idx + 2.0
    assert isinstance(node, graph.add)
    assert np.isclose(_eval(node), 5.0)


def test_index_radd_scalar():
    """Scalar + index produces an add node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = 2.0 + idx
    assert isinstance(node, graph.add)
    assert np.isclose(_eval(node), 5.0)


def test_index_sub():
    """Index - scalar produces a subtract node with the correct value."""
    idx = ConstIndex("a", 5.0)
    node = idx - 2.0
    assert isinstance(node, graph.subtract)
    assert np.isclose(_eval(node), 3.0)


def test_index_rsub():
    """Scalar - index produces a subtract node with the correct value."""
    idx = ConstIndex("a", 2.0)
    node = 5.0 - idx
    assert isinstance(node, graph.subtract)
    assert np.isclose(_eval(node), 3.0)


def test_index_mul_scalar():
    """Index * scalar produces a multiply node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = idx * 4.0
    assert isinstance(node, graph.multiply)
    assert np.isclose(_eval(node), 12.0)


def test_index_rmul_scalar():
    """Scalar * index produces a multiply node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = 4.0 * idx
    assert isinstance(node, graph.multiply)
    assert np.isclose(_eval(node), 12.0)


def test_index_truediv():
    """Index / scalar produces a divide node with the correct value."""
    idx = ConstIndex("a", 6.0)
    node = idx / 2.0
    assert isinstance(node, graph.divide)
    assert np.isclose(_eval(node), 3.0)


def test_index_rtruediv():
    """Scalar / index produces a divide node with the correct value."""
    idx = ConstIndex("a", 2.0)
    node = 6.0 / idx
    assert isinstance(node, graph.divide)
    assert np.isclose(_eval(node), 3.0)


def test_index_pow():
    """Index ** scalar produces a power node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = idx**2.0
    assert isinstance(node, graph.power)
    assert np.isclose(_eval(node), 9.0)


def test_index_rpow():
    """Scalar ** index produces a power node with the correct value."""
    idx = ConstIndex("a", 3.0)
    node = 2.0**idx
    assert isinstance(node, graph.power)
    assert np.isclose(_eval(node), 8.0)


def test_index_add_index():
    """Two Index objects can be combined with operators directly."""
    a = ConstIndex("a", 3.0)
    b = ConstIndex("b", 4.0)
    node = a + b
    assert isinstance(node, graph.add)
    assert np.isclose(_eval(node), 7.0)


def test_index_mul_index():
    """Two Index objects multiplied together produce a multiply node."""
    a = ConstIndex("a", 3.0)
    b = ConstIndex("b", 4.0)
    node = a * b
    assert isinstance(node, graph.multiply)
    assert np.isclose(_eval(node), 12.0)


def test_timeseries_index_arithmetic():
    """TimeseriesIndex participates in formulas without .node access."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
    node = ts * 2.0
    assert isinstance(node, graph.multiply)
    assert np.allclose(_eval(node), [2.0, 4.0, 6.0])


def test_index_comparison_operators():
    """Comparison operators return graph nodes (lazy evaluation)."""
    a = ConstIndex("a", 3.0)
    b = ConstIndex("b", 5.0)

    assert isinstance(a == b, graph.equal)
    assert isinstance(a != b, graph.not_equal)
    assert isinstance(a < b, graph.less)
    assert isinstance(a <= b, graph.less_equal)
    assert isinstance(a > b, graph.greater)
    assert isinstance(a >= b, graph.greater_equal)


def test_index_eq_evaluates_correctly():
    """== returns True when both operands have the same value."""
    node = ConstIndex("a", 3.0) == ConstIndex("b", 3.0)
    assert bool(_eval(node))


def test_index_lt_evaluates_correctly():
    """< returns True when the left operand is smaller."""
    node = ConstIndex("a", 2.0) < ConstIndex("b", 5.0)
    assert bool(_eval(node))


def test_index_hash_is_identity_based():
    """Index objects remain usable as dict keys despite overriding __eq__."""
    a = ConstIndex("a", 1.0)
    b = ConstIndex("b", 1.0)
    d = {a: "x", b: "y"}
    assert d[a] == "x"
    assert d[b] == "y"
    assert a in d
    assert b in d


# ---------------------------------------------------------------------------
# GenericIndex.__neg__
# ---------------------------------------------------------------------------


def test_index_neg_returns_negate_node():
    """__neg__ returns a negate graph node wrapping the index's node."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
    result = -ts
    assert isinstance(result, graph.negate)
    assert result.node is ts.node


def test_index_neg_evaluates_correctly():
    """__neg__ evaluates to the element-wise negation of the index values."""
    ts = TimeseriesIndex("ts", np.array([1.0, -2.0, 3.0]))
    result = _eval(-ts)
    assert np.allclose(result, [-1.0, 2.0, -3.0])


# ---------------------------------------------------------------------------
# GenericIndex.sum / mean
# ---------------------------------------------------------------------------


def test_index_sum_returns_project_using_sum_node():
    """sum() returns a project_using_sum graph node."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
    node = ts.sum()
    assert isinstance(node, graph.project_using_sum)


def test_index_sum_evaluates_correctly():
    """sum() over a 1-D timeseries returns the total."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
    node = ts.sum()
    result = _eval(node)
    assert np.isclose(result, 6.0)


def test_index_sum_batched():
    """sum() over a (size, T) batched timeseries gives per-sample totals with keepdims.

    The result is (size, 1) so it broadcasts correctly against timeseries
    of shape (T,) or (size, T).
    """
    ts = TimeseriesIndex("ts")
    node = ts.sum()
    values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    state = executor.State({ts.node: values})
    executor.evaluate_nodes(state, *linearize.forest(node))
    result = state.values[node]
    assert result.shape == (2, 1)
    assert np.allclose(result, [[6.0], [15.0]])


def test_index_mean_returns_project_using_mean_node():
    """mean() returns a project_using_mean graph node."""
    ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
    node = ts.mean()
    assert isinstance(node, graph.project_using_mean)


def test_index_mean_evaluates_correctly():
    """mean() over a 1-D timeseries returns the average."""
    ts = TimeseriesIndex("ts", np.array([2.0, 4.0, 6.0]))
    node = ts.mean()
    result = _eval(node)
    assert np.isclose(result, 4.0)
