"""Tests for GenericIndex.shift/roll/diff/cumulative.

Mirrors tests/dt_model/symbols/test_index_reduction_methods.py, covering the
shape-preserving per-axis operators alongside the reduction methods tested
there.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import DOMAIN, Axis
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.model.index import ConstTimeseriesIndex, Index, TimeseriesIndex

_TIME_AXIS = Axis("time", DOMAIN)


class TestIndexAxisMethodCreation:
    """Test that index axis methods return correct node types."""

    def test_shift_returns_shift_node(self):
        """shift() returns a graph.shift node."""
        idx = TimeseriesIndex("test_index")
        result = idx.shift()
        assert isinstance(result, graph.shift)
        assert result.axis == _TIME_AXIS

    def test_roll_returns_roll_node(self):
        """roll() returns a graph.roll node."""
        idx = TimeseriesIndex("test_index")
        result = idx.roll()
        assert isinstance(result, graph.roll)
        assert result.axis == _TIME_AXIS

    def test_cumulative_returns_cumulative_node(self):
        """cumulative() returns a graph.cumulative node."""
        idx = TimeseriesIndex("test_index")
        result = idx.cumulative()
        assert isinstance(result, graph.cumulative)
        assert result.axis == _TIME_AXIS

    def test_diff_returns_subtract_node(self):
        """diff() composes as node - shift(node), so it returns a graph.subtract node."""
        idx = TimeseriesIndex("test_index")
        result = idx.diff()
        assert isinstance(result, graph.subtract)
        assert isinstance(result.right, graph.shift)

    def test_shift_default_periods_and_fill_value(self):
        """shift() defaults to periods=1, fill_value=0.0."""
        idx = TimeseriesIndex("test_index")
        result = idx.shift()
        assert isinstance(result, graph.shift)
        assert result.periods == 1
        assert result.fill_value == 0.0

    def test_shift_custom_periods_and_fill_value(self):
        """shift() forwards periods and fill_value."""
        idx = TimeseriesIndex("test_index")
        result = idx.shift(2, fill_value=-1.0)
        assert isinstance(result, graph.shift)
        assert result.periods == 2
        assert result.fill_value == -1.0

    def test_roll_custom_periods(self):
        """roll() forwards periods."""
        idx = TimeseriesIndex("test_index")
        result = idx.roll(3)
        assert isinstance(result, graph.roll)
        assert result.periods == 3


class TestIndexAxisMethodExecution:
    """Test execution of index axis methods against ConstTimeseriesIndex."""

    def test_shift_evaluation(self):
        """shift() moves values forward, filling the exposed position with 0."""
        ts = ConstTimeseriesIndex("ts", np.array([1.0, 2.0, 3.0, 4.0]))
        node = ts.shift()
        plan = linearize.forest(node)
        state = executor.State({}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[node], [0.0, 1.0, 2.0, 3.0])

    def test_roll_evaluation(self):
        """roll() circularly shifts values."""
        ts = ConstTimeseriesIndex("ts", np.array([1.0, 2.0, 3.0, 4.0]))
        node = ts.roll()
        plan = linearize.forest(node)
        state = executor.State({}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[node], [4.0, 1.0, 2.0, 3.0])

    def test_cumulative_evaluation(self):
        """cumulative() matches np.cumsum."""
        ts = ConstTimeseriesIndex("ts", np.array([1.0, 2.0, 3.0, 4.0]))
        node = ts.cumulative()
        plan = linearize.forest(node)
        state = executor.State({}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[node], [1.0, 3.0, 6.0, 10.0])

    def test_diff_evaluation(self):
        """diff() is the first difference, first position vs. fill_value=0."""
        ts = ConstTimeseriesIndex("ts", np.array([1.0, 3.0, 6.0, 10.0]))
        node = ts.diff()
        plan = linearize.forest(node)
        state = executor.State({}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[node], [1.0, 2.0, 3.0, 4.0])

    def test_diff_custom_fill_value(self):
        """diff() uses fill_value for the position exposed by the underlying shift."""
        ts = ConstTimeseriesIndex("ts", np.array([5.0, 8.0, 12.0]))
        node = ts.diff(fill_value=5.0)
        plan = linearize.forest(node)
        state = executor.State({}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[node], [0.0, 3.0, 4.0])


class TestIndexAxisMethodBatched:
    """Test axis methods on batched (size, T) timeseries."""

    def test_shift_batched(self):
        """shift() on a (size, T) batched timeseries shifts along the last (time) axis."""
        ts = TimeseriesIndex("ts")
        node = ts.shift()
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({ts.node: values}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == values.shape
        assert np.array_equal(result, [[0.0, 1.0, 2.0], [0.0, 4.0, 5.0]])

    def test_cumulative_batched(self):
        """cumulative() on a (size, T) batched timeseries accumulates along time."""
        ts = TimeseriesIndex("ts")
        node = ts.cumulative()
        values = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({ts.node: values}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *linearize.forest(node))
        result = state.values[node]
        assert result.shape == values.shape
        assert np.array_equal(result, [[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]])


class TestIndexAxisMethodWithPlaceholders:
    """Test index axis methods with placeholder-backed indexes."""

    def test_roll_with_placeholder(self):
        """roll() works on an Index wrapping an array_placeholder."""
        x_placeholder = graph.array_placeholder("x", axes=(_TIME_AXIS,))
        idx = Index("x", x_placeholder)
        roll_node = idx.roll()
        plan = linearize.forest(roll_node)

        x_val = np.array([1.0, 2.0, 3.0])
        state = executor.State({x_placeholder: x_val}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)

        assert np.array_equal(state.values[roll_node], [3.0, 1.0, 2.0])


class TestIndexAxisMethodMultipleDomainAxes:
    """Test that axis methods require an explicit axis= for multi-axis indexes."""

    def test_shift_requires_explicit_axis(self):
        """shift() raises ValueError without axis= when the index carries several DOMAIN axes."""
        x_axis = Axis("x", DOMAIN)
        y_axis = Axis("y", DOMAIN)
        placeholder = graph.array_placeholder("field", axes=(x_axis, y_axis))
        idx = Index("field", placeholder)
        with pytest.raises(ValueError, match="needs an explicit axis="):
            idx.shift()

    def test_shift_with_explicit_axis(self):
        """shift(axis=...) resolves correctly for a multi-axis index."""
        x_axis = Axis("x", DOMAIN)
        y_axis = Axis("y", DOMAIN)
        placeholder = graph.array_placeholder("field", axes=(x_axis, y_axis))
        idx = Index("field", placeholder)
        result = idx.shift(axis=x_axis)
        assert isinstance(result, graph.shift)
        assert result.axis == x_axis
