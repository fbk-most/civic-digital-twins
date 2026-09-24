"""Tests for the shape-preserving per-axis operators: shift, roll, cumulative.

Unlike the ProjectionOp family (tested in test_axis_reduction_operators.py),
these operators keep the named axis in output_axes and preserve array shape.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import DOMAIN, Axis
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor, numpy_ast

_TIME_AXIS = Axis("time", DOMAIN)


class TestAxisOpCreation:
    """Test creation of shift/roll/cumulative nodes."""

    def test_shift_creation(self):
        """Shift stores node, axis, periods, and fill_value."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS, periods=2, fill_value=-1.0)
        assert result.node is x
        assert result.axis == _TIME_AXIS
        assert result.periods == 2
        assert result.fill_value == -1.0

    def test_shift_defaults(self):
        """Shift defaults to periods=1, fill_value=0.0."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS)
        assert result.periods == 1
        assert result.fill_value == 0.0

    def test_roll_creation(self):
        """Roll stores node, axis, and periods (no fill_value)."""
        x = graph.placeholder("x")
        result = graph.roll(x, _TIME_AXIS, periods=3)
        assert result.node is x
        assert result.axis == _TIME_AXIS
        assert result.periods == 3

    def test_cumulative_creation(self):
        """Cumulative stores node and axis only."""
        x = graph.placeholder("x")
        result = graph.cumulative(x, _TIME_AXIS)
        assert result.node is x
        assert result.axis == _TIME_AXIS


class TestAxisOpOutputAxesShapePreserving:
    """Unlike ProjectionOp, output_axes keeps the operated-on axis."""

    def test_shift_preserves_output_axes(self):
        """Shift does not remove the axis from output_axes."""
        x = graph.array_placeholder("x", axes=(_TIME_AXIS,))
        result = graph.shift(x, _TIME_AXIS)
        assert result.output_axes == (_TIME_AXIS,)

    def test_roll_preserves_output_axes(self):
        """Roll does not remove the axis from output_axes."""
        x = graph.array_placeholder("x", axes=(_TIME_AXIS,))
        result = graph.roll(x, _TIME_AXIS)
        assert result.output_axes == (_TIME_AXIS,)

    def test_cumulative_preserves_output_axes(self):
        """Cumulative does not remove the axis from output_axes."""
        x = graph.array_placeholder("x", axes=(_TIME_AXIS,))
        result = graph.cumulative(x, _TIME_AXIS)
        assert result.output_axes == (_TIME_AXIS,)


class TestAxisOpRepr:
    """Test string representation."""

    def test_shift_repr(self):
        """Shift repr is round-trippable and contains its parameters."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS, periods=2, fill_value=1.5, name="s")
        repr_str = repr(result)
        assert "graph.shift" in repr_str
        assert "periods=2" in repr_str
        assert "fill_value=1.5" in repr_str

    def test_roll_repr(self):
        """Roll repr is round-trippable and contains its parameters."""
        x = graph.placeholder("x")
        result = graph.roll(x, _TIME_AXIS, periods=2, name="r")
        repr_str = repr(result)
        assert "graph.roll" in repr_str
        assert "periods=2" in repr_str

    def test_cumulative_repr(self):
        """Cumulative repr is round-trippable."""
        x = graph.placeholder("x")
        result = graph.cumulative(x, _TIME_AXIS, name="c")
        repr_str = repr(result)
        assert "graph.cumulative" in repr_str


class TestAxisOpExecution:
    """Test execution against direct numpy equivalents."""

    def test_shift_forward_fills_with_zero_by_default(self):
        """shift(1) moves values forward, filling position 0 with fill_value."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS, periods=1)
        plan = linearize.forest(result)
        state = executor.State({x: np.array([1.0, 2.0, 3.0, 4.0])}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[result], [0.0, 1.0, 2.0, 3.0])

    def test_shift_negative_periods(self):
        """shift(-1) moves values backward, filling the last position."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS, periods=-1, fill_value=-9.0)
        plan = linearize.forest(result)
        state = executor.State({x: np.array([1.0, 2.0, 3.0, 4.0])}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[result], [2.0, 3.0, 4.0, -9.0])

    def test_shift_custom_fill_value(self):
        """Shift uses the given fill_value for the exposed position."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS, periods=1, fill_value=-1.0)
        plan = linearize.forest(result)
        state = executor.State({x: np.array([1.0, 2.0, 3.0])}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[result], [-1.0, 1.0, 2.0])

    def test_shift_zero_periods_is_identity(self):
        """shift(0) returns the array unchanged."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS, periods=0)
        plan = linearize.forest(result)
        state = executor.State({x: np.array([1.0, 2.0, 3.0])}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[result], [1.0, 2.0, 3.0])

    def test_roll_wraps_around(self):
        """roll(1) circularly shifts values, wrapping the last to the front."""
        x = graph.placeholder("x")
        result = graph.roll(x, _TIME_AXIS, periods=1)
        plan = linearize.forest(result)
        state = executor.State({x: np.array([1.0, 2.0, 3.0, 4.0])}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[result], [4.0, 1.0, 2.0, 3.0])

    def test_cumulative_matches_numpy_cumsum(self):
        """Cumulative matches np.cumsum along the resolved axis."""
        x = graph.placeholder("x")
        result = graph.cumulative(x, _TIME_AXIS)
        plan = linearize.forest(result)
        x_val = np.array([1.0, 2.0, 3.0, 4.0])
        state = executor.State({x: x_val}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[result], np.cumsum(x_val))

    def test_shift_batched_2d(self):
        """Shift on a (size, T) batched array shifts along the last (time) axis."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS, periods=1)
        plan = linearize.forest(result)
        x_val = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        state = executor.State({x: x_val}, domain_axes=(_TIME_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.array_equal(state.values[result], [[0.0, 1.0, 2.0], [0.0, 4.0, 5.0]])
        assert state.values[result].shape == x_val.shape

    def test_axis_not_in_domain_axes_raises(self):
        """Operating on an axis outside the evaluation's DOMAIN axes raises."""
        other_axis = Axis("space", DOMAIN)
        x = graph.placeholder("x")
        result = graph.shift(x, other_axis)
        plan = linearize.forest(result)
        state = executor.State({x: np.array([1.0, 2.0, 3.0])}, domain_axes=(_TIME_AXIS,))
        with pytest.raises(executor.UnsupportedOperation, match="axis operations"):
            executor.evaluate_nodes(state, *plan)


class TestAxisOpNumpyAst:
    """Test the numpy_ast debug codegen for shift/roll/cumulative."""

    def test_roll_codegen_uses_np_roll(self):
        """Roll renders as a direct np.roll call."""
        x = graph.placeholder("x")
        result = graph.roll(x, _TIME_AXIS, periods=2)
        code = numpy_ast.graph_node_to_numpy_code(result, domain_axes=(_TIME_AXIS,))
        assert "np.roll(" in code
        assert "axis=-1" in code

    def test_cumulative_codegen_uses_np_cumsum(self):
        """Cumulative renders as a direct np.cumsum call."""
        x = graph.placeholder("x")
        result = graph.cumulative(x, _TIME_AXIS)
        code = numpy_ast.graph_node_to_numpy_code(result, domain_axes=(_TIME_AXIS,))
        assert "np.cumsum(" in code
        assert "axis=-1" in code

    def test_shift_codegen_uses_bare_name_helper(self):
        """Shift has no single-call numpy equivalent, so it renders as a bare-name call."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS, periods=1, fill_value=2.0)
        code = numpy_ast.graph_node_to_numpy_code(result, domain_axes=(_TIME_AXIS,))
        assert code.startswith("n") and "= _shift(" in code
        assert "np._shift" not in code
        assert "fill_value=2.0" in code

    def test_shift_helper_zero_periods_is_identity(self):
        """numpy_ast's own _shift helper (used by the debug codegen) is a no-op at periods=0."""
        x = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(numpy_ast._shift(x, 0, axis=-1, fill_value=-1.0), x)

    def test_shift_codegen_is_actually_runnable(self):
        """The generated _shift(...) call, evaluated against numpy_ast's own helper, matches the executor."""
        x = graph.placeholder("x")
        result = graph.shift(x, _TIME_AXIS, periods=1, fill_value=-1.0)
        operand_name = f"n{x.id}"
        code = numpy_ast.graph_node_to_numpy_code(result, domain_axes=(_TIME_AXIS,)).split(" = ", 1)[1]
        evaluated = eval(  # noqa: S307
            code, {"np": np, "_shift": numpy_ast._shift}, {operand_name: np.array([1.0, 2.0, 3.0])}
        )
        assert np.array_equal(evaluated, [-1.0, 1.0, 2.0])
