"""Tests for the SpaceType-gated spatial operators: gradient and laplacian.

gradient is a shape-preserving single-axis AxisOp (tested alongside
shift/roll/cumulative in test_axis_shape_preserving_operators.py, but given
its own file here since it is spatial-specific). laplacian is multi-axis and
does not subclass AxisOp. Both honor a per-axis BoundaryCondition
(Constant/Neumann/Nearest/Wrap/Linear), sharing the ghost-padding logic in
kernels._ghost_pad — see tests/dt_model/engine/numpybackend/test_kernels.py
for direct coverage of that shared primitive across all five boundary kinds.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import DOMAIN, Axis, Neumann, Reflect, Wrap
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor, numpy_ast

_X_AXIS = Axis("x", DOMAIN)
_Y_AXIS = Axis("y", DOMAIN)


class TestGradientCreation:
    """Test creation of gradient nodes."""

    def test_gradient_creation(self):
        """Gradient stores node, axis, spacing, and boundary."""
        f = graph.placeholder("f")
        result = graph.gradient(f, _X_AXIS, spacing=0.5, boundary=Wrap())
        assert result.node is f
        assert result.axis == _X_AXIS
        assert result.spacing == 0.5
        assert isinstance(result.boundary, Wrap)

    def test_gradient_default_spacing_and_boundary(self):
        """Gradient defaults to spacing=1.0 and boundary=Reflect (zero-flux/reflecting)."""
        f = graph.placeholder("f")
        result = graph.gradient(f, _X_AXIS)
        assert result.spacing == 1.0
        assert result.boundary is Reflect

    def test_gradient_preserves_output_axes(self):
        """Gradient does not remove the axis from output_axes."""
        f = graph.array_placeholder("f", axes=(_X_AXIS, _Y_AXIS))
        result = graph.gradient(f, _X_AXIS)
        assert result.output_axes == (_X_AXIS, _Y_AXIS)

    def test_gradient_repr(self):
        """Gradient repr is round-trippable and contains its parameters."""
        f = graph.placeholder("f")
        result = graph.gradient(f, _X_AXIS, spacing=0.25, boundary=Neumann(1.5), name="g")
        repr_str = repr(result)
        assert "graph.gradient" in repr_str
        assert "spacing=0.25" in repr_str
        assert "boundary=Neumann(value=1.5)" in repr_str


class TestGradientExecution:
    """Test execution of gradient, in the interior and at the boundary."""

    def test_gradient_interior_matches_central_difference(self):
        """Away from the boundary, gradient matches the ordinary central-difference formula.

        This is the same formula np.gradient uses in its own interior, but
        gradient no longer delegates to np.gradient (which would use a
        different, one-sided formula at the two edges instead of honoring
        the declared boundary) — see test_gradient_default_boundary_is_zero_at_edges.
        """
        f = graph.placeholder("f")
        result = graph.gradient(f, _X_AXIS, spacing=0.5)
        plan = linearize.forest(result)
        f_val = np.array([0.0, 1.0, 4.0, 9.0])
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS,))
        executor.evaluate_nodes(state, *plan)
        interior_expected = np.gradient(f_val, 0.5)[1:-1]
        assert np.allclose(state.values[result][1:-1], interior_expected)

    def test_gradient_default_boundary_is_zero_at_edges(self):
        """Default boundary (Reflect, zero-flux) makes the edge-point gradient exactly zero.

        This is the fix for the confirmed bug where gradient ignored
        SpaceType.boundary entirely and always used np.gradient's one-sided
        edge formula instead.
        """
        f = graph.placeholder("f")
        result = graph.gradient(f, _X_AXIS, spacing=0.5)
        plan = linearize.forest(result)
        f_val = np.array([0.0, 1.0, 4.0, 9.0])
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.isclose(state.values[result][0], 0.0)
        assert np.isclose(state.values[result][-1], 0.0)

    def test_gradient_neumann_matches_declared_flux_on_a_linear_field(self):
        """On a field with a genuinely constant slope, Neumann(v) gradient equals v everywhere, incl. both edges."""
        f = graph.placeholder("f")
        result = graph.gradient(f, _X_AXIS, spacing=1.0, boundary=Neumann(0.7))
        plan = linearize.forest(result)
        f_val = np.array([0.0, 0.7, 1.4, 2.1])  # slope 0.7, unit spacing
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.allclose(state.values[result], 0.7)

    def test_gradient_2d_resolves_correct_axis(self):
        """Gradient on a 2-D field differentiates along the named axis, not the other one."""
        f = graph.placeholder("f")
        grad_x = graph.gradient(f, _X_AXIS)
        plan = linearize.forest(grad_x)
        # f(x, y) = x^2 + y^2, unit spacing: interior d/dx = 2x, independent of y
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        ys = np.array([0.0, 1.0, 2.0])
        f_val = np.array([[x**2 + y**2 for y in ys] for x in xs])
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS, _Y_AXIS))
        executor.evaluate_nodes(state, *plan)
        expected_interior_rows = np.tile((2 * xs[1:-1]).reshape(-1, 1), (1, 3))
        assert np.allclose(state.values[grad_x][1:-1], expected_interior_rows)

    def test_gradient_axis_not_in_domain_axes_raises(self):
        """Operating on an axis outside the evaluation's DOMAIN axes raises."""
        other_axis = Axis("z", DOMAIN)
        f = graph.placeholder("f")
        result = graph.gradient(f, other_axis)
        plan = linearize.forest(result)
        state = executor.State({f: np.array([1.0, 2.0, 3.0])}, domain_axes=(_X_AXIS,))
        with pytest.raises(executor.UnsupportedOperation, match="axis operations"):
            executor.evaluate_nodes(state, *plan)


class TestLaplacianCreation:
    """Test creation of laplacian nodes."""

    def test_laplacian_creation(self):
        """Laplacian stores node, axes, spacings, and boundaries."""
        f = graph.placeholder("f")
        result = graph.laplacian(f, (_X_AXIS, _Y_AXIS), (1.0, 2.0), (Reflect, Wrap()))
        assert result.node is f
        assert result.axes == (_X_AXIS, _Y_AXIS)
        assert result.spacings == (1.0, 2.0)
        assert result.boundaries[0] is Reflect
        assert isinstance(result.boundaries[1], Wrap)

    def test_laplacian_preserves_output_axes(self):
        """Laplacian does not remove any axis from output_axes."""
        f = graph.array_placeholder("f", axes=(_X_AXIS, _Y_AXIS))
        result = graph.laplacian(f, (_X_AXIS, _Y_AXIS), (1.0, 1.0), (Reflect, Reflect))
        assert result.output_axes == (_X_AXIS, _Y_AXIS)

    def test_laplacian_single_axis(self):
        """Laplacian accepts a single axis (1-D second derivative)."""
        f = graph.placeholder("f")
        result = graph.laplacian(f, (_X_AXIS,), (1.0,), (Reflect,))
        assert result.axes == (_X_AXIS,)

    def test_laplacian_repr(self):
        """Laplacian repr is round-trippable and contains its parameters."""
        f = graph.placeholder("f")
        result = graph.laplacian(f, (_X_AXIS,), (1.0,), (Reflect,), name="lap")
        repr_str = repr(result)
        assert "graph.laplacian" in repr_str
        assert "spacings=(1.0,)" in repr_str
        assert "boundaries=(Neumann(value=0.0),)" in repr_str


class TestLaplacianExecution:
    """Test execution of laplacian against a known analytic result."""

    def test_laplacian_2d_quadratic_interior(self):
        """laplacian(x^2 + y^2) == 4 in the interior (away from boundary effects)."""
        f = graph.placeholder("f")
        result = graph.laplacian(f, (_X_AXIS, _Y_AXIS), (1.0, 1.0), (Reflect, Reflect))
        plan = linearize.forest(result)
        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 1.0, 2.0])
        f_val = np.array([[x**2 + y**2 for y in ys] for x in xs])
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS, _Y_AXIS))
        executor.evaluate_nodes(state, *plan)
        assert np.isclose(state.values[result][1, 1], 4.0)

    def test_laplacian_constant_field_is_zero(self):
        """Laplacian of a constant field is zero everywhere, regardless of boundary."""
        f = graph.placeholder("f")
        result = graph.laplacian(f, (_X_AXIS,), (1.0,), (Wrap(),))
        plan = linearize.forest(result)
        f_val = np.array([5.0, 5.0, 5.0, 5.0])
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.allclose(state.values[result], 0.0)

    def test_laplacian_wrap_boundary_matches_periodic_extension(self):
        """Laplacian with boundary=Wrap() treats the array as periodic."""
        f = graph.placeholder("f")
        result = graph.laplacian(f, (_X_AXIS,), (1.0,), (Wrap(),))
        plan = linearize.forest(result)
        f_val = np.array([1.0, 2.0, 3.0, 2.0])
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS,))
        executor.evaluate_nodes(state, *plan)
        # Periodic neighbours of index 0 are f[-1]=f[3]=2.0 and f[1]=2.0: (2+2-2*1)/1 = 2
        assert np.isclose(state.values[result][0], 2.0)

    def test_laplacian_spacing_scales_result(self):
        """Laplacian scales with 1/spacing**2."""
        f = graph.placeholder("f")
        result_unit = graph.laplacian(f, (_X_AXIS,), (1.0,), (Reflect,))
        result_scaled = graph.laplacian(f, (_X_AXIS,), (2.0,), (Reflect,))
        f_val = np.array([0.0, 1.0, 4.0, 9.0, 16.0])
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS,))
        executor.evaluate_nodes(state, *linearize.forest(result_unit, result_scaled))
        assert np.allclose(state.values[result_unit], state.values[result_scaled] * 4.0)

    def test_axis_not_in_domain_axes_raises(self):
        """Operating on an axis outside the evaluation's DOMAIN axes raises."""
        other_axis = Axis("z", DOMAIN)
        f = graph.placeholder("f")
        result = graph.laplacian(f, (other_axis,), (1.0,), (Reflect,))
        plan = linearize.forest(result)
        state = executor.State({f: np.array([1.0, 2.0, 3.0])}, domain_axes=(_X_AXIS,))
        with pytest.raises(executor.UnsupportedOperation, match="axis operations"):
            executor.evaluate_nodes(state, *plan)


class TestSpatialOperatorNumpyAst:
    """Test the numpy_ast debug codegen for gradient/laplacian."""

    def test_gradient_codegen_uses_bare_name_helper(self):
        """Gradient has no single-call numpy equivalent (it honors a boundary), so it renders as a bare-name call."""
        f = graph.placeholder("f")
        result = graph.gradient(f, _X_AXIS, spacing=0.5, boundary=Neumann(1.5))
        code = numpy_ast.graph_node_to_numpy_code(result, domain_axes=(_X_AXIS,))
        assert "= _gradient(" in code
        assert "np._gradient" not in code
        assert "axis=-1" in code
        assert "spacing=0.5" in code
        assert "boundary=Neumann(value=1.5)" in code

    def test_laplacian_codegen_uses_bare_name_helper(self):
        """Laplacian has no single-call numpy equivalent, so it renders as a bare-name call."""
        f = graph.placeholder("f")
        result = graph.laplacian(f, (_X_AXIS, _Y_AXIS), (1.0, 2.0), (Reflect, Wrap()))
        code = numpy_ast.graph_node_to_numpy_code(result, domain_axes=(_X_AXIS, _Y_AXIS))
        assert "= _laplacian(" in code
        assert "np._laplacian" not in code
        assert "axes=(-2, -1)" in code
        assert "spacings=(1.0, 2.0)" in code
        assert "boundaries=(Neumann(value=0.0), Wrap())" in code

    def test_gradient_codegen_is_actually_runnable(self):
        """The generated _gradient(...) call, evaluated against numpy_ast's own helper, matches the executor."""
        f = graph.placeholder("f")
        result = graph.gradient(f, _X_AXIS, spacing=0.5, boundary=Neumann(0.7))
        operand_name = f"n{f.id}"
        code = numpy_ast.graph_node_to_numpy_code(result, domain_axes=(_X_AXIS,)).split(" = ", 1)[1]
        f_val = np.array([0.0, 1.0, 4.0, 9.0])
        namespace = {"np": np, "_gradient": numpy_ast._gradient, "Neumann": numpy_ast.Neumann}
        evaluated = eval(code, namespace, {operand_name: f_val})  # noqa: S307
        plan = linearize.forest(result)
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.allclose(evaluated, state.values[result])

    def test_laplacian_codegen_is_actually_runnable(self):
        """The generated _laplacian(...) call, evaluated against numpy_ast's own helper, matches the executor."""
        f = graph.placeholder("f")
        result = graph.laplacian(f, (_X_AXIS,), (1.0,), (Reflect,))
        operand_name = f"n{f.id}"
        code = numpy_ast.graph_node_to_numpy_code(result, domain_axes=(_X_AXIS,)).split(" = ", 1)[1]
        f_val = np.array([0.0, 1.0, 4.0, 9.0])
        namespace = {"np": np, "_laplacian": numpy_ast._laplacian, "Neumann": numpy_ast.Neumann}
        evaluated = eval(code, namespace, {operand_name: f_val})  # noqa: S307
        plan = linearize.forest(result)
        state = executor.State({f: f_val}, domain_axes=(_X_AXIS,))
        executor.evaluate_nodes(state, *plan)
        assert np.allclose(evaluated, state.values[result])
