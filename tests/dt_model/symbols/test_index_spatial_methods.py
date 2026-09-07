"""Tests for GenericIndex.gradient/laplacian.

Mirrors tests/dt_model/symbols/test_index_axis_methods.py, covering the
SpaceType-gated spatial operators.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import TIME_AXIS, DomainAxis, SpaceType
from civic_digital_twins.dt_model.engine.frontend import graph, linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.model.index import Index

_X_AXIS = DomainAxis("x", type=SpaceType(spacing=1.0))
_Y_AXIS = DomainAxis("y", type=SpaceType(spacing=2.0, boundary="wrap"))


class TestIndexGradientCreation:
    """Test that gradient() returns correctly configured graph.gradient nodes."""

    def test_gradient_returns_gradient_node(self):
        """gradient() returns a graph.gradient node reading the axis's spacing."""
        placeholder = graph.array_placeholder("field", axes=(_X_AXIS,))
        idx = Index("field", placeholder)
        result = idx.gradient()
        assert isinstance(result, graph.gradient)
        assert result.axis == _X_AXIS
        assert result.spacing == 1.0

    def test_gradient_reads_spacing_from_axis_type(self):
        """gradient() bakes the resolved axis's SpaceType.spacing onto the node."""
        placeholder = graph.array_placeholder("field", axes=(_Y_AXIS,))
        idx = Index("field", placeholder)
        result = idx.gradient()
        assert isinstance(result, graph.gradient)
        assert result.spacing == 2.0

    def test_gradient_requires_space_type(self):
        """gradient() raises ValueError on a non-SpaceType axis."""
        placeholder = graph.array_placeholder("ts", axes=(TIME_AXIS,))
        idx = Index("ts", placeholder)
        with pytest.raises(ValueError, match="require a SpaceType axis"):
            idx.gradient()

    def test_gradient_requires_explicit_axis_when_multiple(self):
        """gradient() raises ValueError without axis= when the index carries several DOMAIN axes."""
        placeholder = graph.array_placeholder("field", axes=(_X_AXIS, _Y_AXIS))
        idx = Index("field", placeholder)
        with pytest.raises(ValueError, match="needs an explicit axis="):
            idx.gradient()

    def test_gradient_with_explicit_axis(self):
        """gradient(axis=...) resolves correctly for a multi-axis index."""
        placeholder = graph.array_placeholder("field", axes=(_X_AXIS, _Y_AXIS))
        idx = Index("field", placeholder)
        result = idx.gradient(axis=_Y_AXIS)
        assert isinstance(result, graph.gradient)
        assert result.axis == _Y_AXIS
        assert result.spacing == 2.0


class TestIndexLaplacianCreation:
    """Test that laplacian() returns correctly configured graph.laplacian nodes."""

    def test_laplacian_defaults_to_all_space_type_axes(self):
        """laplacian() with no axes= defaults to every SpaceType DOMAIN axis, spacing/boundary read off each."""
        placeholder = graph.array_placeholder("field", axes=(_X_AXIS, _Y_AXIS))
        idx = Index("field", placeholder)
        result = idx.laplacian()
        assert isinstance(result, graph.laplacian)
        assert result.axes == (_X_AXIS, _Y_AXIS)
        assert result.spacings == (1.0, 2.0)
        assert result.boundaries == ("reflect", "wrap")

    def test_laplacian_explicit_axes_subset(self):
        """laplacian(axes=...) restricts to the given axes."""
        placeholder = graph.array_placeholder("field", axes=(_X_AXIS, _Y_AXIS))
        idx = Index("field", placeholder)
        result = idx.laplacian(axes=(_X_AXIS,))
        assert isinstance(result, graph.laplacian)
        assert result.axes == (_X_AXIS,)
        assert result.spacings == (1.0,)

    def test_laplacian_no_space_type_axis_raises(self):
        """laplacian() raises ValueError when the index carries no SpaceType DOMAIN axis."""
        placeholder = graph.array_placeholder("ts", axes=(TIME_AXIS,))
        idx = Index("ts", placeholder)
        with pytest.raises(ValueError, match="needs an explicit axes="):
            idx.laplacian()

    def test_laplacian_explicit_axes_requires_space_type(self):
        """laplacian(axes=...) still validates each given axis carries SpaceType."""
        placeholder = graph.array_placeholder("field", axes=(_X_AXIS, TIME_AXIS))
        idx = Index("field", placeholder)
        with pytest.raises(ValueError, match="require a SpaceType axis"):
            idx.laplacian(axes=(_X_AXIS, TIME_AXIS))


class TestIndexSpatialMethodExecution:
    """Test execution of gradient()/laplacian() against a known analytic result."""

    def test_laplacian_evaluation_interior(self):
        """laplacian() of x^2 + y^2 equals 4 in the interior, for unit-spacing axes."""
        x_axis = DomainAxis("x", type=SpaceType(spacing=1.0))
        y_axis = DomainAxis("y", type=SpaceType(spacing=1.0))
        placeholder = graph.array_placeholder("field", axes=(x_axis, y_axis))
        idx = Index("field", placeholder)
        node = idx.laplacian()
        plan = linearize.forest(node)

        xs = np.array([0.0, 1.0, 2.0])
        ys = np.array([0.0, 1.0, 2.0])
        f_val = np.array([[x**2 + y**2 for y in ys] for x in xs])
        state = executor.State({placeholder: f_val}, domain_axes=(x_axis, y_axis))
        executor.evaluate_nodes(state, *plan)
        assert np.isclose(state.values[node][1, 1], 4.0)
