"""Direct tests for kernels.gradient/kernels.laplacian across all five BoundaryCondition kinds.

test_spatial_operators.py covers gradient/laplacian through the graph.py/
executor.py path (the way a model actually uses them); this file tests the
shared numeric core in kernels.py directly, against plain numpy arrays, to
pin down each boundary kind's exact ghost-point behavior without the
graph-construction overhead.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import Constant, Linear, Nearest, Neumann, Reflect, Wrap
from civic_digital_twins.dt_model.engine.numpybackend import kernels


class TestGradientBoundaryKinds:
    """kernels.gradient at the boundary, one test per BoundaryCondition kind."""

    def test_constant_boundary(self):
        """Constant(u): the ghost point is pinned to u, independent of the field's data."""
        f = np.array([1.0, 2.0, 3.0, 4.0])
        g = kernels.gradient(f, 0, 1.0, Constant(10.0))
        # left ghost = 10.0, right neighbour = f[1] = 2.0 -> g[0] = (2.0 - 10.0) / 2 = -4.0
        assert np.isclose(g[0], -4.0)
        # right ghost = 10.0, left neighbour = f[-2] = 3.0 -> g[-1] = (10.0 - 3.0) / 2 = 3.5
        assert np.isclose(g[-1], 3.5)

    def test_neumann_boundary_matches_declared_flux_on_a_linear_field(self):
        """Neumann(v): on a genuinely linear field, the gradient equals v everywhere, both edges included."""
        v = 0.7
        f = np.array([0.0, v, 2 * v, 3 * v])
        g = kernels.gradient(f, 0, 1.0, Neumann(v))
        assert np.allclose(g, v)

    def test_reflect_is_zero_at_the_edges_on_a_curved_field(self):
        """Reflect (the default "reflecting" boundary, Neumann(0)) makes the edge gradient exactly zero."""
        f = np.array([0.0, 1.0, 4.0, 9.0])
        g = kernels.gradient(f, 0, 0.5, Reflect)
        assert np.isclose(g[0], 0.0)
        assert np.isclose(g[-1], 0.0)

    def test_neumann_needs_at_least_2_points(self):
        """Neumann's ghost formula needs a 'neighbor of neighbor' — undefined for a size-1 axis."""
        f = np.array([5.0])
        with pytest.raises(IndexError):
            kernels.gradient(f, 0, 1.0, Neumann(1.0))

    def test_nearest_boundary(self):
        """Nearest: the ghost point repeats the outermost cell."""
        f = np.array([1.0, 2.0, 3.0, 4.0])
        g = kernels.gradient(f, 0, 1.0, Nearest())
        # left ghost = f[0] = 1.0, right neighbour = f[1] = 2.0 -> g[0] = (2.0 - 1.0) / 2 = 0.5
        assert np.isclose(g[0], 0.5)
        # right ghost = f[-1] = 4.0, left neighbour = f[-2] = 3.0 -> g[-1] = (4.0 - 3.0) / 2 = 0.5
        assert np.isclose(g[-1], 0.5)

    def test_wrap_boundary(self):
        """Wrap: the ghost point comes from the opposite end of the array (periodic)."""
        f = np.array([1.0, 2.0, 3.0, 4.0])
        g = kernels.gradient(f, 0, 1.0, Wrap())
        # left ghost = f[-1] = 4.0, right neighbour = f[1] = 2.0 -> g[0] = (2.0 - 4.0) / 2 = -1.0
        assert np.isclose(g[0], -1.0)
        # right ghost = f[0] = 1.0, left neighbour = f[-2] = 3.0 -> g[-1] = (1.0 - 3.0) / 2 = -1.0
        assert np.isclose(g[-1], -1.0)

    def test_linear_boundary_on_a_linear_field_matches_neumann(self):
        """Linear: extends the local slope, so on a genuinely linear field it agrees with the true derivative."""
        f = np.array([0.0, 0.7, 1.4, 2.1])
        g = kernels.gradient(f, 0, 1.0, Linear())
        assert np.allclose(g, 0.7)

    def test_linear_needs_at_least_2_points(self):
        """Linear's extrapolation needs two real points to define a slope — undefined for a size-1 axis."""
        f = np.array([5.0])
        with pytest.raises(IndexError):
            kernels.gradient(f, 0, 1.0, Linear())


class TestLaplacianBoundaryKinds:
    """kernels.laplacian at the boundary, one test per BoundaryCondition kind."""

    def test_constant_boundary_nonzero_curvature_at_edge(self):
        """Constant(0.0) on a field of constant value 5.0 produces nonzero curvature only at the edges."""
        f = np.array([5.0, 5.0, 5.0, 5.0])
        lap = kernels.laplacian(f, (0,), (1.0,), (Constant(0.0),))
        assert np.isclose(lap[0], -5.0)
        assert np.isclose(lap[1], 0.0)
        assert np.isclose(lap[2], 0.0)
        assert np.isclose(lap[-1], -5.0)

    def test_constant_matching_field_value_is_zero_everywhere(self):
        """Constant(5.0) on a field of constant value 5.0 is indistinguishable from an unbounded constant field."""
        f = np.array([5.0, 5.0, 5.0, 5.0])
        lap = kernels.laplacian(f, (0,), (1.0,), (Constant(5.0),))
        assert np.allclose(lap, 0.0)

    def test_neumann_needs_at_least_2_points(self):
        """Same ≥2-point requirement as gradient's Neumann boundary."""
        f = np.array([5.0])
        with pytest.raises(IndexError):
            kernels.laplacian(f, (0,), (1.0,), (Neumann(1.0),))

    def test_nearest_boundary_matches_a_one_sided_zero_flux_approximation(self):
        """Nearest repeats the edge cell, approximating (not exactly enforcing) zero flux at the border."""
        f = np.array([1.0, 2.0, 4.0, 8.0])
        lap = kernels.laplacian(f, (0,), (1.0,), (Nearest(),))
        # left ghost = f[0] = 1.0 -> (1.0 - 2*1.0 + 2.0) / 1 = 1.0
        assert np.isclose(lap[0], 1.0)

    def test_wrap_boundary_matches_periodic_extension(self):
        """Wrap treats the array as periodic for the second-derivative stencil too."""
        f = np.array([1.0, 2.0, 3.0, 2.0])
        lap = kernels.laplacian(f, (0,), (1.0,), (Wrap(),))
        # Periodic neighbours of index 0 are f[-1]=f[3]=2.0 and f[1]=2.0: (2 - 2*1 + 2) / 1 = 2
        assert np.isclose(lap[0], 2.0)

    def test_linear_boundary_matches_zero_curvature_on_a_linear_field(self):
        """Linear extrapolation makes the curvature at the edge exactly zero for a genuinely linear field."""
        f = np.array([0.0, 1.0, 2.0, 3.0])
        lap = kernels.laplacian(f, (0,), (1.0,), (Linear(),))
        assert np.allclose(lap, 0.0)

    def test_linear_needs_at_least_2_points(self):
        """Same ≥2-point requirement as gradient's Linear boundary."""
        f = np.array([5.0])
        with pytest.raises(IndexError):
            kernels.laplacian(f, (0,), (1.0,), (Linear(),))

    def test_unsupported_boundary_type_raises(self):
        """An object that isn't one of the five BoundaryCondition kinds raises TypeError."""
        f = np.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="unsupported boundary condition"):
            kernels.laplacian(f, (0,), (1.0,), ("reflect",))  # type: ignore[arg-type]
