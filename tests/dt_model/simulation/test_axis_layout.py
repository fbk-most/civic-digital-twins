"""Tests for simulation/axis_layout.py: AxisLayout and its leading-array primitives."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import DOMAIN, ENSEMBLE, PARAMETER, TIME_AXIS, Axis
from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout

P1 = Axis("p1", PARAMETER)
P2 = Axis("p2", PARAMETER)
E1 = Axis("e1", ENSEMBLE)
E2 = Axis("e2", ENSEMBLE)


class TestConstruction:
    """AxisLayout construction and validation."""

    def test_entries_positions_sizes(self):
        """Entries preserve order; positions are implicit indices."""
        layout = AxisLayout([(P1, 2), (E1, 3), (TIME_AXIS, 5)])
        assert layout.entries == ((P1, 2), (E1, 3), (TIME_AXIS, 5))
        assert layout.axes == (P1, E1, TIME_AXIS)
        assert len(layout) == 3
        assert layout.positions == {P1: 0, E1: 1, TIME_AXIS: 2}
        assert layout.sizes == {P1: 2, E1: 3, TIME_AXIS: 5}

    def test_empty(self):
        """An empty layout is valid (deterministic scalar evaluation)."""
        layout = AxisLayout([])
        assert layout.entries == ()
        assert layout.full_shape == ()
        assert layout.leading_size == 1

    def test_duplicate_names_rejected(self):
        """Axis names are globally unique within a layout, across roles."""
        with pytest.raises(ValueError, match="duplicate axis names"):
            AxisLayout([(Axis("x", PARAMETER), 2), (Axis("x", ENSEMBLE), 3)])

    def test_unknown_role_rejected(self):
        """Only PARAMETER, ENSEMBLE, and DOMAIN roles are supported."""
        with pytest.raises(ValueError, match="unsupported role"):
            AxisLayout([(Axis("x", "SPATIAL"), 2)])

    def test_role_ordering_enforced(self):
        """Axes must be grouped PARAMETER, then ENSEMBLE, then DOMAIN."""
        with pytest.raises(ValueError, match="canonical order"):
            AxisLayout([(E1, 3), (P1, 2)])
        with pytest.raises(ValueError, match="canonical order"):
            AxisLayout([(TIME_AXIS, 5), (E1, 3)])

    def test_nonpositive_size_rejected(self):
        """Sizes must be positive integers."""
        with pytest.raises(ValueError, match="positive"):
            AxisLayout([(P1, 0)])

    def test_build_groups(self):
        """build() concatenates the role groups in canonical order."""
        layout = AxisLayout.build(parameters=[(P1, 2), (P2, 4)], ensemble=[(E1, 3)], domain=[(TIME_AXIS, 5)])
        assert layout == AxisLayout([(P1, 2), (P2, 4), (E1, 3), (TIME_AXIS, 5)])

    def test_from_positions(self):
        """from_positions() orders entries by position."""
        layout = AxisLayout.from_positions({E1: 1, P1: 0}, {P1: 2, E1: 3})
        assert layout.entries == ((P1, 2), (E1, 3))

    def test_from_positions_noncontiguous_rejected(self):
        """Positions must be exactly 0..n-1."""
        with pytest.raises(ValueError, match="contiguous"):
            AxisLayout.from_positions({P1: 0, E1: 2}, {P1: 2, E1: 3})


class TestQueries:
    """Lookup and role-query methods."""

    def test_contains_and_lookup(self):
        """position_of/size_of work by value equality; KeyError when absent."""
        layout = AxisLayout([(P1, 2), (E1, 3)])
        assert Axis("p1", PARAMETER) in layout
        assert TIME_AXIS not in layout
        assert layout.position_of(Axis("e1", ENSEMBLE)) == 1
        assert layout.size_of(Axis("e1", ENSEMBLE)) == 3
        with pytest.raises(KeyError):
            layout.position_of(TIME_AXIS)

    def test_find_axis(self):
        """find_axis matches by name, optionally constrained by role."""
        layout = AxisLayout([(P1, 2), (E1, 3)])
        assert layout.find_axis("e1") is layout.axes[1]
        assert layout.find_axis("e1", ENSEMBLE) is layout.axes[1]
        assert layout.find_axis("e1", PARAMETER) is None
        assert layout.find_axis("missing") is None

    def test_axes_by_role(self):
        """axes_by_role returns (axis, position) pairs in position order."""
        layout = AxisLayout([(P1, 2), (P2, 4), (E1, 3), (TIME_AXIS, 5)])
        assert layout.axes_by_role(PARAMETER) == ((P1, 0), (P2, 1))
        assert layout.axes_by_role(ENSEMBLE) == ((E1, 2),)
        assert layout.axes_by_role(DOMAIN) == ((TIME_AXIS, 3),)


class TestDerivedShapes:
    """Counts, shapes, and compatibility checks."""

    def test_counts_and_shapes(self):
        """Role counts and derived shapes reflect the layout."""
        layout = AxisLayout([(P1, 2), (P2, 4), (E1, 3), (TIME_AXIS, 5)])
        assert (layout.n_params, layout.n_ensemble, layout.n_domain) == (2, 1, 1)
        assert layout.n_leading == 3
        assert layout.full_shape == (2, 4, 3, 5)
        assert layout.leading_axes == (P1, P2, E1)
        assert layout.leading_shape == (2, 4, 3)
        assert layout.leading_size == 24

    def test_compatible_with(self):
        """Shapes must have one dim per axis, each 1 or the axis size."""
        layout = AxisLayout([(P1, 2), (E1, 3)])
        assert layout.compatible_with((2, 3))
        assert layout.compatible_with((1, 3))
        assert not layout.compatible_with((2,))
        assert not layout.compatible_with((2, 4))


class TestSignatures:
    """Merge-compatibility signatures."""

    def test_role_signature(self):
        """role_signature yields (name, position, size) triples, with exclusion."""
        layout = AxisLayout([(P1, 2), (E1, 3), (E2, 4)])
        assert layout.role_signature(ENSEMBLE) == frozenset({("e1", 1, 3), ("e2", 2, 4)})
        assert layout.role_signature(ENSEMBLE, exclude_name="e1") == frozenset({("e2", 2, 4)})

    def test_parameter_signature_includes_domain(self):
        """parameter_signature covers all non-ENSEMBLE axes with their role."""
        layout = AxisLayout([(P1, 2), (E1, 3), (TIME_AXIS, 5)])
        assert layout.parameter_signature() == frozenset(
            {("p1", PARAMETER, 0, 2), ("time", DOMAIN, 2, 5)},
        )


class TestTransformations:
    """with_grown_axis and with_axis_appended."""

    def test_with_grown_axis(self):
        """Growing an axis changes only its size."""
        layout = AxisLayout([(P1, 2), (E1, 3)])
        grown = layout.with_grown_axis("e1", 8)
        assert grown.entries == ((P1, 2), (E1, 8))
        assert layout.entries == ((P1, 2), (E1, 3))  # immutable

    def test_with_grown_axis_missing(self):
        """Growing an unknown axis raises KeyError."""
        with pytest.raises(KeyError, match="missing"):
            AxisLayout([(P1, 2)]).with_grown_axis("missing", 8)

    def test_with_axis_appended(self):
        """Appending registers the DOMAIN time axis as the last dimension."""
        layout = AxisLayout([(P1, 2), (E1, 3)]).with_axis_appended(TIME_AXIS, 5)
        assert layout.entries == ((P1, 2), (E1, 3), (TIME_AXIS, 5))

    def test_with_axis_appended_revalidates_order(self):
        """Appending re-checks the canonical role ordering."""
        with pytest.raises(ValueError, match="canonical order"):
            AxisLayout([(E1, 3)]).with_axis_appended(P1, 2)


class TestArrayOperations:
    """contract_ensemble and drop_stray_domain."""

    def test_contract_ensemble_weighted_average(self):
        """A non-singleton ENSEMBLE dim is contracted by weighted average."""
        layout = AxisLayout([(P1, 2), (E1, 3)])
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = layout.contract_ensemble(arr, {E1: np.array([1.0, 1.0, 2.0])})
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [(1 + 2 + 6) / 4, (4 + 5 + 12) / 4])

    def test_contract_ensemble_singleton_squeeze(self):
        """A singleton ENSEMBLE dim is squeezed without consulting weights."""
        layout = AxisLayout([(P1, 2), (E1, 3)])
        arr = np.ones((2, 1))
        out = layout.contract_ensemble(arr, {})
        assert out.shape == (2,)

    def test_contract_ensemble_incompatible_shape_rejected(self):
        """A rank or size mismatch is reported instead of contracting a wrong dim."""
        layout = AxisLayout([(P1, 2), (E1, 3)])
        with pytest.raises(ValueError, match="incompatible with layout"):
            layout.contract_ensemble(np.ones((2,)), {})

    def test_contract_ensemble_multiple_axes(self):
        """Multiple ENSEMBLE dims contract in descending position order."""
        layout = AxisLayout([(E1, 2), (E2, 3)])
        arr = np.arange(6.0).reshape(2, 3)
        out = layout.contract_ensemble(arr, {E1: np.ones(2), E2: np.ones(3)})
        assert out.shape == ()
        np.testing.assert_allclose(out, arr.mean())

    def test_drop_stray_domain(self):
        """Size-1 DOMAIN dims not carried by the index are squeezed away."""
        layout = AxisLayout([(P1, 2), (TIME_AXIS, 5)])
        arr = np.ones((2, 1))
        assert layout.drop_stray_domain(arr, ()).shape == (2,)

    def test_drop_stray_domain_keeps_carried(self):
        """DOMAIN dims carried by the index are preserved at any size."""
        layout = AxisLayout([(P1, 2), (TIME_AXIS, 5)])
        assert layout.drop_stray_domain(np.ones((2, 1)), (TIME_AXIS,)).shape == (2, 1)
        assert layout.drop_stray_domain(np.ones((2, 5)), (TIME_AXIS,)).shape == (2, 5)

    def test_drop_stray_domain_requires_contracted_array(self):
        """A full-layout (non-contracted) array is rejected up front."""
        layout = AxisLayout([(P1, 2), (E1, 3), (TIME_AXIS, 5)])
        with pytest.raises(ValueError, match="not ENSEMBLE-contracted"):
            layout.drop_stray_domain(np.ones((2, 3, 5)), ())

    def test_drop_stray_domain_undeclared_variation_rejected(self):
        """A non-carried DOMAIN dim of size > 1 is a contract violation."""
        layout = AxisLayout([(P1, 2), (TIME_AXIS, 5)])
        with pytest.raises(ValueError, match="declared and actual axes disagree"):
            layout.drop_stray_domain(np.ones((2, 5)), ())


class TestSerialization:
    """to_dict/from_dict round-trip in the runner persistence format."""

    def test_to_dict_format(self):
        """to_dict emits [name, role, pos] rows and 'name:role' size keys."""
        layout = AxisLayout([(P1, 2), (E1, 3)])
        assert layout.to_dict() == {
            "axis_layout": [["p1", PARAMETER, 0], ["e1", ENSEMBLE, 1]],
            "axis_sizes": {f"p1:{PARAMETER}": 2, f"e1:{ENSEMBLE}": 3},
        }

    def test_round_trip(self):
        """from_dict(to_dict()) reproduces the layout."""
        layout = AxisLayout([(P1, 2), (E1, 3), (TIME_AXIS, 5)])
        assert AxisLayout.from_dict(layout.to_dict()) == layout


class TestDunder:
    """Equality, hashing, and repr."""

    def test_eq_and_hash(self):
        """Layouts compare by ordered entries; unequal sizes differ."""
        a = AxisLayout([(P1, 2), (E1, 3)])
        b = AxisLayout([(Axis("p1", PARAMETER), 2), (Axis("e1", ENSEMBLE), 3)])
        assert a == b
        assert hash(a) == hash(b)
        assert a != AxisLayout([(P1, 2), (E1, 4)])
        assert a != "not a layout"

    def test_repr(self):
        """The repr lists entries in order."""
        assert "Axis('p1'" in repr(AxisLayout([(P1, 2)]))


class TestLeadingPrimitives:
    """Pure leading-array shape arithmetic."""

    LEAD = AxisLayout([(P1, 2), (E1, 3)])

    def test_spans_leading(self):
        """A spanning shape has >= n_leading dims, each 1 or the axis size."""
        assert self.LEAD.spans_leading((2, 3))
        assert self.LEAD.spans_leading((1, 3))
        assert self.LEAD.spans_leading((2, 3, 7))  # trailing dims allowed
        assert not self.LEAD.spans_leading((2,))  # too few dims
        assert not self.LEAD.spans_leading((4, 3))  # wrong size

    def test_prepend_leading(self):
        """prepend_leading adds one singleton per leading axis."""
        assert self.LEAD.prepend_leading(np.ones((7,))).shape == (1, 1, 7)
        assert self.LEAD.prepend_leading(np.asarray(5.0)).shape == (1, 1)

    def test_broadcast_leading(self):
        """Singleton leading dims expand to the layout; trailing dims survive."""
        out = self.LEAD.broadcast_leading(np.ones((1, 3, 4)))
        assert out.shape == (2, 3, 4)

    def test_broadcast_leading_requires_spanning_shape(self):
        """A shape that does not span the leading layout is rejected."""
        with pytest.raises(ValueError, match="does not span the"):
            self.LEAD.broadcast_leading(np.ones((4, 3)))
        with pytest.raises(ValueError, match="does not span the"):
            self.LEAD.broadcast_leading(np.ones((3,)))

    def test_flatten_unflatten_roundtrip(self):
        """unflatten_leading inverts flatten_leading."""
        arr = np.arange(24.0).reshape(2, 3, 4)
        flat = self.LEAD.flatten_leading(arr)
        assert flat.shape == (6, 4)
        np.testing.assert_array_equal(self.LEAD.unflatten_leading(flat), arr)

    def test_flatten_leading_requires_full_leading_dims(self):
        """A shape whose element count merely coincides is rejected, not scrambled."""
        with pytest.raises(ValueError, match="does not start with the"):
            self.LEAD.flatten_leading(np.ones((3, 2)))  # transposed leading dims

    def test_unflatten_leading_requires_leading_size(self):
        """The first dimension must be exactly leading_size."""
        with pytest.raises(ValueError, match="does not start with"):
            self.LEAD.unflatten_leading(np.ones((5,)))
        with pytest.raises(ValueError, match="does not start with"):
            self.LEAD.unflatten_leading(np.asarray(1.0))

    def test_take_leading(self):
        """take_leading gathers rows of the flattened leading layout."""
        arr = np.arange(6.0).reshape(2, 3)
        np.testing.assert_array_equal(self.LEAD.take_leading(arr, np.array([0, 4])), [0.0, 4.0])

    def test_empty_layout(self):
        """With no leading axes the primitives degenerate gracefully."""
        empty = AxisLayout([])
        assert empty.spans_leading(())
        arr = np.float64(7.0)
        assert empty.prepend_leading(np.asarray(arr)).shape == ()
        assert empty.flatten_leading(np.asarray(arr)).shape == (1,)
