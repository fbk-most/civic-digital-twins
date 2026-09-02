"""Tests for simulation/region_execution.py: RegionArrayOps."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import ENSEMBLE, PARAMETER, TIME_AXIS, Axis
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout
from civic_digital_twins.dt_model.simulation.region_execution import RegionArrayOps

P1 = Axis("p1", PARAMETER)
E1 = Axis("e1", ENSEMBLE)

LEAD = RegionArrayOps(AxisLayout([(P1, 2), (E1, 3)]), domain_axes=())
LEAD_TS = RegionArrayOps(AxisLayout([(P1, 2), (E1, 3)]), domain_axes=(TIME_AXIS,))
SCALAR = RegionArrayOps(AxisLayout([]), domain_axes=())


def _selector() -> graph.Node:
    """Build a minimal variant_selector sentinel node."""
    return graph.variant_selector(graph.placeholder("sel"), {"a": []}, [])


class TestSelectorMask:
    """selector_mask over the leading layout."""

    def test_scalar_layout_scalar_selector(self):
        """With no leading axes a scalar selector yields a 0-d mask."""
        mask = SCALAR.selector_mask(graph.constant("a"), np.asarray("a"), "a")
        assert mask.shape == () and bool(mask)

    def test_scalar_layout_singleton_selector_normalised(self):
        """A (1,) selector under an empty layout is normalised to 0-d."""
        mask = SCALAR.selector_mask(graph.constant("a"), np.asarray(["a"]), "b")
        assert mask.shape == () and not bool(mask)

    def test_scalar_layout_wide_selector_rejected(self):
        """A multi-coordinate selector under an empty layout is unsupported."""
        with pytest.raises(NotImplementedError, match="non-singleton DOMAIN"):
            SCALAR.selector_mask(graph.constant("a"), np.asarray(["a", "b"]), "a")

    def test_mask_broadcasts_over_leading(self):
        """The mask spans the full leading shape."""
        sel = np.array([["a"], ["b"]], dtype=object)  # (P, 1)
        mask = LEAD.selector_mask(graph.constant("x"), sel, "a")
        assert mask.shape == (2, 3)
        assert mask[0].all() and not mask[1].any()

    def test_trailing_singleton_squeezed(self):
        """A trailing singleton (timeseries) dim on the selector is dropped."""
        node = graph.timeseries_constant([1.0])
        sel = np.full((2, 3, 1), "a", dtype=object)
        mask = LEAD.selector_mask(node, sel, "a")
        assert mask.shape == (2, 3) and mask.all()

    def test_trailing_wide_rejected(self):
        """A selector varying along a DOMAIN axis is unsupported."""
        node = graph.timeseries_constant([1.0, 2.0])
        sel = np.full((2, 3, 2), "a", dtype=object)
        with pytest.raises(NotImplementedError, match="non-singleton DOMAIN"):
            LEAD.selector_mask(node, sel, "a")


class TestGather:
    """gather into a branch-local first axis."""

    def test_passthrough(self):
        """Empty layouts and selector sentinels bypass the gather."""
        assert SCALAR.gather(graph.constant(1.0), 7.0, np.array([0])).shape == ()
        assert LEAD.gather(_selector(), np.array(["a"]), np.array([0])).shape == (1,)

    def test_gather_selects_flat_coordinates(self):
        """Gather flattens the leading layout and takes the given rows."""
        arr = np.arange(6.0).reshape(2, 3)
        np.testing.assert_array_equal(LEAD.gather(graph.constant(1.0), arr, np.array([0, 4])), [0.0, 4.0])

    def test_gather_aligns_scalar_value(self):
        """A 0-d value gains leading dims and is broadcast before the gather."""
        out = LEAD.gather(graph.constant(1.0), 5.0, np.array([1, 2]))
        np.testing.assert_array_equal(out, [5.0, 5.0])

    def test_gather_aligns_domain_only_value(self):
        """A raw (T,) timeseries value is recognised via output_axes and aligned."""
        node = graph.timeseries_constant([1.0, 2.0, 3.0])
        out = LEAD.gather(node, np.array([1.0, 2.0, 3.0]), np.array([0, 5]))
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(out[1], [1.0, 2.0, 3.0])

    def test_gather_aligns_incompatible_dims(self):
        """Dims matching neither 1 nor the axis size trigger a leading prepend."""
        out = LEAD.gather(graph.constant(1.0), np.ones((4, 7)), np.array([0, 3]))
        assert out.shape == (2, 4, 7)


class TestScatter:
    """scatter branch-local values back into the full leading layout."""

    def test_passthrough(self):
        """Empty layouts and selector sentinels bypass the scatter."""
        assert SCALAR.scatter(graph.constant(1.0), 7.0, np.array([0])).shape == ()
        assert LEAD.scatter(_selector(), np.array(["a"]), np.array([0])).shape == (1,)

    def test_scatter_roundtrip_with_nan_fill(self):
        """Scattered floats land at their coordinates; the rest is NaN."""
        idx = np.array([0, 4])
        out = LEAD.scatter(graph.constant(1.0), np.array([1.0, 2.0]), idx)
        assert out.shape == (2, 3)
        flat = out.reshape(-1)
        np.testing.assert_array_equal(flat[idx], [1.0, 2.0])
        assert np.isnan(np.delete(flat, idx)).all()

    def test_scatter_fill_by_dtype(self):
        """Booleans fill with False, integers with 0, objects with None."""
        idx = np.array([1])
        out_b = LEAD.scatter(graph.constant(True), np.array([True]), idx)
        assert out_b.dtype.kind == "b" and out_b.reshape(-1)[1] and not out_b.reshape(-1)[0]
        out_i = LEAD.scatter(graph.constant(1), np.array([9]), idx)
        assert out_i.dtype.kind == "i" and out_i.reshape(-1)[0] == 0
        out_o = LEAD.scatter(graph.constant("x"), np.array(["v"], dtype=object), idx)
        assert out_o.dtype == object and out_o.reshape(-1)[0] is None

    def test_scatter_broadcasts_scalar(self):
        """A 0-d branch value is broadcast across the selected coordinates."""
        out = LEAD.scatter(graph.constant(1.0), np.float64(9.0), np.array([1, 2]))
        assert (out.reshape(-1)[[1, 2]] == 9.0).all()

    def test_scatter_broadcasts_singleton_first_dim(self):
        """A (1, ...) branch value is broadcast across k selected coordinates."""
        out = LEAD.scatter(graph.constant(1.0), np.array([5.0]), np.array([0, 3]))
        assert (out.reshape(-1)[[0, 3]] == 5.0).all()

    def test_scatter_broadcasts_domain_only_value(self):
        """A (T,) value from a DOMAIN node is repeated per coordinate."""
        node = graph.timeseries_constant([1.0, 2.0, 3.0])
        out = LEAD_TS.scatter(node, np.array([1.0, 2.0, 3.0]), np.array([0, 5]))
        assert out.shape == (2, 3, 3)
        np.testing.assert_array_equal(out.reshape(-1, 3)[5], [1.0, 2.0, 3.0])

    def test_scatter_size_mismatch_rejected(self):
        """A non-DOMAIN value with a wrong first dimension is an error."""
        with pytest.raises(ValueError, match="does not match selected size"):
            LEAD.scatter(graph.constant(1.0), np.ones(3), np.array([0, 1]))

    def test_scatter_pads_scalar_values_with_timeseries(self):
        """With timeseries, per-coordinate scalars gain a trailing singleton."""
        out = LEAD_TS.scatter(graph.constant(1.0), np.array([1.0, 2.0]), np.array([0, 1]))
        assert out.shape == (2, 3, 1)


class TestEmptyBranchValue:
    """empty_branch_value placeholders."""

    def test_shape_without_timeseries(self):
        """The placeholder spans the leading shape."""
        out = LEAD.empty_branch_value()
        assert out.shape == (2, 3) and np.isnan(out).all()

    def test_shape_with_timeseries(self):
        """With timeseries a trailing singleton is appended."""
        assert LEAD_TS.empty_branch_value().shape == (2, 3, 1)
