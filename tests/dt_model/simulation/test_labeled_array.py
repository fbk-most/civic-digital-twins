"""Tests for simulation/labeled_array.py: LabeledArray."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import DOMAIN, PARAMETER, Axis
from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout
from civic_digital_twins.dt_model.simulation.labeled_array import LabeledArray

P = Axis("p", PARAMETER)
X = Axis("x", DOMAIN)
Y = Axis("y", DOMAIN)


class TestConstruction:
    """LabeledArray construction and basic accessors."""

    def test_dims_and_values(self):
        """Dims reports axis names in layout order; values is the array unchanged."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        assert la.dims == ("x", "y")
        assert la.values is arr

    def test_shape_mismatch_rejected(self):
        """The array's shape must match the layout's full_shape."""
        with pytest.raises(ValueError, match=r"does not match"):
            LabeledArray(np.zeros((2, 4)), AxisLayout([(X, 2), (Y, 3)]))

    def test_repr(self):
        """Repr shows dims and shape, not the raw values."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (Y, 3)]))
        assert repr(la) == "LabeledArray(dims=('x', 'y'), shape=(2, 3))"


class TestSel:
    """Name-based selection along axes."""

    def test_integer_selection_drops_the_axis(self):
        """An int selection fixes that axis and removes it from the result's dims."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        sel = la.sel(x=1)
        assert sel.dims == ("y",)
        np.testing.assert_array_equal(sel.values, arr[1, :])

    def test_slice_selection_keeps_the_axis(self):
        """A slice selection keeps the axis, at its reduced size."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        sel = la.sel(y=slice(0, 2))
        assert sel.dims == ("x", "y")
        assert sel.values.shape == (2, 2)
        np.testing.assert_array_equal(sel.values, arr[:, 0:2])

    def test_multiple_selections(self):
        """Several axes can be selected in one call."""
        arr = np.arange(24.0).reshape(2, 3, 4)
        la = LabeledArray(arr, AxisLayout([(P, 2), (X, 3), (Y, 4)]))
        sel = la.sel(p=0, y=2)
        assert sel.dims == ("x",)
        np.testing.assert_array_equal(sel.values, arr[0, :, 2])

    def test_combined_int_and_slice_selection(self):
        """An int selection (dropped) and a slice selection (kept) combine in one call."""
        arr = np.arange(24.0).reshape(2, 3, 4)
        la = LabeledArray(arr, AxisLayout([(P, 2), (X, 3), (Y, 4)]))
        sel = la.sel(p=1, y=slice(1, 3))
        assert sel.dims == ("x", "y")
        assert sel.values.shape == (3, 2)
        np.testing.assert_array_equal(sel.values, arr[1, :, 1:3])

    def test_unknown_axis_name_raises(self):
        """Selecting an axis not in the layout raises KeyError naming the valid ones."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (Y, 3)]))
        with pytest.raises(KeyError, match=r"no axis named 'z'"):
            la.sel(z=0)

    def test_sel_returns_a_new_labeled_array(self):
        """sel() does not mutate the original."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        la.sel(x=0)
        assert la.dims == ("x", "y")
        assert la.values.shape == (2, 3)


class TestToXarray:
    """Optional xarray interop."""

    def test_missing_xarray_raises_import_error(self, monkeypatch):
        """Without xarray installed, to_xarray() raises a clear ImportError."""
        import builtins

        real_import = builtins.__import__

        def _blocked_import(name, *args, **kwargs):
            if name == "xarray":
                raise ImportError("No module named 'xarray'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked_import)
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (Y, 3)]))
        with pytest.raises(ImportError, match="optional 'xarray' package"):
            la.to_xarray()

    def test_to_xarray_round_trips_values_and_dims(self):
        """When xarray is installed, to_xarray() produces a matching DataArray."""
        xr = pytest.importorskip("xarray")
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        da = la.to_xarray()
        assert isinstance(da, xr.DataArray)
        assert da.dims == ("x", "y")
        np.testing.assert_array_equal(da.values, arr)
