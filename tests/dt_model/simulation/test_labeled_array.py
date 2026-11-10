"""Tests for simulation/labeled_array.py: LabeledArray."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model.axes import DOMAIN, PARAMETER, Axis, DomainAxis, SetType
from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout
from civic_digital_twins.dt_model.simulation.labeled_array import LabeledArray

P = Axis("p", PARAMETER)
X = Axis("x", DOMAIN)
Y = Axis("y", DOMAIN)
ZONE = DomainAxis("zone", type=SetType(labels=("centre", "north", "south")))


class TestConstruction:
    """LabeledArray construction and basic accessors."""

    def test_axes_and_values(self):
        """Axes reports the axes in layout order; values is the array unchanged."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        assert la.axes == (X, Y)
        assert la.values is arr

    def test_dims_is_deprecated(self):
        """Dims still gives the axis names, with a DeprecationWarning pointing to axes."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (Y, 3)]))
        with pytest.warns(DeprecationWarning, match=r"LabeledArray.dims is deprecated.*Use LabeledArray.axes instead"):
            assert la.dims == ("x", "y")

    def test_sizes_and_layout(self):
        """Sizes are the dimension sizes; layout gives them back as an AxisLayout."""
        layout = AxisLayout([(X, 2), (Y, 3)])
        la = LabeledArray(np.zeros((2, 3)), layout)
        assert la.sizes == (2, 3)
        assert la.layout == layout

    def test_pairs_in_any_role_order(self):
        """(axis, size) pairs may interleave roles, which an AxisLayout does not allow."""
        la = LabeledArray(np.zeros((3, 2)), [(X, 3), (P, 2)])
        assert la.axes == (X, P)
        with pytest.raises(ValueError, match=r"LabeledArray.layout: the axes \['x', 'p'\] are not grouped by role"):
            _ = la.layout

    def test_duplicate_axis_names_rejected(self):
        """Axis names identify dimensions, so they must be distinct."""
        with pytest.raises(ValueError, match=r"duplicate axis names in \['x', 'x'\]"):
            LabeledArray(np.zeros((2, 2)), [(X, 2), (Axis("x", PARAMETER), 2)])

    def test_shape_mismatch_rejected(self):
        """The array's shape must match the layout's full_shape."""
        with pytest.raises(ValueError, match=r"does not match"):
            LabeledArray(np.zeros((2, 4)), AxisLayout([(X, 2), (Y, 3)]))

    def test_repr(self):
        """Repr shows the axis names and shape, not the raw values."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (Y, 3)]))
        assert repr(la) == "LabeledArray(axes=('x', 'y'), shape=(2, 3))"


class TestIsel:
    """Selection by position along axes named by keyword."""

    def test_integer_selection_drops_the_axis(self):
        """An int selection fixes that axis and removes it from the result's axes."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        sel = la.isel(x=1)
        assert sel.axes == (Y,)
        np.testing.assert_array_equal(sel.values, arr[1, :])

    def test_slice_selection_keeps_the_axis(self):
        """A slice selection keeps the axis, at its reduced size."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        sel = la.isel(y=slice(0, 2))
        assert sel.axes == (X, Y)
        assert sel.values.shape == (2, 2)
        np.testing.assert_array_equal(sel.values, arr[:, 0:2])

    def test_multiple_selections(self):
        """Several axes can be selected in one call."""
        arr = np.arange(24.0).reshape(2, 3, 4)
        la = LabeledArray(arr, AxisLayout([(P, 2), (X, 3), (Y, 4)]))
        sel = la.isel(p=0, y=2)
        assert sel.axes == (X,)
        np.testing.assert_array_equal(sel.values, arr[0, :, 2])

    def test_combined_int_and_slice_selection(self):
        """An int selection (dropped) and a slice selection (kept) combine in one call."""
        arr = np.arange(24.0).reshape(2, 3, 4)
        la = LabeledArray(arr, AxisLayout([(P, 2), (X, 3), (Y, 4)]))
        sel = la.isel(p=1, y=slice(1, 3))
        assert sel.axes == (X, Y)
        assert sel.values.shape == (3, 2)
        np.testing.assert_array_equal(sel.values, arr[1, :, 1:3])

    def test_unknown_axis_name_raises(self):
        """Selecting an axis not in the layout raises KeyError naming the valid ones."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (Y, 3)]))
        with pytest.raises(KeyError, match=r"LabeledArray.isel: no axis named 'z'"):
            la.isel(z=0)

    def test_isel_returns_a_new_labeled_array(self):
        """isel() does not mutate the original."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        la.isel(x=0)
        assert la.axes == (X, Y)
        assert la.values.shape == (2, 3)

    @pytest.mark.parametrize("position", [np.int64(1), np.intp(1), np.int32(1)])
    def test_numpy_integer_selects_like_an_int(self, position):
        """A NumPy integer (as returned by np.argmax and friends) drops the axis like a Python int (#283)."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        sel = la.isel(x=position)
        assert sel.axes == (Y,)
        np.testing.assert_array_equal(sel.values, arr[1, :])

    def test_position_from_argmax(self):
        """The position of the maximum, located programmatically, selects that cell."""
        arr = np.array([3.0, 9.0, 1.0])
        la = LabeledArray(arr, AxisLayout([(X, 3)]))
        assert float(la.isel(x=np.argmax(la.values)).values) == 9.0

    @pytest.mark.parametrize("value", [1.0, True, np.bool_(True), None, [0, 1]])
    def test_non_integer_raises(self, value):
        """Anything but an integer or a slice is a TypeError naming the axis."""
        la = LabeledArray(np.zeros(3), AxisLayout([(X, 3)]))
        with pytest.raises(TypeError, match=r"axis 'x': expected an integer or a slice"):
            la.isel(x=value)

    @pytest.mark.parametrize("value", ["north", slice("north", None)])
    def test_label_raises_pointing_to_sel(self, value):
        """Labels select through sel()."""
        la = LabeledArray(np.zeros(3), AxisLayout([(ZONE, 3)]))
        with pytest.raises(TypeError, match=r"isel selects by position; use sel\(\) for labels"):
            la.isel(zone=value)


class TestSelByPosition:
    """Positional selection through sel(), deprecated in favour of isel()."""

    @pytest.mark.parametrize("value", [1, np.int64(1), slice(0, 1)])
    def test_positional_sel_warns_and_selects(self, value):
        """An integer or slice in sel() warns, pointing to isel(), and selects like isel()."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 3)]))
        match = r"selecting axis 'x' by position is deprecated.*Use isel\(x=...\) instead"
        with pytest.warns(DeprecationWarning, match=match):
            sel = la.sel(x=value)
        np.testing.assert_array_equal(sel.values, la.isel(x=value).values)

    def test_other_values_raise(self):
        """A value that is neither a label nor a position is a TypeError."""
        la = LabeledArray(np.zeros(3), AxisLayout([(ZONE, 3)]))
        with pytest.raises(TypeError, match=r"axis 'zone': expected a label or a list of labels, got float"):
            la.sel(zone=1.0)  # type: ignore[arg-type]


class TestLabels:
    """Labelled dimensions: axes typed SetType(labels=...) whose size matches the labels."""

    def test_labels_come_from_the_axis_type(self):
        """A labelled axis at full size carries its labels; other axes are absent."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (ZONE, 3)]))
        assert la.labels == {"zone": ("centre", "north", "south")}

    def test_labels_are_dropped_when_the_size_does_not_match(self):
        """A size-1 dimension of a labelled axis cannot be labelled by position."""
        la = LabeledArray(np.zeros((2, 1)), AxisLayout([(X, 2), (ZONE, 1)]))
        assert la.labels == {}

    def test_labels_property_returns_a_copy(self):
        """Mutating the returned dict does not affect the array."""
        la = LabeledArray(np.zeros(3), AxisLayout([(ZONE, 3)]))
        la.labels.clear()
        assert la.labels == {"zone": ("centre", "north", "south")}

    def test_label_selection_drops_the_axis(self):
        """A label selects one position, like an int, and drops the dimension and its labels."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (ZONE, 3)]))
        sel = la.sel(zone="north")
        assert sel.axes == (X,)
        assert sel.labels == {}
        np.testing.assert_array_equal(sel.values, arr[:, 1])

    def test_slice_keeps_the_matching_labels(self):
        """After a slice, labels still match positions, so later label selection stays correct."""
        arr = np.array([10.0, 20.0, 30.0])
        la = LabeledArray(arr, AxisLayout([(ZONE, 3)]))
        tail = la.isel(zone=slice(1, 3))
        assert tail.labels == {"zone": ("north", "south")}
        assert float(tail.sel(zone="south").values) == 30.0

    def test_selection_on_another_axis_keeps_labels(self):
        """Selecting along an unlabelled axis leaves the labels untouched."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (ZONE, 3)]))
        assert la.isel(x=0).labels == {"zone": ("centre", "north", "south")}

    def test_list_of_labels_keeps_the_axis_in_the_given_order(self):
        """A list of labels selects those members, in the order given, and keeps them as labels."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (ZONE, 3)]))
        sel = la.sel(zone=["south", "centre"])
        assert sel.axes == (X, ZONE)
        assert sel.labels == {"zone": ("south", "centre")}
        np.testing.assert_array_equal(sel.values, arr[:, [2, 0]])
        assert float(sel.isel(x=1).sel(zone="centre").values) == 3.0

    def test_tuple_of_labels_is_accepted(self):
        """A tuple selects like a list."""
        la = LabeledArray(np.array([10.0, 20.0, 30.0]), AxisLayout([(ZONE, 3)]))
        np.testing.assert_array_equal(la.sel(zone=("north",)).values, [20.0])

    def test_lists_on_two_axes_select_every_combination(self):
        """Lists on two axes cross, giving every pair, rather than pairing their entries."""
        mode = DomainAxis("mode", type=SetType(labels=("bike", "bus", "car")))
        arr = np.arange(9.0).reshape(3, 3)
        la = LabeledArray(arr, AxisLayout([(mode, 3), (ZONE, 3)]))
        sel = la.sel(mode=["car", "bike"], zone=["north", "south"])
        assert sel.values.shape == (2, 2)
        assert sel.labels == {"mode": ("car", "bike"), "zone": ("north", "south")}
        np.testing.assert_array_equal(sel.values, arr[np.ix_([2, 0], [1, 2])])

    def test_list_combined_with_a_slice_and_an_int(self):
        """A list on one axis composes with positional selection on the others."""
        arr = np.arange(24.0).reshape(2, 4, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (Y, 4), (ZONE, 3)]))
        sel = la.isel(x=1, y=slice(1, 3)).sel(zone=["south"])
        assert sel.axes == (Y, ZONE)
        np.testing.assert_array_equal(sel.values, arr[1, 1:3][:, [2]])

    def test_unknown_label_raises(self):
        """A label outside the axis's labels is a KeyError listing them."""
        la = LabeledArray(np.zeros(3), AxisLayout([(ZONE, 3)]))
        with pytest.raises(KeyError, match=r"'east' is not a label of axis 'zone'"):
            la.sel(zone="east")

    def test_label_on_an_unlabelled_axis_raises(self):
        """A str on an axis without labels is a KeyError, not passed to numpy."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (ZONE, 3)]))
        with pytest.raises(KeyError, match=r"axis 'x' has no labels; select it by position with isel\(\)"):
            la.sel(x="a")

    def test_unknown_label_in_a_list_raises(self):
        """A list is checked label by label."""
        la = LabeledArray(np.zeros(3), AxisLayout([(ZONE, 3)]))
        with pytest.raises(KeyError, match=r"'east' is not a label of axis 'zone'"):
            la.sel(zone=["north", "east"])

    def test_list_on_an_unlabelled_axis_raises(self):
        """A list of labels needs a labelled axis."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (ZONE, 3)]))
        with pytest.raises(KeyError, match=r"axis 'x' has no labels"):
            la.sel(x=["a"])

    def test_empty_list_raises(self):
        """An empty list selects nothing and is rejected."""
        la = LabeledArray(np.zeros(3), AxisLayout([(ZONE, 3)]))
        with pytest.raises(ValueError, match=r"the list of labels is empty"):
            la.sel(zone=[])

    def test_repeated_label_in_a_list_raises(self):
        """A repeated label would duplicate a member and is rejected."""
        la = LabeledArray(np.zeros(3), AxisLayout([(ZONE, 3)]))
        with pytest.raises(ValueError, match=r"repeats a label"):
            la.sel(zone=["north", "north"])

    def test_non_label_in_a_list_raises(self):
        """A list selects by label only; positions in a list are rejected."""
        la = LabeledArray(np.zeros(3), AxisLayout([(ZONE, 3)]))
        with pytest.raises(TypeError, match=r"a list selects by label, but 0 is not a label"):
            la.sel(zone=[0, 1])  # type: ignore[list-item]

    @pytest.mark.parametrize("bounds", [("north", "south"), ("north", None), (None, "south")])
    def test_label_slice_raises(self, bounds):
        """Slices select by position; a label bound points to a list of labels instead."""
        la = LabeledArray(np.zeros(3), AxisLayout([(ZONE, 3)]))
        with pytest.raises(TypeError, match=r"slices select by position; to select several labels, pass a list"):
            la.sel(zone=slice(*bounds))


class TestTranspose:
    """Reading an array with its dimensions in a chosen order."""

    def test_reorders_values_layout_and_labels(self):
        """The dimensions follow the order given; labels go with their dimension."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(X, 2), (ZONE, 3)]))
        out = la.transpose("zone", X)
        assert out.axes == (ZONE, X)
        np.testing.assert_array_equal(out.values, arr.T)
        assert out.labels == {"zone": ("centre", "north", "south")}
        assert float(out.isel(x=1).sel(zone="south").values) == 5.0

    def test_roles_may_be_interleaved(self):
        """Any order is accepted, across roles; selection works on the reordered array."""
        arr = np.arange(6.0).reshape(2, 3)
        la = LabeledArray(arr, AxisLayout([(P, 2), (ZONE, 3)]))
        out = la.transpose(ZONE, P)
        assert out.axes == (ZONE, P)
        assert out.sizes == (3, 2)
        np.testing.assert_array_equal(out.sel(zone="north").values, arr[:, 1])
        np.testing.assert_array_equal(out.isel(p=1).values, arr[1, :])

    def test_same_order_returns_the_array(self):
        """The current order needs no transposition."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (Y, 3)]))
        assert la.transpose(X, Y) is la

    @pytest.mark.parametrize(
        ("axes", "problem"),
        [
            (("x",), r"missing \['y'\]"),
            (("x", "y", "z"), r"unknown \['z'\]"),
            (("x", "y", "x"), r"repeated \['x'\]"),
        ],
    )
    def test_axes_must_be_all_the_axes_each_once(self, axes, problem):
        """Missing, unknown and repeated axes are named."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (Y, 3)]))
        match = r"LabeledArray.transpose: the axes must be all of \['x', 'y'\].*" + problem
        with pytest.raises(ValueError, match=match):
            la.transpose(*axes)

    def test_axis_object_must_match(self):
        """An axis object with the right name but another role is not this array's axis."""
        la = LabeledArray(np.zeros((2, 3)), AxisLayout([(X, 2), (Y, 3)]))
        with pytest.raises(ValueError, match=r"does not match this array's axis"):
            la.transpose(Axis("x", PARAMETER), Y)


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

    def test_to_xarray_carries_labels_as_coordinates(self):
        """Labelled dimensions become xarray coordinates, so xarray's sel works by label."""
        pytest.importorskip("xarray")
        la = LabeledArray(np.array([[1.0, 2.0, 3.0]]), AxisLayout([(X, 1), (ZONE, 3)]))
        da = la.to_xarray()
        assert list(da.coords["zone"].values) == ["centre", "north", "south"]
        assert "x" not in da.coords
        assert float(da.sel(zone="north", x=0)) == 2.0


class TestLabelIndex:
    """Label lookups use a label-to-position index built once per labelled axis."""

    def test_index_is_built_once_and_reused(self):
        """Selecting one label at a time does not rebuild the index."""
        arr = LabeledArray(np.arange(3.0), [(ZONE, 3)])
        assert float(arr.sel(zone="north").values) == 1.0
        index = arr._label_index["zone"][1]
        assert float(arr.sel(zone="south").values) == 2.0
        assert arr._label_index["zone"][1] is index

    def test_index_follows_the_labels(self):
        """An index built for other labels is not used."""
        arr = LabeledArray(np.arange(3.0), [(ZONE, 3)])
        arr.sel(zone="north")
        arr._labels["zone"] = ("south", "north", "centre")
        assert float(arr.sel(zone="south").values) == 0.0
