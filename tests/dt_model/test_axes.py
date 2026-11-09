"""Tests for the axis utilities in dt_model.axes."""

# SPDX-License-Identifier: Apache-2.0

import pytest

from civic_digital_twins.dt_model.axes import (
    DOMAIN,
    ENSEMBLE,
    PARAMETER,
    TIME_AXIS,
    Axis,
    Constant,
    Dirichlet,
    DomainAxis,
    Linear,
    Nearest,
    Neumann,
    Reflect,
    SequenceType,
    SetType,
    SpaceType,
    TimeType,
    Wrap,
    declared_labels,
    declared_length,
    domain_axis_position,
    filter_by_role,
    union_axes,
    validate_axis_extent,
    validate_value_shape,
)


class TestTimeAxisSingleton:
    """The canonical TIME_AXIS singleton."""

    def test_identity(self):
        """TIME_AXIS is the time DOMAIN axis."""
        assert TIME_AXIS.name == "time"
        assert TIME_AXIS.role == DOMAIN

    def test_value_equality_with_local_construction(self):
        """TIME_AXIS compares and hashes equal to a locally constructed copy."""
        assert TIME_AXIS == Axis("time", DOMAIN)
        assert hash(TIME_AXIS) == hash(Axis("time", DOMAIN))


class TestUnionAxes:
    """union_axes() merge semantics."""

    def test_empty(self):
        """No input tuples (or empty ones) yield an empty tuple."""
        assert union_axes() == ()
        assert union_axes((), ()) == ()

    def test_preserves_first_seen_order(self):
        """Axes appear in first-seen order across input tuples."""
        a = Axis("a", PARAMETER)
        b = Axis("b", ENSEMBLE)
        c = Axis("c", DOMAIN)
        assert union_axes((a, b), (c, a)) == (a, b, c)

    def test_deduplicates_by_value(self):
        """Two distinct objects with equal (name, role) count as one axis."""
        a1 = Axis("a", PARAMETER)
        a2 = Axis("a", PARAMETER)
        assert union_axes((a1,), (a2,)) == (a1,)

    def test_same_name_different_role_are_distinct(self):
        """Axes sharing a name but not a role are kept separate."""
        ad = Axis("x", DOMAIN)
        ap = Axis("x", PARAMETER)
        assert union_axes((ad,), (ap,)) == (ad, ap)


class TestFilterByRole:
    """filter_by_role() selection semantics."""

    def test_filters_and_preserves_order(self):
        """Only axes with the requested role survive, in input order."""
        p1 = Axis("p1", PARAMETER)
        e1 = Axis("e1", ENSEMBLE)
        p2 = Axis("p2", PARAMETER)
        assert filter_by_role((p1, e1, p2), PARAMETER) == (p1, p2)
        assert filter_by_role((p1, e1, p2), ENSEMBLE) == (e1,)

    def test_no_match(self):
        """An empty tuple is returned when no axis has the role."""
        assert filter_by_role((Axis("p", PARAMETER),), DOMAIN) == ()

    def test_accepts_any_iterable(self):
        """Any iterable of axes is accepted, not just tuples."""
        p = Axis("p", PARAMETER)
        assert filter_by_role(iter([p, TIME_AXIS]), DOMAIN) == (TIME_AXIS,)


class TestDomainAxis:
    """DomainAxis equality/hash invariant and type-carrying behaviour."""

    def test_equals_plain_axis_of_same_name_and_role(self):
        """A typed DomainAxis compares equal to a plain untyped Axis (load-bearing)."""
        typed = DomainAxis("time", type=TimeType())
        assert typed == Axis("time", DOMAIN)
        assert Axis("time", DOMAIN) == typed
        assert hash(typed) == hash(Axis("time", DOMAIN))

    def test_type_is_excluded_from_equality(self):
        """Two DomainAxis instances with different types still compare equal."""
        a = DomainAxis("space", type=SpaceType(spacing=2.0))
        b = DomainAxis("space", type=None)
        assert a == b
        assert hash(a) == hash(b)

    def test_role_is_fixed_to_domain(self):
        """DomainAxis always has role DOMAIN regardless of the type given."""
        ax = DomainAxis("cell", type=None)
        assert ax.role == DOMAIN

    def test_default_type_is_none(self):
        """An untyped DomainAxis stores type=None."""
        ax = DomainAxis("region")
        assert ax.type is None

    def test_repr_roundtrip(self):
        """DomainAxis.__repr__ is executable and reconstructs an equal instance."""
        ax = DomainAxis("x", type=SpaceType(spacing=5.0, boundary=Wrap()))
        ctx = {"DomainAxis": DomainAxis, "SpaceType": SpaceType, "Wrap": Wrap}
        rebuilt = eval(repr(ax), ctx)  # noqa: S307
        assert rebuilt == ax
        assert isinstance(rebuilt.type, SpaceType)
        assert rebuilt.type.spacing == 5.0
        assert isinstance(rebuilt.type.boundary, Wrap)

    def test_sequence_derived_types_allow_optional_length(self):
        """TimeType/SpaceType can be used with and without declared length."""
        t_free = DomainAxis("t_free", type=TimeType())
        t_len = DomainAxis("t_len", type=TimeType(length=24))
        x_free = DomainAxis("x_free", type=SpaceType())
        x_len = DomainAxis("x_len", type=SpaceType(length=128))

        assert isinstance(t_free.type, TimeType) and t_free.type.length is None
        assert isinstance(t_len.type, TimeType) and t_len.type.length == 24
        assert isinstance(x_free.type, SpaceType) and x_free.type.length is None
        assert isinstance(x_len.type, SpaceType) and x_len.type.length == 128


class TestTimeAxisTyped:
    """TIME_AXIS is now a typed DomainAxis but keeps full backward compatibility."""

    def test_time_axis_is_domain_axis(self):
        """TIME_AXIS is a DomainAxis carrying a TimeType."""
        assert isinstance(TIME_AXIS, DomainAxis)
        assert isinstance(TIME_AXIS.type, TimeType)

    def test_time_axis_still_equals_plain_axis(self):
        """TIME_AXIS still equals and hashes as a plain Axis("time", DOMAIN)."""
        assert TIME_AXIS == Axis("time", DOMAIN)
        assert hash(TIME_AXIS) == hash(Axis("time", DOMAIN))


class TestDomainTypeLattice:
    """DomainType concrete marker classes are metadata-only carriers."""

    def test_set_type_repr_roundtrip(self):
        """SetType's repr shows labels, else length, else nothing."""
        assert repr(SetType()) == "SetType()"
        assert repr(SetType(length=3)) == "SetType(length=3)"
        assert repr(SetType(labels=["a", "b"])) == "SetType(labels=('a', 'b'))"
        assert repr(SetType(length=2, labels=("a", "b"))) == "SetType(labels=('a', 'b'))"

    def test_set_type_defaults_to_no_length_and_no_labels(self):
        """A bare SetType leaves the extent unconstrained."""
        s = SetType()
        assert s.length is None
        assert s.labels is None

    def test_set_type_length(self):
        """SetType(length=n) declares a length without labels."""
        s = SetType(length=3)
        assert s.length == 3
        assert s.labels is None

    def test_set_type_labels_declare_the_length(self):
        """Labels are stored as a tuple and their number is the length."""
        s = SetType(labels=["centre", "north", "south"])
        assert s.labels == ("centre", "north", "south")
        assert s.length == 3

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            pytest.param({"length": 0}, ValueError, r"length must be positive; got 0", id="zero-length"),
            pytest.param({"labels": []}, ValueError, r"labels must not be empty", id="empty-labels"),
            pytest.param({"labels": ["a", 1]}, TypeError, r"labels must be strings; got \[1\]", id="non-str"),
            pytest.param({"labels": ["a", "b", "a"]}, ValueError, r"distinct; duplicated: \['a'\]", id="duplicates"),
            pytest.param(
                {"length": 3, "labels": ["a", "b"]}, ValueError, r"length=3 does not match the 2 labels", id="mismatch"
            ),
        ],
    )
    def test_set_type_rejects_invalid_declarations(self, kwargs, error, message):
        """Invalid lengths and labels are rejected when the type is built."""
        with pytest.raises(error, match=message):
            SetType(**kwargs)

    def test_set_type_validate_extent(self):
        """A set axis with a declared length accepts that size or 1, like a sequence axis."""
        s = SetType(labels=("a", "b", "c"))
        s.validate_extent(size=3, axis_name="zone")
        s.validate_extent(size=1, axis_name="zone")
        SetType().validate_extent(size=7, axis_name="zone")
        with pytest.raises(ValueError, match=r"axis 'zone' declares length=3, but got dimension size 2"):
            s.validate_extent(size=2, axis_name="zone")

    def test_sequence_type_defaults_to_undefined_length(self):
        """SequenceType supports optional declared length and defaults to None."""
        s = SequenceType()
        assert s.length is None
        assert repr(s) == "SequenceType()"

    def test_sequence_type_supports_declared_length(self):
        """SequenceType(length=...) is round-tripped in repr."""
        s = SequenceType(length=12)
        assert s.length == 12
        assert repr(s) == "SequenceType(length=12)"

    def test_sequence_type_rejects_non_positive_length(self):
        """Declared lengths must be positive when provided."""
        import pytest

        with pytest.raises(ValueError, match="must be positive"):
            SequenceType(length=0)

    def test_sequence_type_validates_extent(self):
        """SequenceType validates concrete extents via its declared length."""
        import pytest

        axis = DomainAxis("t", type=SequenceType(length=3))
        validate_axis_extent(axis, 3)
        validate_axis_extent(axis, 1)
        with pytest.raises(ValueError, match="declares length=3"):
            validate_axis_extent(axis, 2)

    def test_untyped_axis_has_no_extent_validation(self):
        """Untyped axes impose no static extent constraints."""
        validate_axis_extent(DomainAxis("u"), 17)

    def test_time_type_is_a_sequence_type(self):
        """TimeType extends SequenceType (lattice: SequenceType subset of TimeType)."""
        t = TimeType(length=24)
        assert isinstance(t, SequenceType)
        assert t.length == 24
        assert repr(t) == "TimeType(length=24)"

    def test_space_type_is_a_sequence_type(self):
        """SpaceType extends SequenceType and carries metric + boundary + optional length."""
        s = SpaceType(spacing=2.5, boundary=Constant(1.0), length=7)
        assert isinstance(s, SequenceType)
        assert s.length == 7
        assert s.spacing == 2.5
        assert isinstance(s.boundary, Constant)
        assert s.boundary.value == 1.0
        assert repr(s) == "SpaceType(spacing=2.5, boundary=Constant(value=1.0), length=7)"

    def test_space_type_defaults(self):
        """SpaceType defaults to unit spacing, Reflect boundary, and undefined length."""
        s = SpaceType()
        assert s.length is None
        assert s.spacing == 1.0
        assert s.boundary is Reflect


class TestBoundaryConditions:
    """BoundaryCondition classes: repr round-trip and default values."""

    def test_constant_repr_and_default(self):
        """Constant defaults to value=0.0 and has a round-trippable repr."""
        assert Constant().value == 0.0
        assert repr(Constant(2.5)) == "Constant(value=2.5)"

    def test_dirichlet_is_an_alias_for_constant(self):
        """Dirichlet is the same class as Constant, not a distinct subclass."""
        assert Dirichlet is Constant
        assert isinstance(Dirichlet(1.0), Constant)

    def test_neumann_repr_and_default(self):
        """Neumann defaults to value=0.0 (the zero-flux/"reflecting" case)."""
        assert Neumann().value == 0.0
        assert repr(Neumann(1.5)) == "Neumann(value=1.5)"

    def test_reflect_is_an_alias_for_the_zero_flux_neumann_instance(self):
        """Reflect is Neumann(0.0) itself, not a distinct subclass or a fresh equal instance."""
        assert isinstance(Reflect, Neumann)
        assert Reflect.value == 0.0
        assert repr(Reflect) == "Neumann(value=0.0)"
        assert SpaceType().boundary is Reflect

    def test_nearest_repr(self):
        """Nearest is a parameterless marker."""
        assert repr(Nearest()) == "Nearest()"

    def test_wrap_repr(self):
        """Wrap is a parameterless marker."""
        assert repr(Wrap()) == "Wrap()"

    def test_linear_repr(self):
        """Linear is a parameterless marker."""
        assert repr(Linear()) == "Linear()"


class TestTopLevelExports:
    """Axis vocabulary re-exported by the top-level package."""

    def test_time_axis_reexported(self):
        """TIME_AXIS is importable from civic_digital_twins.dt_model."""
        import civic_digital_twins.dt_model as dt

        assert dt.TIME_AXIS is TIME_AXIS
        assert dt.Axis is Axis


class TestDomainAxisPosition:
    """Mapping a named DOMAIN axis to its trailing numpy dimension."""

    def test_sole_axis_is_the_last_dimension(self):
        """One DOMAIN axis sits at -1 — the convention that used to be hard-coded."""
        assert domain_axis_position((TIME_AXIS,), TIME_AXIS) == -1

    def test_positions_count_back_from_the_end(self):
        """Axes map to trailing dimensions in the order given."""
        x, y, z = Axis("x", DOMAIN), Axis("y", DOMAIN), Axis("z", DOMAIN)
        assert domain_axis_position((x, y, z), x) == -3
        assert domain_axis_position((x, y, z), y) == -2
        assert domain_axis_position((x, y, z), z) == -1

    def test_negative_indexing_survives_extra_leading_dimensions(self):
        """The position stays correct however many PARAMETER/ENSEMBLE dims precede it."""
        import numpy as np

        x, y = Axis("x", DOMAIN), Axis("y", DOMAIN)
        # Same trailing (2, 3) DOMAIN block under different leading ranks.
        for shape in ((2, 3), (5, 2, 3), (7, 5, 2, 3)):
            arr = np.zeros(shape)
            assert arr.shape[domain_axis_position((x, y), x)] == 2
            assert arr.shape[domain_axis_position((x, y), y)] == 3

    def test_unknown_axis_raises_value_error(self):
        """An axis the evaluation does not carry has no position."""
        import pytest

        with pytest.raises(ValueError, match="not one of the DOMAIN axes"):
            domain_axis_position((TIME_AXIS,), Axis("space", DOMAIN))

    def test_empty_domain_axes_raises(self):
        """With no DOMAIN axes nothing is projectable."""
        import pytest

        with pytest.raises(ValueError, match="not one of the DOMAIN axes"):
            domain_axis_position((), TIME_AXIS)


class TestDeclaredLength:
    """declared_length reads the length an axis declares through its type."""

    def test_plain_axis_declares_none(self):
        """A plain Axis has no type, hence no length."""
        assert declared_length(Axis("x", DOMAIN)) is None

    def test_untyped_and_length_less_axes_declare_none(self):
        """An untyped DomainAxis and a type without length declare none."""
        assert declared_length(DomainAxis("x")) is None
        assert declared_length(DomainAxis("x", type=SetType())) is None
        assert declared_length(DomainAxis("x", type=SequenceType())) is None

    def test_sequence_like_types_declare_their_length(self):
        """SequenceType, TimeType and SpaceType return their declared length."""
        assert declared_length(DomainAxis("x", type=SequenceType(length=4))) == 4
        assert declared_length(DomainAxis("t", type=TimeType(length=24))) == 24
        assert declared_length(DomainAxis("s", type=SpaceType(length=7))) == 7


class TestDeclaredLabels:
    """declared_labels reads the labels an axis declares through its type."""

    def test_only_labelled_set_axes_declare_labels(self):
        """Plain, untyped, length-only and sequence axes declare none."""
        assert declared_labels(Axis("x", DOMAIN)) is None
        assert declared_labels(DomainAxis("x")) is None
        assert declared_labels(DomainAxis("x", type=SetType(length=2))) is None
        assert declared_labels(DomainAxis("x", type=SequenceType(length=2))) is None
        assert declared_labels(DomainAxis("x", type=SetType(labels=("a", "b")))) == ("a", "b")

    def test_set_axis_declares_its_length(self):
        """declared_length covers set axes too, through length= or labels=."""
        assert declared_length(DomainAxis("x", type=SetType(length=4))) == 4
        assert declared_length(DomainAxis("x", type=SetType(labels=("a", "b")))) == 2


class TestValidateValueShape:
    """validate_value_shape checks a concrete value's shape against its declared axes."""

    ZONE = DomainAxis("zone", type=SetType())
    HOUR = DomainAxis("hour", type=TimeType(length=24))

    def test_matching_shape_passes(self):
        """One dimension per axis, sizes free or equal to the declared length."""
        validate_value_shape("Index 'v'", (self.ZONE, self.HOUR), (3, 24))

    def test_size_one_passes_against_a_declared_length(self):
        """A value that does not vary along an axis has size 1 there."""
        validate_value_shape("Index 'v'", (self.ZONE, self.HOUR), (3, 1))

    def test_empty_axes_and_scalar_shape_pass(self):
        """No axes and a scalar shape agree."""
        validate_value_shape("Index 'v'", (), ())

    @pytest.mark.parametrize(
        ("axes", "shape", "message"),
        [
            pytest.param(
                (ZONE,), (2, 3), r"axes=\(zone,\) declares 1 dimension but the value has shape \(2, 3\)", id="too-many"
            ),
            pytest.param(
                (ZONE, HOUR),
                (3,),
                r"axes=\(zone, hour\) declares 2 dimensions but the value has shape \(3,\)",
                id="too-few",
            ),
            pytest.param((ZONE,), (), r"axes=\(zone,\) declares 1 dimension but the value has shape \(\)", id="scalar"),
        ],
    )
    def test_rank_mismatch_raises(self, axes, shape, message):
        """The rank must equal the number of axes; the error names owner, axes and shape."""
        with pytest.raises(ValueError, match=r"^Index 'v': " + message):
            validate_value_shape("Index 'v'", axes, shape)

    def test_size_other_than_one_or_declared_length_raises(self):
        """A transposed table is caught when the swapped axes have different lengths."""
        with pytest.raises(
            ValueError,
            match=(
                r"^Index 'inflow': axis 'hour' declares length=24, but the value has size 3 along it "
                r"\(axes=\(zone, hour\) matched to value shape \(24, 3\)\)\.$"
            ),
        ):
            validate_value_shape("Index 'inflow'", (self.ZONE, self.HOUR), (24, 3))

    def test_set_axis_length_is_checked(self):
        """A set axis with labels gets the same check as a sequence axis with a length."""
        zone = DomainAxis("zone", type=SetType(labels=("centre", "north", "south")))
        validate_value_shape("Index 'v'", (zone,), (3,))
        with pytest.raises(ValueError, match=r"axis 'zone' declares length=3, but the value has size 4"):
            validate_value_shape("Index 'v'", (zone,), (4,))
