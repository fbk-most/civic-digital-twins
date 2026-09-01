"""Tests for the axis utilities in dt_model.axes."""

# SPDX-License-Identifier: Apache-2.0

from civic_digital_twins.dt_model.axes import (
    DOMAIN,
    ENSEMBLE,
    PARAMETER,
    TIME_AXIS,
    Axis,
    DomainAxis,
    MeshType,
    SequenceType,
    SetType,
    SpaceType,
    TimeType,
    domain_axis_position,
    filter_by_role,
    union_axes,
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
        ax = DomainAxis("x", type=SpaceType(spacing=5.0, boundary="wrap"))
        ctx = {"DomainAxis": DomainAxis, "SpaceType": SpaceType}
        rebuilt = eval(repr(ax), ctx)  # noqa: S307
        assert rebuilt == ax
        assert isinstance(rebuilt.type, SpaceType)
        assert rebuilt.type.spacing == 5.0
        assert rebuilt.type.boundary == "wrap"


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
        """SetType is a parameterless marker."""
        assert repr(SetType()) == "SetType()"

    def test_sequence_type_defaults(self):
        """SequenceType defaults to non-periodic."""
        assert SequenceType().periodic is False
        assert SequenceType(periodic=True).periodic is True

    def test_time_type_is_a_sequence_type(self):
        """TimeType extends SequenceType (lattice: SequenceType subset of TimeType)."""
        t = TimeType(periodic=True)
        assert isinstance(t, SequenceType)
        assert t.periodic is True

    def test_space_type_is_a_sequence_type(self):
        """SpaceType extends SequenceType and carries a metric + boundary."""
        s = SpaceType(spacing=2.5, boundary="constant")
        assert isinstance(s, SequenceType)
        assert s.spacing == 2.5
        assert s.boundary == "constant"

    def test_space_type_defaults(self):
        """SpaceType defaults to unit spacing and reflect boundary."""
        s = SpaceType()
        assert s.spacing == 1.0
        assert s.boundary == "reflect"

    def test_mesh_type_carries_adjacency(self):
        """MeshType stores an opaque adjacency structure."""
        adj = {"a": ["b"]}
        m = MeshType(adjacency=adj)
        assert m.adjacency is adj


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
