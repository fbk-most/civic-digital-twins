"""Tests for the axis utilities in dt_model.axes."""

# SPDX-License-Identifier: Apache-2.0

from civic_digital_twins.dt_model.axes import (
    DOMAIN,
    ENSEMBLE,
    PARAMETER,
    TIME_AXIS,
    Axis,
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
