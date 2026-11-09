"""Construction of replace_axes nodes: carrying a value by other DOMAIN axes."""

# SPDX-License-Identifier: Apache-2.0

import pytest

from civic_digital_twins.dt_model.axes import (
    DOMAIN,
    PARAMETER,
    Axis,
    DomainAxis,
    SequenceType,
    SetType,
    TimeType,
    replaced_axis,
)
from civic_digital_twins.dt_model.engine.frontend import graph

LOCAL_TIME = DomainAxis("local_time", type=TimeType(length=24))
TIME = DomainAxis("time", type=TimeType())
X = Axis("x", DOMAIN)
Y = Axis("y", DOMAIN)
Z = Axis("z", DOMAIN)


def _over(*axes):
    return graph.placeholder("v").broadcast(*axes)


class TestReplacedAxis:
    """The axis a value carries after a replacement: the target refined with the source."""

    def test_target_gains_the_source_length(self):
        """A generic target takes the source's length."""
        assert replaced_axis(LOCAL_TIME, TIME) == TIME
        assert replaced_axis(LOCAL_TIME, TIME).type == TimeType(length=24)  # type: ignore[attr-defined]

    def test_untyped_target_takes_the_source_type(self):
        """A target without a type takes the source's type, labels included."""
        zone = DomainAxis("zone", type=SetType(labels=("a", "b")))
        district = replaced_axis(zone, Axis("district", DOMAIN))
        assert isinstance(district, DomainAxis)
        assert (district.name, district.type) == ("district", SetType(labels=("a", "b")))

    def test_untyped_source_leaves_the_target(self):
        """A source without a type adds nothing."""
        assert replaced_axis(X, TIME) is TIME

    def test_target_already_as_specific_is_returned(self):
        """A target declaring at least what the source declares is returned as is."""
        target = DomainAxis("time", type=TimeType(length=24))
        assert replaced_axis(LOCAL_TIME, target) is target

    @pytest.mark.parametrize(
        ("source", "target", "reason"),
        [
            (LOCAL_TIME, DomainAxis("zone", type=SetType()), "different kinds of axis"),
            (LOCAL_TIME, DomainAxis("time", type=TimeType(length=12)), "length 24 against length 12"),
        ],
    )
    def test_conflicting_types_raise(self, source, target, reason):
        """Types that do not merge are an error naming both axes."""
        match = rf"axis 'local_time' cannot be replaced by axis '{target.name}'.*{reason}"
        with pytest.raises(ValueError, match=match):
            replaced_axis(source, target)

    def test_only_domain_axes(self):
        """PARAMETER and ENSEMBLE axes cannot be replaced."""
        with pytest.raises(ValueError, match=r"only DOMAIN axes can be replaced"):
            replaced_axis(Axis("p", PARAMETER), X)


class TestReplaceAxesNode:
    """The replace_axes node and the Node methods building it."""

    def test_target_takes_the_place_of_the_source(self):
        """The output axes keep their order, with the target where the source was."""
        node = _over(LOCAL_TIME, X).replace_axis(LOCAL_TIME, TIME)
        assert isinstance(node, graph.replace_axes)
        assert node.output_axes == (TIME, X)
        assert node.output_axes[0].type == TimeType(length=24)  # type: ignore[attr-defined]

    def test_several_pairs_at_once(self):
        """A mapping replaces several axes in one node."""
        node = _over(X, Y).replace_axes({X: Z, Y: TIME})
        assert node.output_axes == (Z, TIME)

    def test_swap(self):
        """All pairs apply at once, so two axes can be swapped."""
        node = _over(X, Y).replace_axes({X: Y, Y: X})
        assert node.output_axes == (Y, X)

    def test_pairs_are_accepted(self):
        """The mapping may be given as pairs."""
        node = graph.replace_axes(_over(X), [(X, Y)])
        assert node.mapping == ((X, Y),)

    def test_carried_declaration_is_used(self):
        """The source as the value carries it (with its declarations) decides what the target gains."""
        node = _over(LOCAL_TIME).replace_axis(Axis("local_time", DOMAIN), TIME)
        assert node.output_axes[0].type == TimeType(length=24)  # type: ignore[attr-defined]

    def test_empty_mapping_raises(self):
        """At least one pair is needed."""
        with pytest.raises(ValueError, match=r"no axes to replace"):
            _over(X).replace_axes({})

    def test_axis_replaced_by_itself_raises(self):
        """A pair must name two different axes."""
        with pytest.raises(ValueError, match=r"axis 'x' is replaced by itself"):
            _over(X).replace_axis(X, X)

    @pytest.mark.parametrize(
        ("pairs", "label"),
        [([(X, Z), (X, Y)], "source"), ([(X, Z), (Y, Z)], "target")],
    )
    def test_repeated_axes_raise(self, pairs, label):
        """A source or a target cannot appear twice."""
        with pytest.raises(ValueError, match=rf"{label} axes \['[xz]'\] appear more than once"):
            graph.replace_axes(_over(X, Y), pairs)

    def test_source_not_carried_raises(self):
        """The value must carry every source; broadcast is the way to add an axis."""
        with pytest.raises(ValueError, match=r"does not carry axis 'y' \(it carries \['x'\]\).*use broadcast"):
            _over(X).replace_axis(Y, Z)

    def test_target_already_carried_raises(self):
        """Two dimensions cannot collapse onto one axis."""
        with pytest.raises(ValueError, match=r"already carries axis 'y', so 'x' cannot be replaced by it"):
            _over(X, Y).replace_axis(X, Y)

    def test_conflicting_types_raise(self):
        """The types of a pair must merge."""
        with pytest.raises(ValueError, match=r"cannot be replaced by axis 'n'.*different kinds of axis"):
            _over(LOCAL_TIME).replace_axis(LOCAL_TIME, DomainAxis("n", type=SequenceType()))

    def test_source_conflicting_with_the_carried_axis_raises(self):
        """The source argument and the axis the value carries name one axis, so their types must merge."""
        zone = DomainAxis("zone", type=SetType(labels=("a", "b", "c")))
        match = r"axis 'zone' is declared with conflicting types.*length 3 against length 7"
        with pytest.raises(ValueError, match=match):
            _over(zone).replace_axis(DomainAxis("zone", type=SetType(length=7)), X)

    def test_source_argument_declaration_is_used(self):
        """What the source argument declares beyond the carried axis also goes to the target."""
        node = _over(Axis("local_time", DOMAIN)).replace_axis(LOCAL_TIME, TIME)
        assert node.output_axes[0].type == TimeType(length=24)  # type: ignore[attr-defined]
