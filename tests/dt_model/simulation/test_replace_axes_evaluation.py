"""Evaluating models that move values between DOMAIN axes with replace_axis (#298)."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import (
    ConstIndex,
    DistributionEnsemble,
    DistributionIndex,
    Evaluation,
    Index,
    Model,
    Scenario,
    define,
    inputs,
    outputs,
)
from civic_digital_twins.dt_model.axes import DomainAxis, SequenceType, SetType, TimeType

LOCAL_TIME = DomainAxis("local_time")
LIB_TIME = DomainAxis("time", type=TimeType())


def _evaluate(build, **evaluate_kwargs):
    """Evaluate a model whose single output is built by *build*."""

    @define("M")
    class M(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            out: Index

        def compute(self, inputs):
            return M.Outputs(out=build())

    m = M(inputs=M.Inputs())
    return m, Evaluation(Scenario(m)).evaluate(**evaluate_kwargs)


def test_round_trip_through_a_library_axis():
    """A value goes onto the library's axis, an operator runs there, the result comes back."""

    def build():
        load = ConstIndex("load", np.array([1.0, 2.0, 3.0, 4.0]), axes=(LOCAL_TIME,))
        running = load.replace_axis(LOCAL_TIME, LIB_TIME).cumulative(LIB_TIME)
        return Index("running", running.replace_axis(LIB_TIME, LOCAL_TIME), axes=(LOCAL_TIME,))

    m, result = _evaluate(build)
    np.testing.assert_array_equal(result.expected(m.outputs.out).values, [1.0, 3.0, 6.0, 10.0])


def test_target_without_a_length_takes_the_source_length():
    """The library axis declares no length and no value carries it: the link gives it the source's."""

    def build():
        constant = ConstIndex("constant", np.full(1, 5.0), axes=(LOCAL_TIME,))
        sized = ConstIndex("sized", np.zeros(3), axes=(LOCAL_TIME,))
        moved = (constant + sized).replace_axis(LOCAL_TIME, LIB_TIME)
        return Index("total", moved.sum(LIB_TIME))

    m, result = _evaluate(build)
    assert float(result.expected(m.outputs.out).values) == 15.0


def test_length_chains_through_several_links():
    """Links chain: a -> b -> c gives c the length of a."""
    a, b, c = DomainAxis("a"), DomainAxis("b"), DomainAxis("c")

    def build():
        value = ConstIndex("v", np.ones(4), axes=(a,))
        return Index("total", value.replace_axis(a, b).replace_axis(b, c).sum(c))

    m, result = _evaluate(build)
    assert float(result.expected(m.outputs.out).values) == 4.0


def test_declared_target_length_applies_to_the_source():
    """A length declared on the target is the source's too: a size-1 source is read at it."""
    target = DomainAxis("time", type=TimeType(length=6))

    def build():
        constant = ConstIndex("constant", np.full(1, 2.0), axes=(LOCAL_TIME,))
        return Index("total", constant.replace_axis(LOCAL_TIME, target).sum(target))

    m, result = _evaluate(build)
    assert float(result.expected(m.outputs.out).values) == 12.0


def test_source_data_disagreeing_with_the_declared_target_length_raises():
    """The source's data must have the length the target declares."""
    target = DomainAxis("time", type=TimeType(length=6))

    def build():
        load = ConstIndex("load", np.zeros(4), axes=(LOCAL_TIME,))
        return Index("moved", load.replace_axis(LOCAL_TIME, target), axes=(target,))

    with pytest.raises(ValueError, match=r"linked by replace_axes.*'time' declares length 6, but 'load' has 4 along"):
        _evaluate(build)


def test_linked_axes_with_different_sizes_raise():
    """Data along the source and along the target must agree on the length."""

    def build():
        load = ConstIndex("load", np.zeros(4), axes=(LOCAL_TIME,))
        other = ConstIndex("other", np.zeros(3), axes=(LIB_TIME,))
        return Index("sum", load.replace_axis(LOCAL_TIME, LIB_TIME) + other, axes=(LIB_TIME,))

    with pytest.raises(ValueError, match=r"\['local_time', 'time'\] \(linked by replace_axes.*'load' has 4 along"):
        _evaluate(build)


def test_linked_axes_declaring_different_lengths_raise():
    """A source declared with one length elsewhere in the model cannot be linked to a target with another."""
    source = DomainAxis("a", type=SequenceType(length=4))
    target = DomainAxis("b", type=SequenceType(length=5))

    def build():
        other = ConstIndex("other", np.zeros(4), axes=(source,))
        moved = ConstIndex("v", np.zeros(1), axes=(DomainAxis("a"),)).replace_axis(DomainAxis("a"), target)
        return Index("moved", moved + other.sum(source), axes=(target,))

    match = r"\['a', 'b'\] \(linked by replace_axes.*'a' declares length 4, 'b' declares length 5"
    with pytest.raises(ValueError, match=match):
        _evaluate(build)


def test_unlinked_axis_without_length_still_raises():
    """An axis reached only through broadcast still needs a length."""

    def build():
        return Index("b", ConstIndex("c", 1.0).broadcast(LOCAL_TIME), axes=(LOCAL_TIME,))

    with pytest.raises(ValueError, match=r"DOMAIN axis 'local_time' has no length"):
        _evaluate(build)


def test_labels_move_with_the_data():
    """Moving a labelled axis onto a generic one keeps the labels, so the result reads by label."""
    zone = DomainAxis("zone", type=SetType(labels=("centre", "north", "south")))
    district = DomainAxis("district", type=SetType())

    def build():
        load = ConstIndex("load", np.array([1.0, 2.0, 3.0]), axes=(zone,))
        moved = load.replace_axis(zone, district)
        return Index("moved", moved, axes=(DomainAxis("district", type=SetType(labels=("centre", "north", "south"))),))

    m, result = _evaluate(build)
    assert float(result.expected(m.outputs.out).sel(district="north").values) == 2.0


X = DomainAxis("x", type=SequenceType())
Y = DomainAxis("y", type=SequenceType())


def _swap(base):
    """Evaluate a grid over (x, y) and its swap, under a parameter sweep and an ensemble."""

    @define("Grid")
    class Grid(Model):
        @inputs
        class Inputs:
            scale: Index
            factor: Index

        @outputs
        class Outputs:
            grid: Index
            swapped: Index

        def compute(self, inputs):
            grid = Index("grid", ConstIndex("base", base, axes=(X, Y)) * inputs.scale * inputs.factor, axes=(X, Y))
            return Grid.Outputs(grid=grid, swapped=Index("swapped", grid.replace_axes({X: Y, Y: X}), axes=(X, Y)))

    scale = DistributionIndex("scale", stats.norm, {"loc": 1.0, "scale": 0.1})
    factor = Index("factor", 1.0)
    m = Grid(inputs=Grid.Inputs(scale=scale, factor=factor))
    scenario = Scenario(m)
    ensemble = DistributionEnsemble(scenario, size=3, rng=np.random.default_rng(0))
    return m, Evaluation(scenario).evaluate(ensemble=ensemble, parameters={factor: np.array([1.0, 2.0])})


def test_swap_under_parameters_and_an_ensemble():
    """A swap exchanges the data of two axes; PARAMETER and ENSEMBLE dimensions stay in place."""
    m, result = _swap(np.arange(9.0).reshape(3, 3))
    grid = result.samples(m.outputs.grid).values
    np.testing.assert_array_equal(result.samples(m.outputs.swapped).values, np.swapaxes(grid, -1, -2))


def test_swap_needs_axes_of_one_length():
    """Swapping links the two axes, so they must have the same length."""
    with pytest.raises(ValueError, match=r"\['x', 'y'\] \(linked by replace_axes.*different sizes"):
        _swap(np.arange(6.0).reshape(2, 3))
