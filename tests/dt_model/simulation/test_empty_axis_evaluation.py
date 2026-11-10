"""Evaluating models over a DOMAIN axis of length 0 (e.g. an entity axis with no entities)."""

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
from civic_digital_twins.dt_model.axes import DomainAxis, SequenceType, SetType

ZONE = DomainAxis("zone", type=SetType(labels=("a", "b", "c")))


def _costs_model(item_axis, costs):
    """Cost items per zone, summed per zone and overall; *costs* is (zone, item)."""

    @define("Costs")
    class Costs(Model):
        @inputs
        class Inputs:
            scale: Index
            factor: Index

        @outputs
        class Outputs:
            items: Index
            per_zone: Index
            total: Index

        def compute(self, inputs):
            base = ConstIndex("base", costs, axes=(ZONE, item_axis))
            items = Index("items", base * inputs.scale * inputs.factor, axes=(ZONE, item_axis))
            per_zone = Index("per_zone", items.sum(axis=item_axis), axes=(ZONE,))
            total = Index("total", per_zone.sum(axis=ZONE) + 100.0)
            return Costs.Outputs(items=items, per_zone=per_zone, total=total)

    scale = DistributionIndex("scale", stats.norm, {"loc": 1.0, "scale": 0.1})
    factor = Index("factor", 1.0)
    m = Costs(inputs=Costs.Inputs(scale=scale, factor=factor))
    scenario = Scenario(m)
    ensemble = DistributionEnsemble(scenario, size=4, rng=np.random.default_rng(0))
    result = Evaluation(scenario).evaluate(ensemble=ensemble, parameters={factor: np.array([1.0, 2.0])})
    return m, result


@pytest.mark.parametrize(
    "item_axis",
    [DomainAxis("item"), DomainAxis("item", type=SequenceType(length=0)), DomainAxis("item", type=SetType(labels=()))],
    ids=["inferred", "declared-length", "empty-labels"],
)
def test_no_cost_items(item_axis):
    """With no items, sums are 0 and every reader returns arrays empty along the item axis (#287)."""
    m, result = _costs_model(item_axis, np.zeros((3, 0)))
    assert result.layout.sizes[item_axis] == 0

    items = result.expected(m.outputs.items, axes=("factor", "zone", "item"))
    assert items.sizes == (2, 3, 0)
    assert result.samples(m.outputs.items, axes=("factor", "_ensemble", "zone", "item")).sizes == (2, 4, 3, 0)

    np.testing.assert_array_equal(result.expected(m.outputs.per_zone).values, np.zeros((2, 3)))
    np.testing.assert_array_equal(result.expected(m.outputs.total).values, [100.0, 100.0])


def test_selection_and_transpose_on_an_empty_axis():
    """Selection along other axes and reordering keep the empty axis."""
    m, result = _costs_model(DomainAxis("item"), np.zeros((3, 0)))
    items = result.expected(m.outputs.items)
    assert items.isel(factor=0).sel(zone="b").sizes == (0,)
    assert items.transpose("item", "zone", "factor").sizes == (0, 3, 2)


def test_value_constant_along_an_empty_axis_sums_to_0():
    """A value of size 1 along an axis declared empty is constant over no positions: its sum is 0."""
    item = DomainAxis("item", type=SequenceType(length=0))
    m, result = _costs_model(item, np.full((3, 1), 7.0))
    np.testing.assert_array_equal(result.expected(m.outputs.per_zone).values, np.zeros((2, 3)))
    assert result.expected(m.outputs.items, axes=("factor", "zone", "item")).sizes == (2, 3, 0)


def test_empty_and_non_empty_values_disagree():
    """An axis has one length per evaluation: an empty and a non-empty value of it are an error."""
    item = DomainAxis("item")

    @define("Mixed")
    class Mixed(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            total: Index

        def compute(self, inputs):
            a = ConstIndex("a", np.zeros(0), axes=(item,))
            b = ConstIndex("b", np.ones(2), axes=(item,))
            return Mixed.Outputs(total=Index("total", (a + b).sum(axis=item)))

    with pytest.raises(ValueError, match=r"DOMAIN axis 'item' has different sizes in different values"):
        Evaluation(Scenario(Mixed(inputs=Mixed.Inputs()))).evaluate()
