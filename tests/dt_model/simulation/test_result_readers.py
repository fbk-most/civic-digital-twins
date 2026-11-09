"""Result readers: EvaluationResult.expected() and samples()."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import (
    TIME_AXIS,
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
from civic_digital_twins.dt_model.axes import DOMAIN, ENSEMBLE, PARAMETER, Axis, DomainAxis, SetType
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout
from civic_digital_twins.dt_model.simulation.evaluation import EvaluationResult

ZONE = DomainAxis("zone", type=SetType(labels=("centre", "north", "south")))


@define("Roads")
class _Roads(Model):
    """A per-zone capacity constant over the day, broadcast over time, and an hourly load."""

    @inputs
    class Inputs:
        capacity: Index
        demand: Index

    @outputs
    class Outputs:
        hourly_capacity: Index
        load: Index

    def compute(self, inputs):
        hourly_capacity = Index("hourly capacity", inputs.capacity.broadcast(TIME_AXIS), axes=(ZONE, TIME_AXIS))
        load = Index("load", inputs.demand / hourly_capacity, axes=(ZONE, TIME_AXIS))
        return _Roads.Outputs(hourly_capacity=hourly_capacity, load=load)


def _roads(**evaluate_kwargs):
    m = _Roads(
        inputs=_Roads.Inputs(
            capacity=Index("capacity", np.array([1000.0, 800.0, 600.0]), axes=(ZONE,)),
            demand=Index("demand", np.full((3, 24), 500.0), axes=(ZONE, TIME_AXIS)),
        )
    )
    return m, Evaluation(Scenario(m)).evaluate(**evaluate_kwargs)


@define("Noisy")
class _Noisy(Model):
    """An uncertain scale times a deterministic per-zone grid, with a swept factor."""

    @inputs
    class Inputs:
        scale: Index
        factor: Index

    @outputs
    class Outputs:
        grid: Index
        scaled: Index

    def compute(self, inputs):
        grid = ConstIndex("grid", np.array([1.0, 2.0, 3.0]), axes=(ZONE,))
        scaled = Index("scaled", inputs.scale * inputs.factor * grid, axes=(ZONE,))
        return _Noisy.Outputs(grid=grid, scaled=scaled)


def _noisy(size=5):
    scale = DistributionIndex("scale", stats.norm, {"loc": 3.0, "scale": 0.5})
    factor = Index("factor", 1.0)
    m = _Noisy(inputs=_Noisy.Inputs(scale=scale, factor=factor))
    scenario = Scenario(m)
    ensemble = DistributionEnsemble(scenario, size=size, rng=np.random.default_rng(0))
    result = Evaluation(scenario).evaluate(ensemble=ensemble, parameters={factor: np.array([1.0, 10.0])})
    return m, scale, result


class TestExpected:
    """The expected value over the ensemble, named and at full size."""

    def test_broadcast_output_is_read_at_full_size(self):
        """An output constant along an axis is repeated along it, matching layout_of (#281)."""
        m, result = _roads()
        capacity = result.expected(m.outputs.hourly_capacity)
        assert capacity.layout == result.layout_of(m.outputs.hourly_capacity)
        assert capacity.values.shape == (24, 3)
        np.testing.assert_array_equal(capacity.isel(time=7).values, [1000.0, 800.0, 600.0])
        assert float(capacity.isel(time=23).sel(zone="south").values) == 600.0

    def test_values_are_a_read_only_view(self):
        """Writing into the expanded result raises NumPy's read-only error."""
        m, result = _roads()
        values = result.expected(m.outputs.hourly_capacity).values
        with pytest.raises(ValueError, match=r"read-only"):
            values[0, 0] = 0.0

    def test_axes_sets_the_order(self):
        """Axes= gives the result's axes in the order wanted, by object or by name (#288)."""
        m, result = _roads()
        default = result.expected(m.outputs.load)
        assert default.axes == (TIME_AXIS, ZONE)
        ordered = result.expected(m.outputs.load, axes=(ZONE, "time"))
        assert ordered.axes == (ZONE, TIME_AXIS)
        np.testing.assert_array_equal(ordered.values, default.values.T)

    def test_axes_must_be_all_the_axes(self):
        """A missing axis is named, with the reader in the message."""
        m, result = _roads()
        with pytest.raises(ValueError, match=r"EvaluationResult.expected: axes=: .*missing \['zone'\]"):
            result.expected(m.outputs.load, axes=("time",))

    def test_ensemble_is_contracted_and_parameters_kept(self):
        """ENSEMBLE axes are averaged with the weights; PARAMETER axes stay, at full size."""
        m, scale, result = _noisy()
        expected = result.expected(m.outputs.scaled)
        assert [ax.role for ax in expected.axes] == [PARAMETER, DOMAIN]
        draws = np.asarray(result[scale]).reshape(-1)
        np.testing.assert_allclose(expected.isel(factor=1).values, 10.0 * draws.mean() * np.array([1.0, 2.0, 3.0]))

    def test_deterministic_index_is_read_at_the_parameter_size(self):
        """An index not depending on the swept parameter is repeated along its axis."""
        m, _, result = _noisy()
        grid = result.expected(m.outputs.grid)
        assert grid.values.shape == (2, 3)
        np.testing.assert_array_equal(grid.values, [[1.0, 2.0, 3.0]] * 2)


class TestSamples:
    """The ensemble's samples, named and at full size."""

    def test_ensemble_axis_is_kept(self):
        """Samples keep the ENSEMBLE axis; their weighted average is the expected value (#276)."""
        m, _, result = _noisy(size=5)
        samples = result.samples(m.outputs.scaled)
        assert [ax.role for ax in samples.axes] == [PARAMETER, ENSEMBLE, DOMAIN]
        assert samples.values.shape == (2, 5, 3)
        ((_, weights),) = result.factorized_weights.items()
        np.testing.assert_allclose(
            np.average(samples.values, axis=1, weights=weights), result.expected(m.outputs.scaled).values
        )

    def test_deterministic_index_is_repeated_along_the_ensemble(self):
        """A deterministic index has the ensemble's full size along the ENSEMBLE axis."""
        m, _, result = _noisy(size=5)
        grid = result.samples(m.outputs.grid)
        assert grid.values.shape == (2, 5, 3)
        np.testing.assert_array_equal(grid.isel(factor=1).values, [[1.0, 2.0, 3.0]] * 5)

    def test_stray_domain_axes_are_dropped(self):
        """DOMAIN axes the index does not carry are dropped."""
        _, scale, result = _noisy(size=5)
        samples = result.samples(scale)
        assert [ax.role for ax in samples.axes] == [PARAMETER, ENSEMBLE]

    def test_axes_sets_the_order(self):
        """Axes= works as for expected()."""
        m, _, result = _noisy(size=5)
        ensemble_axis = next(ax for ax in result.layout.axes if ax.role == ENSEMBLE)
        samples = result.samples(m.outputs.scaled, axes=("zone", ensemble_axis, "factor"))
        assert samples.values.shape == (3, 5, 2)

    def test_axes_errors_name_the_reader(self):
        """An unknown axis is named, with the reader in the message."""
        m, _, result = _noisy(size=5)
        with pytest.raises(ValueError, match=r"EvaluationResult.samples: axes=: .*unknown \['x'\]"):
            result.samples(m.outputs.grid, axes=("x",))

    def test_value_varying_along_an_undeclared_axis_raises(self):
        """A value varying along a DOMAIN axis its index does not carry is reported."""
        x = Axis("x", DOMAIN)
        idx = Index("v", 1.0)
        layout = AxisLayout([(x, 3)])
        result = EvaluationResult(executor.State({idx.node: np.arange(3.0)}), layout, {})
        with pytest.raises(ValueError, match=r"DOMAIN axis 'x' is not carried .* varies along it \(size 3\)"):
            result.samples(idx)

    def test_incompatible_shape_raises(self):
        """A stored value that does not fit the layout is reported."""
        idx = Index("v", 1.0)
        layout = AxisLayout([(Axis("x", DOMAIN), 3)])
        result = EvaluationResult(executor.State({idx.node: np.zeros((2, 3))}), layout, {})
        with pytest.raises(ValueError, match=r"EvaluationResult.samples: shape \(2, 3\) is incompatible"):
            result.samples(idx)


class TestDeprecatedReaders:
    """expected_value() and labeled() still work, with a DeprecationWarning naming the replacement."""

    def test_expected_value_warns_and_returns_the_stored_size(self):
        """expected_value() returns the array at the size it is stored."""
        m, result = _roads()
        match = r"expected_value\(\) is deprecated.*Use expected\(index\).values instead"
        with pytest.warns(DeprecationWarning, match=match):
            values = result.expected_value(m.outputs.hourly_capacity)
        assert values.shape == (1, 3)

    def test_labeled_warns_and_returns_expected(self):
        """labeled() returns what expected() returns."""
        m, result = _roads()
        with pytest.warns(DeprecationWarning, match=r"labeled\(\) is deprecated.*Use expected\(index\) instead"):
            labeled = result.labeled(m.outputs.hourly_capacity)
        np.testing.assert_array_equal(labeled.values, result.expected(m.outputs.hourly_capacity).values)
