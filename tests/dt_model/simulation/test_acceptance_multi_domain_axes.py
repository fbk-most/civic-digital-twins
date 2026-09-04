"""Acceptance tests for models carrying several DOMAIN axes.

Covers:
- Axis-order independence: a model built with axes declared in one order
  and the same model built with them declared in another order produce
  identical results — operations align and reduce by axis *name*, never
  by position.
- Time-only parity: existing time-only models are unaffected.
- 2-D spatial + time end-to-end: a (TIME, X, Y) model can be authored and
  evaluated, including reduction over spatial axes.
- Self-describing results: a caller can map axis name to position for
  every returned array, raw and ensemble-marginalized alike.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    ConstIndex,
    DistributionEnsemble,
    DistributionIndex,
    Evaluation,
    Index,
    Model,
    Scenario,
    TimeseriesIndex,
    define,
    inputs,
    outputs,
)
from civic_digital_twins.dt_model.axes import TIME_AXIS, DomainAxis, SpaceType

X = DomainAxis("x", type=SpaceType(spacing=1.0))
Y = DomainAxis("y", type=SpaceType(spacing=1.0))


def _evaluate(model):
    return Evaluation(Scenario(model)).evaluate(ensemble=None)


# ---------------------------------------------------------------------------
# Axis-order independence
# ---------------------------------------------------------------------------


def test_axis_order_independence_across_three_domain_axes():
    """The same (time, x, y) data, declared in two different axis orders, evaluates identically.

    T=2, X=3, Y=4 (all distinct sizes, so a position/name mixup would show
    up as a shape mismatch or misplaced values, not just a coincidental
    match). One model declares axes=(TIME_AXIS, X, Y) with data laid out
    (T, X, Y); the other declares axes=(Y, X, TIME_AXIS) with the *same*
    logical data transposed to (Y, X, T). Both must produce the same
    canonically-ordered (time, x, y) result.
    """
    data_txy = np.arange(24.0).reshape(2, 3, 4)

    def _make(axes, data):
        @define("Field")
        class FieldModel(Model):
            @inputs
            class Inputs:
                pass

            @outputs
            class Outputs:
                field: Index

            def compute(self, inputs):
                f = ConstIndex("field", data, axes=axes)
                return FieldModel.Outputs(field=Index("field_out", f))

        return FieldModel(inputs=FieldModel.Inputs())

    m1 = _make((TIME_AXIS, X, Y), data_txy)
    m2 = _make((Y, X, TIME_AXIS), data_txy.transpose(2, 1, 0))

    r1 = _evaluate(m1)
    r2 = _evaluate(m2)

    assert r1.layout.entries == ((TIME_AXIS, 2), (X, 3), (Y, 4))
    assert r2.layout.entries == ((TIME_AXIS, 2), (X, 3), (Y, 4))
    np.testing.assert_array_equal(r1[m1.outputs.field], data_txy)
    np.testing.assert_array_equal(r2[m2.outputs.field], data_txy)


# ---------------------------------------------------------------------------
# Time-only parity
# ---------------------------------------------------------------------------


def test_time_only_model_unaffected_by_multi_domain_support():
    """A plain time-only model evaluates exactly as it did before multi-domain axes existed."""

    @define("TimeOnly")
    class TimeOnlyModel(Model):
        @inputs
        class Inputs:
            demand: Index

        @outputs
        class Outputs:
            total: Index

        def compute(self, inputs):
            return TimeOnlyModel.Outputs(total=Index("total", inputs.demand.sum(axis=TIME_AXIS)))

    demand = TimeseriesIndex("demand", np.array([1.0, 2.0, 3.0, 4.0]))
    m = TimeOnlyModel(inputs=TimeOnlyModel.Inputs(demand=demand))
    result = _evaluate(m)

    # sum(axis=TIME_AXIS) keeps the reduced axis at size 1 (broadcast
    # convention) rather than removing it from output_axes, so the
    # marginalized layout for "total" is still empty: no DOMAIN axis
    # survives in its output_axes for layout_of to keep.
    assert result.layout_of(m.outputs.total).entries == ()
    assert np.isclose(float(result[m.outputs.total].ravel()[0]), 10.0)


# ---------------------------------------------------------------------------
# 2-D spatial + time end-to-end
# ---------------------------------------------------------------------------


def test_time_and_two_space_axes_end_to_end():
    """A (time, x, y) field authored with typed SpaceType axes evaluates and reduces correctly.

    A temperature-like field varying over time and a 2-D grid is reduced
    over one spatial axis (a column mean), then over both (a spatial
    average per time step), each reduction targeting the axis it names
    regardless of declaration order.
    """

    @define("Temperature")
    class TemperatureModel(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            field: Index
            col_mean: Index
            space_avg: Index

        def compute(self, inputs):
            data = np.arange(24.0).reshape(2, 3, 4)  # (time=2, x=3, y=4)
            temperature = ConstIndex("temperature", data, axes=(TIME_AXIS, X, Y))
            col_mean = Index("col_mean", temperature.mean(axis=Y))  # (time, x)
            space_avg = Index("space_avg", col_mean.mean(axis=X))  # (time,)
            return TemperatureModel.Outputs(
                field=temperature,
                col_mean=col_mean,
                space_avg=space_avg,
            )

    m = TemperatureModel(inputs=TemperatureModel.Inputs())
    result = _evaluate(m)

    data = np.arange(24.0).reshape(2, 3, 4)
    assert result.layout.entries == ((TIME_AXIS, 2), (X, 3), (Y, 4))
    np.testing.assert_array_equal(result[m.outputs.field], data)

    expected_col_mean = data.mean(axis=2, keepdims=True)  # (time, x, 1)
    np.testing.assert_allclose(result[m.outputs.col_mean], expected_col_mean)

    expected_space_avg = data.mean(axis=(1, 2), keepdims=True)  # (time, 1, 1)
    np.testing.assert_allclose(result[m.outputs.space_avg], expected_space_avg)


def test_lower_rank_operand_broadcasts_against_full_spacetime_block():
    """A per-column (x,) weight combines correctly with a (time, x, y) field."""

    @define("Weighted")
    class WeightedModel(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            weighted: Index

        def compute(self, inputs):
            field = ConstIndex("field", np.ones((2, 3, 4)), axes=(TIME_AXIS, X, Y))
            wx = ConstIndex("wx", np.array([10.0, 20.0, 30.0]), axes=(X,))
            return WeightedModel.Outputs(weighted=Index("weighted", field * wx))

    m = WeightedModel(inputs=WeightedModel.Inputs())
    result = _evaluate(m)

    expected = np.ones((2, 3, 4)) * np.array([10.0, 20.0, 30.0]).reshape(1, 3, 1)
    np.testing.assert_array_equal(result[m.outputs.weighted], expected)


# ---------------------------------------------------------------------------
# Self-describing results
# ---------------------------------------------------------------------------


def test_layout_of_matches_the_marginalized_array_two_outputs_carry_different_axes():
    """result.layout_of(idx) names every dimension of expected_value(idx), per output.

    One model, one ENSEMBLE axis, two outputs that carry different DOMAIN
    axes: "field" keeps (time, x, y), "col_mean" drops y. result.layout (the
    *raw*, pre-marginalization layout) is the same object for both — it
    cannot tell them apart. result.layout_of(idx) can: it always drops the
    ENSEMBLE axis (contracted by expected_value) and keeps only the DOMAIN
    axes the given index actually carries.
    """

    @define("Marginalized")
    class MarginalizedModel(Model):
        @inputs
        class Inputs:
            noise: Index

        @outputs
        class Outputs:
            field: Index
            col_mean: Index

        def compute(self, inputs):
            data = np.arange(24.0).reshape(2, 3, 4)  # (time=2, x=3, y=4)
            temperature = ConstIndex("temperature", data, axes=(TIME_AXIS, X, Y))
            field = Index("field", temperature * inputs.noise)
            col_mean = Index("col_mean", field.mean(axis=Y))
            return MarginalizedModel.Outputs(field=field, col_mean=col_mean)

    # scale=0.0: a deterministic "ensemble" isolates the axis-bookkeeping
    # behaviour under test from sampling noise.
    noise = DistributionIndex("noise", stats.norm, {"loc": 1.0, "scale": 0.0})
    m = MarginalizedModel(inputs=MarginalizedModel.Inputs(noise=noise))
    scenario = Scenario(m)
    ensemble = DistributionEnsemble(scenario, size=5, rng=np.random.default_rng(0))
    result = Evaluation(scenario).evaluate(ensemble=ensemble)

    assert result.layout_of(m.outputs.field).entries == ((TIME_AXIS, 2), (X, 3), (Y, 4))
    assert result.layout_of(m.outputs.col_mean).entries == ((TIME_AXIS, 2), (X, 3))

    # The per-output layout always matches the array expected_value actually returns.
    assert result.layout_of(m.outputs.field).full_shape == result.expected_value(m.outputs.field).shape
    assert result.layout_of(m.outputs.col_mean).full_shape == result.expected_value(m.outputs.col_mean).shape
