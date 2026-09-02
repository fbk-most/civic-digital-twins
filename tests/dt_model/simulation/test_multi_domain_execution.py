"""End-to-end evaluation of models carrying several DOMAIN axes."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model import (
    ConstIndex,
    Evaluation,
    Index,
    Model,
    Scenario,
    define,
    inputs,
    outputs,
)
from civic_digital_twins.dt_model.axes import DOMAIN, TIME_AXIS, Axis
from civic_digital_twins.dt_model.engine.numpybackend.executor import align_to_domain_block

X = Axis("x", DOMAIN)
Y = Axis("y", DOMAIN)


def _evaluate(model):
    return Evaluation(Scenario(model)).evaluate(ensemble=None)


def test_two_domain_axes_evaluate_and_name_every_dimension():
    """A (x, y) model evaluates, and the result layout names both axes with their sizes."""

    @define("Field")
    class FieldModel(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            scaled: Index

        def compute(self, inputs):
            field = ConstIndex("field", np.arange(6.0).reshape(2, 3), axes=(X, Y))
            return FieldModel.Outputs(scaled=Index("scaled", field * 2.0))

    m = FieldModel(inputs=FieldModel.Inputs())
    result = _evaluate(m)
    assert result.layout.entries == ((X, 2), (Y, 3))
    assert np.array_equal(result[m.outputs.scaled], np.arange(6.0).reshape(2, 3) * 2.0)


def test_reduction_targets_the_axis_it_names():
    """With two DOMAIN axes, reducing y collapses y — not simply the last dimension."""

    @define("Reduce")
    class ReduceModel(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            over_x: Index
            over_y: Index

        def compute(self, inputs):
            field = ConstIndex("field", np.arange(6.0).reshape(2, 3), axes=(X, Y))
            return ReduceModel.Outputs(
                over_x=Index("over_x", field.sum(axis=X)),
                over_y=Index("over_y", field.sum(axis=Y)),
            )

    m = ReduceModel(inputs=ReduceModel.Inputs())
    result = _evaluate(m)
    assert np.array_equal(result[m.outputs.over_x].ravel(), [3.0, 5.0, 7.0])
    assert np.array_equal(result[m.outputs.over_y].ravel(), [3.0, 12.0])


def test_lower_rank_operand_aligns_to_its_own_axis():
    """An x-only operand broadcasts along x, not onto the trailing axis.

    This is the case a single ``has_timeseries`` boolean could not express: an
    ``(x,)`` value is right-aligned by numpy onto ``y`` unless it is padded to
    the full domain block first.
    """

    @define("Weighted")
    class WeightedModel(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            weighted: Index

        def compute(self, inputs):
            field = ConstIndex("field", np.ones((2, 3)), axes=(X, Y))
            wx = ConstIndex("wx", np.array([10.0, 20.0]), axes=(X,))
            return WeightedModel.Outputs(weighted=Index("weighted", field * wx))

    m = WeightedModel(inputs=WeightedModel.Inputs())
    result = _evaluate(m)
    assert np.array_equal(result[m.outputs.weighted], [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])


def test_non_canonical_declaration_order_is_permuted_into_the_layout():
    """Declaring axes as (y, x) still yields a canonically ordered (x, y) result."""

    @define("Transposed")
    class TransposedModel(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            out: Index

        def compute(self, inputs):
            # Array shape matches the declared (y, x) order: 3 rows of y, 2 of x.
            tr = ConstIndex("tr", np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), axes=(Y, X))
            return TransposedModel.Outputs(out=Index("out", tr * 1.0))

    m = TransposedModel(inputs=TransposedModel.Inputs())
    result = _evaluate(m)
    assert result.layout.entries == ((X, 2), (Y, 3))
    assert np.array_equal(result[m.outputs.out], [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])


def test_time_combines_with_a_spatial_axis():
    """Time is not special: it takes its canonical slot beside another DOMAIN axis."""

    @define("SpaceTime")
    class SpaceTimeModel(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            out: Index

        def compute(self, inputs):
            ts = ConstIndex("ts", np.array([1.0, 2.0]), axes=(TIME_AXIS,))
            space = ConstIndex("space", np.array([10.0, 20.0, 30.0]), axes=(X,))
            return SpaceTimeModel.Outputs(out=Index("out", ts * space, axes=(TIME_AXIS, X)))

    m = SpaceTimeModel(inputs=SpaceTimeModel.Inputs())
    result = _evaluate(m)
    # Canonical order is by axis name: "time" < "x".
    assert result.layout.entries == ((TIME_AXIS, 2), (X, 3))
    assert np.array_equal(result[m.outputs.out], [[10.0, 20.0, 30.0], [20.0, 40.0, 60.0]])


class TestAlignToDomainBlock:
    """Unit coverage for the alignment primitive the executor applies to leaves."""

    def test_absent_axes_become_singletons(self):
        """A value carrying only x is padded to the full (x, y) block."""
        arr = np.array([1.0, 2.0])
        assert align_to_domain_block(arr, (X,), (X, Y)).shape == (2, 1)

    def test_present_axes_keep_their_sizes(self):
        """A value carrying both axes is unchanged when already canonical."""
        arr = np.zeros((2, 3))
        assert align_to_domain_block(arr, (X, Y), (X, Y)).shape == (2, 3)

    def test_leading_dimensions_are_preserved(self):
        """PARAMETER/ENSEMBLE dimensions ahead of the domain block are untouched."""
        arr = np.zeros((5, 2))
        assert align_to_domain_block(arr, (X,), (X, Y)).shape == (5, 2, 1)

    def test_no_domain_axes_is_a_no_op(self):
        """A scalar evaluation leaves values alone."""
        arr = np.zeros((5,))
        assert align_to_domain_block(arr, (), ()).shape == (5,)

    def test_domain_carrying_node_in_a_scalar_evaluation_is_rejected(self):
        """An evaluation declaring no DOMAIN axes cannot place a domain-carrying value.

        This is the same failure as an unknown axis, and must not be masked by
        the empty-``domain_axes`` shortcut: silently returning the value would
        leave an unpadded dimension for broadcasting to misalign later.
        """
        from civic_digital_twins.dt_model.engine.numpybackend.executor import UnsupportedOperation

        with pytest.raises(UnsupportedOperation, match="not among"):
            align_to_domain_block(np.zeros((3,)), (X,), ())

    def test_unknown_axis_is_rejected(self):
        """An axis the evaluation does not carry cannot be placed."""
        from civic_digital_twins.dt_model.engine.numpybackend.executor import UnsupportedOperation

        with pytest.raises(UnsupportedOperation, match="not among"):
            align_to_domain_block(np.zeros((2,)), (Y,), (X,))


def test_array_placeholder_without_a_value_names_its_axes():
    """The missing-value diagnostic identifies which domain-carrying placeholder is unset."""
    from civic_digital_twins.dt_model.engine.frontend import graph, linearize
    from civic_digital_twins.dt_model.engine.numpybackend import executor

    node = graph.array_placeholder("field", (X, Y))
    state = executor.State({}, domain_axes=(X, Y))
    with pytest.raises(executor.PlaceholderValueNotProvided, match=r"array placeholder 'field'.*\['x', 'y'\]"):
        executor.evaluate_nodes(state, *linearize.forest(node))


# ---------------------------------------------------------------------------
# Injected (placeholder-backed) domain-carrying indexes
# ---------------------------------------------------------------------------


@define("Injected")
class _InjectedModel(Model):
    @inputs
    class Inputs:
        field: Index

    @outputs
    class Outputs:
        doubled: Index
        over_x: Index

    def compute(self, inputs):
        doubled = Index("doubled", inputs.field * 2.0)
        return _InjectedModel.Outputs(doubled=doubled, over_x=Index("over_x", doubled.sum(axis=X)))


def _injected_model():
    field = Index("field", np.arange(6.0).reshape(2, 3), axes=(X, Y))
    return field, _InjectedModel(inputs=_InjectedModel.Inputs(field=field))


def test_placeholder_backed_domain_index_uses_its_default():
    """A domain-carrying placeholder evaluates from the default supplied at construction."""
    _, m = _injected_model()
    result = _evaluate(m)
    assert result.layout.entries == ((X, 2), (Y, 3))
    assert np.array_equal(result[m.outputs.doubled], np.arange(6.0).reshape(2, 3) * 2.0)


def test_scenario_override_of_a_domain_carrying_index():
    """A Scenario override replaces the whole field and flows through the formula."""
    field, m = _injected_model()
    override = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    result = Evaluation(Scenario(m, overrides={field: override})).evaluate(ensemble=None)
    assert np.array_equal(result[m.outputs.doubled], override * 2.0)
    assert np.array_equal(result[m.outputs.over_x].ravel(), [6.0, 6.0, 6.0])


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(np.array([1.0, 2.0]), id="too-few-dimensions"),
        pytest.param(np.zeros((2, 3, 1)), id="too-many-dimensions"),
        pytest.param(5.0, id="not-an-array"),
    ],
)
def test_override_rank_must_match_declared_axes(bad):
    """Each declared axis is zipped against one dimension, so the rank must agree."""
    field, m = _injected_model()
    with pytest.raises(TypeError, match=r"must be a 2-D ndarray over axes \(x, y\)"):
        Scenario(m, overrides={field: bad})


def test_timeseries_override_message_still_describes_the_time_axis():
    """The generalized gate keeps naming the axis a TimeseriesIndex declares."""

    @define("TsOnly")
    class TsModel(Model):
        @inputs
        class Inputs:
            ts: Index

        @outputs
        class Outputs:
            out: Index

        def compute(self, inputs):
            return TsModel.Outputs(out=Index("out", inputs.ts * 1.0))

    from civic_digital_twins.dt_model import TimeseriesIndex

    ts = TimeseriesIndex("ts", np.array([1.0, 2.0]))
    m = TsModel(inputs=TsModel.Inputs(ts=ts))
    with pytest.raises(TypeError, match=r"must be a 1-D ndarray over axes \(time\)"):
        Scenario(m, overrides={ts: np.zeros((2, 2))})


def test_injected_lower_rank_value_aligns_to_its_own_axis():
    """An injected x-only value broadcasts along x, not onto the trailing axis."""

    @define("MixedInjected")
    class MixedModel(Model):
        @inputs
        class Inputs:
            field: Index
            wx: Index

        @outputs
        class Outputs:
            out: Index

        def compute(self, inputs):
            return MixedModel.Outputs(out=Index("out", inputs.field * inputs.wx))

    field = Index("field", np.ones((2, 3)), axes=(X, Y))
    wx = Index("wx", np.array([10.0, 20.0]), axes=(X,))
    m = MixedModel(inputs=MixedModel.Inputs(field=field, wx=wx))
    result = _evaluate(m)
    assert np.array_equal(result[m.outputs.out], [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
