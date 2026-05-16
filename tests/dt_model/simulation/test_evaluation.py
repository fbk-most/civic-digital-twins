"""Unit tests for Evaluation.evaluate() — both 1-D and axes (grid) modes."""

# SPDX-License-Identifier: Apache-2.0

# NOTE: scenario assignments dicts must be explicitly annotated as
# ``dict[GenericIndex, Any]`` rather than letting Pyright infer
# ``dict[Index, float]``.  dict is invariant in its key type, so
# ``dict[Index, float]`` is not assignable to ``dict[GenericIndex, Any]``
# even though ``Index`` extends ``GenericIndex``.

from typing import Any

import numpy as np
import pytest

from civic_digital_twins.dt_model.model.index import ConstIndex, GenericIndex, Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.ensemble import WeightedScenario
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from civic_digital_twins.dt_model.simulation.scenario import Scenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(*indexes):
    """Wrap indexes in a named model."""
    return Model("test", list(indexes))


# ---------------------------------------------------------------------------
# 1-D batch mode
# ---------------------------------------------------------------------------


def test_1d_single_scenario_constant_model():
    """A fully-concrete model evaluates with an empty scenario."""
    I_a = Index("a", 3.0)
    I_b = Index("b", 4.0)
    I_result = Index("result", I_a.node + I_b.node)
    model = _make_model(I_a, I_b, I_result)

    # No abstract indexes → scenarios list is empty; still evaluates constants.
    result = Evaluation(model).evaluate([])
    assert np.isclose(result[I_result], 7.0)


def test_1d_single_scenario_placeholder():
    """Single scenario with one placeholder index."""
    I_x = Index("x", None)
    I_scale = Index("scale", 2.0)
    I_result = Index("result", I_scale.node * I_x.node)
    model = _make_model(I_x, I_scale, I_result)

    a: dict[GenericIndex, Any] = {I_x: 5.0}
    scenarios: list[WeightedScenario] = [(1.0, a)]
    result = Evaluation(model).evaluate(scenarios)
    # Shape (S,): S=1 scenario; value = 2 * 5 = 10
    # (No trailing DOMAIN placeholder in non-timeseries models after bug fix #155.)
    assert result[I_result].shape == (1,)
    assert np.isclose(result[I_result][0], 10.0)


def test_1d_multiple_scenarios():
    """Multiple scenarios are stacked; result has one entry per scenario."""
    I_x = Index("x", None)
    I_result = Index("result", I_x.node * I_x.node)
    model = _make_model(I_x, I_result)

    a0: dict[GenericIndex, Any] = {I_x: 2.0}
    a1: dict[GenericIndex, Any] = {I_x: 3.0}
    scenarios: list[WeightedScenario] = [(0.5, a0), (0.5, a1)]
    result = Evaluation(model).evaluate(scenarios)
    assert result[I_result].shape == (2,)
    assert np.isclose(result[I_result][0], 4.0)
    assert np.isclose(result[I_result][1], 9.0)


def test_1d_raises_on_unresolved_abstract_index():
    """ValueError is raised when a scenario is missing an abstract index."""
    I_x = Index("x", None)
    I_y = Index("y", None)
    model = _make_model(I_x, I_y)

    a: dict[GenericIndex, Any] = {I_x: 1.0}
    with pytest.raises(ValueError, match="abstract index"):
        Evaluation(model).evaluate([(1.0, a)])


# ---------------------------------------------------------------------------
# axes (grid) mode — shape checks
# ---------------------------------------------------------------------------


def test_axes_single_axis_result_shape():
    """Single axis index produces result shape (N, 1)."""
    I_x = Index("x", None)
    model = _make_model(I_x)
    xs = np.array([1.0, 2.0, 3.0])

    result = Evaluation(model).evaluate([(1.0, {})], parameters={I_x: xs})
    assert result[I_x].shape == (3, 1)


def test_axes_two_axes_result_shape():
    """Two axis indexes produce result shape (N0, N1, 1)."""
    I_x = Index("x", None)
    I_y = Index("y", None)
    model = _make_model(I_x, I_y)
    xs = np.array([1.0, 2.0])
    ys = np.array([10.0, 20.0, 30.0])

    result = Evaluation(model).evaluate([(1.0, {})], parameters={I_x: xs, I_y: ys})
    assert result[I_x].shape == (2, 1, 1)
    assert result[I_y].shape == (1, 3, 1)


def test_axes_non_axis_abstract_has_shape_1_1_s():
    """A non-axis abstract index has shape (1, …, 1, S)."""
    I_x = Index("x", None)
    I_factor = Index("factor", None)
    model = _make_model(I_x, I_factor)
    xs = np.array([1.0, 2.0, 3.0])

    a0: dict[GenericIndex, Any] = {I_factor: 1.0}
    a1: dict[GenericIndex, Any] = {I_factor: 2.0}
    scenarios: list[WeightedScenario] = [(0.5, a0), (0.5, a1)]
    result = Evaluation(model).evaluate(scenarios, parameters={I_x: xs})
    # Non-axis abstract: shape (1, S) = (1, 2)
    assert result[I_factor].shape == (1, 2)


# ---------------------------------------------------------------------------
# axes (grid) mode — values
# ---------------------------------------------------------------------------


def test_axes_single_axis_formula_values():
    """Formula with a single axis index evaluates on the full grid."""
    I_x = Index("x", None)
    I_scale = Index("scale", 3.0)
    I_result = Index("result", I_scale.node * I_x.node)
    model = _make_model(I_x, I_scale, I_result)
    xs = np.array([1.0, 2.0, 4.0])

    result = Evaluation(model).evaluate([(1.0, {})], parameters={I_x: xs})
    # shape (3, 1); marginalize: tensordot(..., [1.0], axes=([-1],[0])) → (3,)
    marginalised = result.expected_value(I_result)
    assert np.allclose(marginalised, [3.0, 6.0, 12.0])


def test_axes_two_axes_additive_formula():
    """Sum formula over two axes produces the correct (N0, N1, S) array."""
    I_x = Index("x", None)
    I_y = Index("y", None)
    I_result = Index("result", I_x.node + I_y.node)
    model = _make_model(I_x, I_y, I_result)
    xs = np.array([1.0, 2.0])
    ys = np.array([10.0, 20.0, 30.0])

    result = Evaluation(model).evaluate([(1.0, {})], parameters={I_x: xs, I_y: ys})
    marginalised = result.expected_value(I_result)
    # result[i, j] = xs[i] + ys[j]
    expected = xs[:, None] + ys[None, :]
    assert np.allclose(marginalised, expected)


def test_axes_non_axis_factor_marginalised_correctly():
    """Weighted marginalisation over a non-axis index gives the correct mean."""
    I_x = Index("x", None)
    I_factor = Index("factor", None)
    I_result = Index("result", I_x.node * I_factor.node)
    model = _make_model(I_x, I_factor, I_result)
    xs = np.array([1.0, 2.0, 3.0])
    # Two equiprobable scenarios: factor=1 and factor=3 → mean factor=2
    a0: dict[GenericIndex, Any] = {I_factor: 1.0}
    a1: dict[GenericIndex, Any] = {I_factor: 3.0}
    scenarios: list[WeightedScenario] = [(0.5, a0), (0.5, a1)]

    result = Evaluation(model).evaluate(scenarios, parameters={I_x: xs})
    marginalised = result.expected_value(I_result)
    # result[i] = xs[i] * mean_factor = xs[i] * 2
    assert np.allclose(marginalised, [2.0, 4.0, 6.0])


# ---------------------------------------------------------------------------
# axes (grid) mode — error handling
# ---------------------------------------------------------------------------


def test_axes_raises_on_unresolved_non_axis_abstract():
    """ValueError when a non-axis abstract index is missing from a scenario."""
    I_x = Index("x", None)
    I_missing = Index("missing", None)
    model = _make_model(I_x, I_missing)

    with pytest.raises(ValueError, match="abstract index"):
        Evaluation(model).evaluate([(1.0, {})], parameters={I_x: np.array([1.0])})


def test_axes_axis_index_not_required_in_scenario():
    """Axis indexes do not need to appear in the scenario assignments."""
    I_x = Index("x", None)
    model = _make_model(I_x)

    # Should not raise — I_x is an axis, not required in scenario dict.
    result = Evaluation(model).evaluate([(1.0, {})], parameters={I_x: np.array([5.0, 10.0])})
    assert result[I_x].shape == (2, 1)


# ---------------------------------------------------------------------------
# EvaluationResult properties
# ---------------------------------------------------------------------------


def test_evaluation_result_weights_property():
    """EvaluationResult.weights returns the scenario weight array."""
    I_x = Index("x", None)
    model = _make_model(I_x)

    a0: dict[GenericIndex, Any] = {I_x: 1.0}
    a1: dict[GenericIndex, Any] = {I_x: 2.0}
    scenarios: list[WeightedScenario] = [(0.3, a0), (0.7, a1)]
    result = Evaluation(model).evaluate(scenarios)

    weights = result.weights
    assert weights.shape == (2,)
    assert np.isclose(weights[0], 0.3)
    assert np.isclose(weights[1], 0.7)


def test_evaluation_result_parameter_values_property():
    """EvaluationResult.parameter_values returns the dict passed to evaluate()."""
    I_x = Index("x", None)
    model = _make_model(I_x)
    xs = np.array([1.0, 2.0, 3.0])

    result = Evaluation(model).evaluate([(1.0, {})], parameters={I_x: xs})
    pv = result.parameter_values
    assert I_x in pv
    assert np.array_equal(pv[I_x], xs)


def test_evaluation_result_parameter_values_empty_in_1d_mode():
    """EvaluationResult.parameter_values is empty when no parameters are passed."""
    I_x = Index("x", 1.0)
    model = _make_model(I_x)

    result = Evaluation(model).evaluate([])
    assert result.parameter_values == {}


def test_value_constant_index():
    """Marginalize on an index with no abstract dependency returns the constant."""
    I_c = Index("c", 42.0)
    model = _make_model(I_c)
    result = Evaluation(model).evaluate([(1.0, {})])
    marginalised = result.expected_value(I_c)
    assert float(marginalised) == pytest.approx(42.0)


def test_value_1d_squeeze_scalar():
    """Marginalize of a pure-ENSEMBLE scalar result is a 0-d scalar array."""
    from scipy import stats

    from civic_digital_twins.dt_model.model.index import DistributionIndex
    from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble

    I_x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
    I_result = Index("result", I_x * I_x)
    model = _make_model(I_x, I_result)

    ensemble = DistributionEnsemble(model, size=50)
    result = Evaluation(model).evaluate(ensemble)
    marginalised = result.expected_value(I_result)
    # ENSEMBLE axis contracted, DOMAIN placeholder squeezed → 0-d scalar.
    assert np.ndim(marginalised) == 0


# ---------------------------------------------------------------------------
# DistributionEnsemble — error and rng paths
# ---------------------------------------------------------------------------


def test_result_axes_deprecated_property():
    """EvaluationResult.axes emits DeprecationWarning and returns parameter_values."""
    import warnings

    I_x = Index("x", None)
    model = _make_model(I_x)
    xs = np.array([1.0, 2.0])

    result = Evaluation(model).evaluate([(1.0, {})], parameters={I_x: xs})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        axes = result.axes
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("result.axes" in str(w.message) for w in deprecations)
    assert I_x in axes


def test_result_parameter_values_for():
    """EvaluationResult.parameter_values_for() returns the array for a given index."""
    I_x = Index("x", None)
    model = _make_model(I_x)
    xs = np.array([1.0, 2.0, 3.0])

    result = Evaluation(model).evaluate([(1.0, {})], parameters={I_x: xs})
    assert np.array_equal(result.parameter_values_for(I_x), xs)


def test_result_full_shape_no_axes():
    """EvaluationResult.full_shape is () when there are no axes at all."""
    I_c = Index("c", 5.0)
    model = _make_model(I_c)

    result = Evaluation(model).evaluate(ensemble=None)
    assert result.full_shape == ()


def test_result_weights_no_ensemble():
    """EvaluationResult.weights returns empty array when there are no ENSEMBLE axes."""
    I_c = Index("c", 5.0)
    model = _make_model(I_c)

    result = Evaluation(model).evaluate(ensemble=None)
    assert result.weights.shape == (0,)


def test_evaluate_raises_when_both_scenarios_and_ensemble():
    """TypeError when both 'scenarios' and 'ensemble=' are supplied."""
    from scipy import stats

    from civic_digital_twins.dt_model.model.index import DistributionIndex
    from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble

    I_x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 1.0})
    model = _make_model(I_x)
    ens = DistributionEnsemble(model, size=3)

    scenario: dict[GenericIndex, Any] = {I_x: 0.5}
    legacy: list[WeightedScenario] = [(1.0, scenario)]
    with pytest.raises(TypeError, match="both"):
        Evaluation(model).evaluate(legacy, ensemble=ens)


def test_evaluate_raises_when_both_axes_and_parameters():
    """TypeError when both 'axes=' and 'parameters=' are supplied."""
    import warnings

    I_x = Index("x", None)
    model = _make_model(I_x)
    xs = np.array([1.0, 2.0])

    with pytest.raises(TypeError, match="both"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Evaluation(model).evaluate([(1.0, {})], axes={I_x: xs}, parameters={I_x: xs})


def test_evaluate_deprecated_axes_kwarg():
    """Passing axes= emits DeprecationWarning and is equivalent to parameters=."""
    import warnings

    I_x = Index("x", None)
    model = _make_model(I_x)
    xs = np.array([1.0, 2.0])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Evaluation(model).evaluate([(1.0, {})], axes={I_x: xs})

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("axes" in str(w.message) for w in deprecations)
    assert result[I_x].shape[0] == 2


def test_distribution_ensemble_raises_for_non_distribution_abstract():
    """DistributionEnsemble raises ValueError when an abstract index has no distribution."""
    from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble

    I_placeholder = Index("p", None)  # abstract but not distribution-backed
    model = _make_model(I_placeholder)

    with pytest.raises(ValueError, match="unsupported indexes"):
        DistributionEnsemble(model, size=10)


def test_distribution_ensemble_iteration_without_rng():
    """DistributionEnsemble.__iter__ works without an explicit rng (rng=None path)."""
    from scipy import stats

    from civic_digital_twins.dt_model.model.index import DistributionIndex
    from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble

    I_x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 1.0})
    model = _make_model(I_x)

    scenarios = list(DistributionEnsemble(model, size=5))
    assert len(scenarios) == 5
    w, a = scenarios[0]
    assert np.isclose(w, 1.0 / 5)
    assert I_x in a


def test_distribution_ensemble_with_rng_is_reproducible():
    """Passing an rng to DistributionEnsemble produces reproducible samples."""
    from scipy import stats

    from civic_digital_twins.dt_model.model.index import DistributionIndex
    from civic_digital_twins.dt_model.simulation.ensemble import DistributionEnsemble

    I_x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 1.0})
    model = _make_model(I_x)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    scenarios1 = list(DistributionEnsemble(model, size=5, rng=rng1))
    scenarios2 = list(DistributionEnsemble(model, size=5, rng=rng2))

    for (w1, a1), (w2, a2) in zip(scenarios1, scenarios2):
        assert w1 == w2
        assert np.array_equal(a1[I_x], a2[I_x])


# ---------------------------------------------------------------------------
# functions= and NumpyBackend (Step 2 / #162)
# ---------------------------------------------------------------------------


def test_evaluate_functions_numpy_backend_adapt():
    """NumpyBackend.adapt() binds a callable to numpy and produces correct results."""
    from civic_digital_twins.dt_model import NumpyBackend
    from civic_digital_twins.dt_model.engine.frontend import graph
    from civic_digital_twins.dt_model.model.index import Index

    p = graph.placeholder("x", default_value=3.0)
    fc = graph.function_call("double", p)
    I_x = Index("x", p)
    I_out = Index("out", fc)
    model = _make_model(I_x, I_out)

    result = Evaluation(model).evaluate(
        functions={"double": NumpyBackend.adapt(lambda x: x * 2)},
        backend=NumpyBackend,
    )
    assert float(result[I_out]) == pytest.approx(float(result[I_x]) * 2)


def test_evaluate_functions_adapt_functor_passed_through_unchanged():
    """A Functor from NumpyBackend.adapt() is accepted and used as-is."""
    from civic_digital_twins.dt_model import NumpyBackend
    from civic_digital_twins.dt_model.engine.frontend import graph
    from civic_digital_twins.dt_model.model.index import Index

    p = graph.placeholder("x", default_value=5.0)
    fc = graph.function_call("negate", p)
    I_x = Index("x", p)
    I_out = Index("out", fc)
    model = _make_model(I_x, I_out)

    functor = NumpyBackend.adapt(lambda x: -x)
    result = Evaluation(model).evaluate(
        functions={"negate": functor},
        backend=NumpyBackend,
    )
    assert float(result[I_out]) == pytest.approx(-float(result[I_x]))


def test_lambda_adapter_deprecated():
    """Constructing LambdaAdapter directly triggers a DeprecationWarning."""
    from civic_digital_twins.dt_model.engine.numpybackend import executor

    with pytest.warns(DeprecationWarning, match="NumpyBackend.adapt"):
        executor.LambdaAdapter(lambda x: x)


def test_unsupported_backend_raises():
    """Passing an unsupported backend raises NotImplementedError."""
    from civic_digital_twins.dt_model.model.index import Index

    model = _make_model(Index("x", 1.0))

    class _FakeBackend:
        pass

    with pytest.raises(NotImplementedError, match="not supported"):
        Evaluation(model).evaluate(backend=_FakeBackend)  # type: ignore[arg-type]


def test_build_plan_unknown_strategy_raises():
    """build_plan raises ValueError for an unrecognised strategy string."""
    model = _make_model(Index("x", 1.0))
    with pytest.raises(ValueError, match="Unknown strategy"):
        Evaluation(model).build_plan(strategy="turbo")


def test_evaluation_rejects_scenario_like_object_after_duck_typing_removal():
    """Evaluation() no longer accepts duck-typed Scenario-like objects; a TypeError is raised."""
    I_x = Index("x", 3.0)
    I_y = Index("y", I_x.node * 2.0)
    model = _make_model(I_x, I_y)

    class _FakeScenario:
        def __init__(self, m):
            self.model = m

    with pytest.raises(TypeError, match="Scenario, Model, or ModelVariant"):
        Evaluation(_FakeScenario(model))  # type: ignore[arg-type]


def test_evaluation_rejects_object_without_model_attribute():
    """Evaluation() raises TypeError when the argument is not a Scenario, Model, or ModelVariant."""
    with pytest.raises(TypeError, match="Scenario, Model, or ModelVariant"):
        Evaluation(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Parameter_axes= + callable parameters (correlated PARAMETER axes)
# ---------------------------------------------------------------------------


def test_parameter_axes_single_named_axis_one_callable():
    """Single named axis + one callable → shape (N,), correct values."""
    cost = Index("cost", None)
    result = Index("result", cost.node * 2.0)
    model = _make_model(cost, result)
    ev = Evaluation(Scenario(model))

    base_arr = np.array([1.0, 2.0, 3.0])
    res = ev.evaluate(
        parameter_axes={"base": base_arr},
        parameters={cost: lambda base: base * 3.0},
    )
    # cost = base*3 → [3,6,9]; result = cost*2 → [6,12,18]
    assert res.expected_value(result).shape == (3,)
    np.testing.assert_allclose(res.expected_value(result), [6.0, 12.0, 18.0])


def test_parameter_axes_two_named_axes_multiple_callables():
    """Two named axes + multiple callables → shape (N, M), correct values."""
    E = 4
    cost_e = [Index(f"cost_{e}", None) for e in range(E)]
    numbers = [1.0, 2.0, 3.0, 4.0]
    number_e = [ConstIndex(f"number_{e}", numbers[e]) for e in range(E)]
    total = Index("total", sum(cost_e[e].node * number_e[e].node for e in range(E)))
    model = _make_model(*cost_e, *number_e, total)
    ev = Evaluation(Scenario(model))

    base_arr = np.linspace(4.0, 10.0, 5)  # N=5
    grad_arr = np.linspace(0.1, 0.5, 4)  # M=4
    res = ev.evaluate(
        parameter_axes={"base": base_arr, "gradient": grad_arr},
        parameters={cost_e[e]: (lambda base, gradient, e=e: base - gradient * e) for e in range(E)},
    )
    assert res.expected_value(total).shape == (5, 4)

    # Brute-force: cost_e[e] = base[:,None] - gradient[None,:] * e → (5,4)
    # total = sum_e cost_e[e] * numbers[e] → (5,4)
    expected = sum((base_arr[:, None] - grad_arr[None, :] * e) * numbers[e] for e in range(E))
    np.testing.assert_allclose(res.expected_value(total), expected)


def test_parameter_axes_model1_model2_equivalence():
    """Model 1 (formula-based cost_e) and Model 2 (callable cost_e) are numerically identical.

    Both compute total = SUM_e cost_e[e] * number_e[e] where
    cost_e[e] = base - gradient * e.  The traditional model expresses cost_e
    as graph formulas; the callable model supplies them via parameter_axes= +
    callables.  Results must be equal in shape, values, and axis layout.
    """
    E = 7
    numbers = [float(e + 1) for e in range(E)]
    base_arr = np.linspace(4.0, 10.0, 6)  # N=6
    grad_arr = np.linspace(0.2, 0.6, 5)  # M=5

    # --- Model 1: traditional formula-based cost_e ---
    base1 = Index("base", None)
    gradient1 = Index("gradient", None)
    cost1_e = [Index(f"cost1_{e}", base1.node - gradient1.node * e) for e in range(E)]
    number1_e = [ConstIndex(f"number1_{e}", numbers[e]) for e in range(E)]
    total1 = Index("total1", sum(cost1_e[e].node * number1_e[e].node for e in range(E)))
    model1 = _make_model(base1, gradient1, *cost1_e, *number1_e, total1)
    res1 = Evaluation(Scenario(model1)).evaluate(
        parameters={base1: base_arr, gradient1: grad_arr},
    )

    # --- Model 2: parameter_axes= + callables ---
    cost2_e = [Index(f"cost2_{e}", None) for e in range(E)]
    number2_e = [ConstIndex(f"number2_{e}", numbers[e]) for e in range(E)]
    total2 = Index("total2", sum(cost2_e[e].node * number2_e[e].node for e in range(E)))
    model2 = _make_model(*cost2_e, *number2_e, total2)
    res2 = Evaluation(Scenario(model2)).evaluate(
        parameter_axes={"base": base_arr, "gradient": grad_arr},
        parameters={cost2_e[e]: (lambda base, gradient, e=e: base - gradient * e) for e in range(E)},
    )

    ev1 = res1.expected_value(total1)
    ev2 = res2.expected_value(total2)
    assert ev1.shape == ev2.shape
    np.testing.assert_allclose(ev1, ev2)


def test_parameter_axes_mix_named_and_anonymous():
    """Named axes + anonymous array parameter → shape (N, M, K)."""
    cost = Index("cost", None)
    scale = Index("scale", None)
    result = Index("result", cost.node * scale.node)
    model = _make_model(cost, scale, result)
    ev = Evaluation(Scenario(model))

    base_arr = np.array([1.0, 2.0, 3.0])  # N=3
    grad_arr = np.array([0.1, 0.2])  # M=2
    scale_arr = np.array([10.0, 20.0, 30.0, 40.0])  # K=4

    res = ev.evaluate(
        parameter_axes={"base": base_arr, "grad": grad_arr},
        parameters={
            cost: lambda base, grad: base - grad,
            scale: scale_arr,
        },
    )
    # Named axes first (positions 0,1), anonymous axis last (position 2).
    assert res.expected_value(result).shape == (3, 2, 4)
    expected_cost = base_arr[:, None] - grad_arr[None, :]  # (3, 2)
    expected = expected_cost[:, :, None] * scale_arr[None, None, :]  # (3, 2, 4)
    np.testing.assert_allclose(res.expected_value(result), expected)


def test_parameter_axes_named_axis_values_accessor():
    """named_axis_values returns the raw 1-D input arrays keyed by name."""
    cost = Index("cost", None)
    model = _make_model(cost)
    ev = Evaluation(Scenario(model))

    base_arr = np.linspace(1.0, 5.0, 5)
    grad_arr = np.linspace(0.1, 0.3, 3)
    res = ev.evaluate(
        parameter_axes={"base": base_arr, "gradient": grad_arr},
        parameters={cost: lambda base, gradient: base - gradient},
    )
    np.testing.assert_array_equal(res.named_axis_values["base"], base_arr)
    np.testing.assert_array_equal(res.named_axis_values["gradient"], grad_arr)


def test_parameter_axes_callable_not_in_parameter_values():
    """Callable-backed indexes are not PARAMETER-axis indexes; parameter_values_for raises."""
    cost = Index("cost", None)
    model = _make_model(cost)
    ev = Evaluation(Scenario(model))

    res = ev.evaluate(
        parameter_axes={"base": np.array([1.0, 2.0])},
        parameters={cost: lambda base: base},
    )
    with pytest.raises(KeyError):
        res.parameter_values_for(cost)
    assert cost not in res.parameter_values


def test_parameter_axes_callable_without_parameter_axes_raises():
    """Callable in parameters= without parameter_axes= raises ValueError."""
    cost = Index("cost", None)
    model = _make_model(cost)
    ev = Evaluation(Scenario(model))

    with pytest.raises(ValueError, match="parameter_axes"):
        ev.evaluate(parameters={cost: lambda base: base})


def test_parameter_axes_callable_unknown_required_param_raises():
    """Callable with a required parameter not in parameter_axes= raises ValueError."""
    cost = Index("cost", None)
    model = _make_model(cost)
    ev = Evaluation(Scenario(model))

    with pytest.raises(ValueError, match="required parameter"):
        ev.evaluate(
            parameter_axes={"base": np.array([1.0, 2.0])},
            parameters={cost: lambda base, unknown_axis: base + unknown_axis},
        )


def test_parameter_axes_closure_e_equals_e_idiom():
    """The e=e default-arg idiom correctly freezes the loop variable in each callable."""
    E = 3
    cost_e = [Index(f"cost_{e}", None) for e in range(E)]
    total = Index("total", sum(cost_e[e].node for e in range(E)))
    model = _make_model(*cost_e, total)
    ev = Evaluation(Scenario(model))

    base_arr = np.array([10.0])
    # cost_e[e] = base * e → [0, 10, 20]; total = 30
    # Without e=e all lambdas would use e=2 → total = 10+20+20 = 50 (wrong)
    res = ev.evaluate(
        parameter_axes={"base": base_arr},
        parameters={cost_e[e]: (lambda base, e=e: base * e) for e in range(E)},
    )
    np.testing.assert_allclose(res.expected_value(total), [30.0])


def test_parameter_axes_callable_with_var_keyword_receives_all_axes():
    """A callable with **kwargs receives all named axis arrays."""
    cost = Index("cost", None)
    model = _make_model(cost)
    ev = Evaluation(Scenario(model))

    base_arr = np.array([1.0, 2.0])
    grad_arr = np.array([0.1, 0.2, 0.3])
    res = ev.evaluate(
        parameter_axes={"base": base_arr, "gradient": grad_arr},
        parameters={cost: lambda **axes: axes["base"] - axes["gradient"]},
    )
    assert res.expected_value(cost).shape == (2, 3)
    expected = base_arr[:, None] - grad_arr[None, :]
    np.testing.assert_allclose(res.expected_value(cost), expected)


def test_parameter_axes_callable_with_var_positional_raises():
    """A callable with *args raises TypeError."""
    cost = Index("cost", None)
    model = _make_model(cost)
    ev = Evaluation(Scenario(model))

    with pytest.raises(TypeError, match=r"\*args"):
        ev.evaluate(
            parameter_axes={"base": np.array([1.0, 2.0])},
            parameters={cost: lambda *args: args[0]},
        )
