"""Tests for the @functions decorator and Model._node_functions contract."""

# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest

from civic_digital_twins.dt_model import NumpyBackend, functions, inputs, outputs
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.model.index import Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation


# ---------------------------------------------------------------------------
# @functions decorator — unit tests
# ---------------------------------------------------------------------------


def test_functions_decorator_sets_declared_field():
    @functions
    class F:
        solve: Any

    functor = NumpyBackend.adapt(lambda x: x)
    f = F(solve=functor)
    assert f.solve is functor


def test_functions_decorator_missing_required_raises():
    @functions
    class F:
        solve: Any

    with pytest.raises(TypeError, match="missing required argument.*solve"):
        F()


def test_functions_decorator_extra_allow_default():
    @functions
    class F:
        solve: Any

    functor = NumpyBackend.adapt(lambda x: x)
    smooth = NumpyBackend.adapt(lambda x: x)
    f = F(solve=functor, smooth=smooth)
    assert f._extra == {"smooth": smooth}


def test_functions_decorator_extra_forbid_raises():
    @functions(extra="forbid")
    class F:
        solve: Any

    functor = NumpyBackend.adapt(lambda x: x)
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        F(solve=functor, extra_fn=NumpyBackend.adapt(lambda x: x))


def test_functions_decorator_items_yields_declared_then_extra():
    @functions
    class F:
        a: Any
        b: Any

    fa = NumpyBackend.adapt(lambda x: x)
    fb = NumpyBackend.adapt(lambda x: -x)
    fc = NumpyBackend.adapt(lambda x: x * 2)
    f = F(a=fa, b=fb, c=fc)
    items = list(f.items())
    assert items[0] == ("a", fa)
    assert items[1] == ("b", fb)
    assert items[2] == ("c", fc)


def test_functions_decorator_repr():
    @functions
    class F:
        solve: Any

    functor = NumpyBackend.adapt(lambda x: x)
    f = F(solve=functor)
    r = repr(f)
    assert r.startswith("F(")
    assert "solve=" in r


def test_functions_decorator_class_level_default():
    """A class-level attribute is treated as the default for that field."""

    @functions
    class F:
        solve: Any = None  # type: ignore[assignment]

    f = F()
    assert f.solve is None


def test_functions_decorator_declared_frozenset():
    @functions
    class F:
        x: Any
        y: Any

    assert F._declared == frozenset({"x", "y"})


def test_functions_is_functions_flag():
    @functions
    class F:
        x: Any

    assert getattr(F, "_is_functions", False) is True


# ---------------------------------------------------------------------------
# Model._node_functions — BFS node-ownership
# ---------------------------------------------------------------------------


def _make_model_with_function_call():
    """Build a simple model with one function_call node."""
    p = graph.placeholder("inp", default_value=2.0)
    fc = graph.function_call("solve", p)
    inp = Index("inp", p)
    out = Index("out", fc)
    return inp, out, p, fc


def test_model_node_functions_empty_without_functions_arg():
    inp, out, _p, _fc = _make_model_with_function_call()

    @inputs
    class Inputs:
        inp: Index

    @outputs
    class Outputs:
        out: Index

    class M(Model):
        def __init__(self, inp: Index) -> None:
            super().__init__(
                "M",
                inputs=Inputs(inp=inp),
                outputs=Outputs(out=out),
            )

    m = M(inp)
    assert m._node_functions == {}


def test_model_node_functions_populated_with_functions_arg():
    inp, out, _p, fc = _make_model_with_function_call()
    functor = NumpyBackend.adapt(lambda x: x * 3)

    @inputs
    class Inputs:
        inp: Index

    @outputs
    class Outputs:
        out: Index

    @functions
    class F:
        solve: Any

    class M(Model):
        def __init__(self, inp: Index, *, fns: F) -> None:
            super().__init__(
                "M",
                inputs=Inputs(inp=inp),
                outputs=Outputs(out=out),
                functions=fns,
            )

    m = M(inp, fns=F(solve=functor))
    assert fc in m._node_functions
    assert m._node_functions[fc] is functor


def test_model_node_functions_input_node_not_claimed():
    """The placeholder node of an input index must not be claimed even if it matches a name."""
    p = graph.placeholder("solve", default_value=1.0)  # same name as the function
    fc = graph.function_call("solve", p)
    inp = Index("inp", p)
    out = Index("out", fc)
    functor = NumpyBackend.adapt(lambda x: x)

    @inputs
    class Inputs:
        inp: Index

    @outputs
    class Outputs:
        out: Index

    @functions
    class F:
        solve: Any

    class M(Model):
        def __init__(self, inp_idx: Index, *, fns: F) -> None:
            super().__init__(
                "M",
                inputs=Inputs(inp=inp_idx),
                outputs=Outputs(out=out),
                functions=fns,
            )

    m = M(inp, fns=F(solve=functor))
    # The function_call node fc should be claimed; the input placeholder node p should not.
    assert fc in m._node_functions
    assert p not in m._node_functions


def test_model_node_functions_submodel_inherits():
    """Parent model aggregates sub-model _node_functions without re-claiming."""
    p_inner = graph.placeholder("inp", default_value=5.0)
    fc_inner = graph.function_call("solve", p_inner)
    inner_inp = Index("inp", p_inner)
    inner_out = Index("out", fc_inner)

    functor = NumpyBackend.adapt(lambda x: x * 7)

    @inputs
    class InnerInputs:
        inp: Index

    @outputs
    class InnerOutputs:
        out: Index

    @functions
    class InnerF:
        solve: Any

    class InnerModel(Model):
        def __init__(self, inp: Index, *, fns: InnerF) -> None:
            super().__init__(
                "Inner",
                inputs=InnerInputs(inp=inp),
                outputs=InnerOutputs(out=inner_out),
                functions=fns,
            )

    class OuterModel(Model):
        def __init__(self, inp: Index) -> None:
            self.inner = InnerModel(inp, fns=InnerF(solve=functor))
            super().__init__(
                "Outer",
                inputs=InnerInputs(inp=inp),
                outputs=InnerOutputs(out=inner_out),
            )

    m = OuterModel(inner_inp)
    # The outer model should carry the inner model's node_functions.
    assert fc_inner in m._node_functions
    assert m._node_functions[fc_inner] is functor


# ---------------------------------------------------------------------------
# Executor two-level dispatch
# ---------------------------------------------------------------------------


def test_executor_node_identity_dispatch_takes_priority():
    """node_functions[node] wins over functions[name] when both are present."""
    p = graph.placeholder("x", default_value=3.0)
    fc = graph.function_call("f", p)

    node_functor = NumpyBackend.adapt(lambda x: x * 10)  # should be used
    name_functor = NumpyBackend.adapt(lambda x: x + 1)   # should be ignored

    state = executor.State(
        {p: np.array(3.0)},
        functions={"f": name_functor},
        node_functions={fc: node_functor},
    )

    result = executor.evaluate_single_node(state, fc)
    assert float(result) == pytest.approx(30.0)


def test_executor_name_dispatch_fallback():
    """functions[name] is used when node_functions does not contain the node."""
    p = graph.placeholder("x", default_value=3.0)
    fc = graph.function_call("f", p)

    name_functor = NumpyBackend.adapt(lambda x: x + 1)

    state = executor.State(
        {p: np.array(3.0)},
        functions={"f": name_functor},
        node_functions={},
    )

    result = executor.evaluate_single_node(state, fc)
    assert float(result) == pytest.approx(4.0)


def test_executor_no_function_raises():
    p = graph.placeholder("x", default_value=1.0)
    fc = graph.function_call("missing", p)

    state = executor.State({p: np.array(1.0)})
    with pytest.raises(executor.FunctionNotFound):
        executor.evaluate_single_node(state, fc)


# ---------------------------------------------------------------------------
# End-to-end: @functions contract through Evaluation
# ---------------------------------------------------------------------------


def test_evaluation_uses_node_functions():
    """Evaluation picks up _node_functions from the model and dispatches correctly."""
    p = graph.placeholder("x", default_value=4.0)
    fc = graph.function_call("double", p)
    inp_idx = Index("x", p)
    out_idx = Index("out", fc)

    @inputs
    class Inputs:
        x: Index

    @outputs
    class Outputs:
        out: Index

    @functions
    class F:
        double: Any

    class M(Model):
        def __init__(self, x: Index, *, fns: F) -> None:
            super().__init__(
                "M",
                inputs=Inputs(x=x),
                outputs=Outputs(out=out_idx),
                functions=fns,
            )

    functor = NumpyBackend.adapt(lambda x: x * 2)
    m = M(inp_idx, fns=F(double=functor))

    result = Evaluation(m).evaluate(backend=NumpyBackend)
    # 4.0 * 2 = 8.0
    assert float(result[out_idx]) == pytest.approx(8.0)


def test_evaluation_two_submodels_same_function_name_different_functors():
    """Two sub-models with the same function name get independent functors via node identity."""
    # Sub-model A: doubles its input.
    p_a = graph.placeholder("a_inp", default_value=3.0)
    fc_a = graph.function_call("transform", p_a)
    a_inp = Index("a_inp", p_a)
    a_out = Index("a_out", fc_a)

    # Sub-model B: triples its input.
    p_b = graph.placeholder("b_inp", default_value=5.0)
    fc_b = graph.function_call("transform", p_b)
    b_inp = Index("b_inp", p_b)
    b_out = Index("b_out", fc_b)

    @inputs
    class SingleInputs:
        inp: Index

    @outputs
    class SingleOutputs:
        out: Index

    @functions
    class SubF:
        transform: Any

    class SubModel(Model):
        def __init__(self, inp: Index, out: Index, *, fns: SubF) -> None:
            super().__init__(
                "Sub",
                inputs=SingleInputs(inp=inp),
                outputs=SingleOutputs(out=out),
                functions=fns,
            )

    @outputs
    class ParentOutputs:
        a_out: Index
        b_out: Index

    class ParentModel(Model):
        def __init__(self) -> None:
            self.sub_a = SubModel(a_inp, a_out, fns=SubF(transform=NumpyBackend.adapt(lambda x: x * 2)))
            self.sub_b = SubModel(b_inp, b_out, fns=SubF(transform=NumpyBackend.adapt(lambda x: x * 3)))
            super().__init__(
                "Parent",
                outputs=ParentOutputs(a_out=a_out, b_out=b_out),
            )

    parent = ParentModel()
    result = Evaluation(parent).evaluate(backend=NumpyBackend)

    # sub_a: 3.0 * 2 = 6.0
    assert float(result[a_out]) == pytest.approx(6.0)
    # sub_b: 5.0 * 3 = 15.0
    assert float(result[b_out]) == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# ModelVariant: per-branch @functions dispatch
# ---------------------------------------------------------------------------


def test_model_variant_node_functions_merged_from_branches():
    """ModelVariant._node_functions is the union of all branch models' maps."""
    from civic_digital_twins.dt_model.model.model_variant import ModelVariant
    from civic_digital_twins.dt_model.model.index import CategoricalIndex

    p_bike = graph.placeholder("inp", default_value=1.0)
    fc_bike = graph.function_call("compute", p_bike)
    bike_inp = Index("inp", p_bike)
    bike_out = Index("out", fc_bike)

    p_train = graph.placeholder("inp", default_value=1.0)
    fc_train = graph.function_call("compute", p_train)
    train_inp = Index("inp", p_train)
    train_out = Index("out", fc_train)

    bike_fn = NumpyBackend.adapt(lambda x: x * 2)
    train_fn = NumpyBackend.adapt(lambda x: x * 10)

    @functions
    class F:
        compute: Any

    @inputs
    class Inp:
        inp: Index

    @outputs
    class Out:
        out: Index

    class BikeModel(Model):
        def __init__(self, inp: Index, *, fns: F) -> None:
            super().__init__("Bike", inputs=Inp(inp=inp), outputs=Out(out=bike_out), functions=fns)

    class TrainModel(Model):
        def __init__(self, inp: Index, *, fns: F) -> None:
            super().__init__("Train", inputs=Inp(inp=inp), outputs=Out(out=train_out), functions=fns)

    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = ModelVariant(
        "Transport",
        {
            "bike": BikeModel(bike_inp, fns=F(compute=bike_fn)),
            "train": TrainModel(train_inp, fns=F(compute=train_fn)),
        },
        selector=mode,
    )

    # Both branch nodes should be present in the merged map.
    assert fc_bike in mv._node_functions
    assert fc_train in mv._node_functions
    assert mv._node_functions[fc_bike] is bike_fn
    assert mv._node_functions[fc_train] is train_fn
