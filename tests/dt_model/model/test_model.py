"""Tests for civic_digital_twins.dt_model.model.Model."""

# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model import expose, functions, inputs, outputs
from civic_digital_twins.dt_model.model.index import Distribution, GenericIndex, Index, TimeseriesIndex
from civic_digital_twins.dt_model.model.model import (
    AbstractIndexNotInInputsError,
    FunctionsTypeMismatchError,
    InputsContractError,
    InputsTypeMismatchError,
    IOProxy,
    Model,
    ModelContractError,
    ModelContractViolation,
    ModelContractWarning,
)

c1: Distribution = stats.norm(loc=2.0, scale=1.0)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# IOProxy — direct construction
# ---------------------------------------------------------------------------


def test_ioproxy_attribute_access():
    """IOProxy returns the correct index for each registered attribute name."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    proxy = IOProxy([("alpha", a), ("beta", b)])
    assert proxy.alpha is a
    assert proxy.beta is b


def test_ioproxy_iteration():
    """Iterating over IOProxy yields indexes in declaration order."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    proxy = IOProxy([("alpha", a), ("beta", b)])
    assert list(proxy) == [a, b]


def test_ioproxy_len():
    """len(proxy) returns the number of declared entries."""
    a = Index("a", 1.0)
    proxy = IOProxy([("alpha", a)])
    assert len(proxy) == 1


def test_ioproxy_contains():
    """Membership test uses identity, not equality."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    proxy = IOProxy([("alpha", a)])
    assert a in proxy
    assert b not in proxy


def test_ioproxy_unknown_attribute_raises():
    """Accessing an undeclared attribute name raises AttributeError."""
    proxy = IOProxy([])
    with pytest.raises(AttributeError, match="No input/output"):
        _ = proxy.nonexistent


def test_ioproxy_is_readonly():
    """Assigning to any attribute on IOProxy raises AttributeError."""
    proxy = IOProxy([])
    with pytest.raises(AttributeError):
        proxy.something = "value"  # type: ignore[misc]


def test_ioproxy_repr():
    """repr(proxy) lists the declared attribute names."""
    a = Index("a", 1.0)
    proxy = IOProxy([("alpha", a)])
    assert "alpha" in repr(proxy)


# ---------------------------------------------------------------------------
# __init_subclass__ guard — bare __init__ requires legacy=True
# ---------------------------------------------------------------------------


def test_bare_init_without_legacy_raises_typeerror():
    """Defining __init__ directly without legacy=True raises TypeError at class creation."""
    with pytest.raises(TypeError, match="defines __init__ directly"):

        class _Bad(Model):
            def __init__(self) -> None:
                super().__init__("bad")


# ---------------------------------------------------------------------------
# abstract_indexes — index-kind classification
# ---------------------------------------------------------------------------


def test_abstract_indexes_includes_timeseries_placeholder():
    """A value-less TimeseriesIndex is abstract; a concrete one is not."""

    @inputs
    class _In:
        demand: TimeseriesIndex
        history: TimeseriesIndex

    placeholder = TimeseriesIndex("demand")
    concrete = TimeseriesIndex("history", np.array([1.0, 2.0]))
    model = Model("M", inputs=_In(demand=placeholder, history=concrete))

    abstract = model.abstract_indexes()
    assert any(idx is placeholder for idx in abstract)
    assert not any(idx is concrete for idx in abstract)


# ---------------------------------------------------------------------------
# AbstractIndexNotInInputsError — convention check (hard error)
# ---------------------------------------------------------------------------


def test_abstract_index_not_in_inputs_error_raised_for_unresolved_output():
    """Raise when an Output holds a genuine unresolved abstract Index absent from Inputs."""

    class _Bad(Model, legacy=True):
        @outputs
        class Outputs:
            placeholder: Index

        def __init__(self) -> None:
            placeholder = Index("dangling", None)
            super().__init__("Bad", outputs=_Bad.Outputs(placeholder=placeholder))

    with pytest.raises(AbstractIndexNotInInputsError, match="'dangling'"):
        _Bad()


def test_abstract_index_not_in_inputs_error_names_every_missing_index():
    """Raise once, naming every undeclared abstract Output index."""

    class _Bad(Model, legacy=True):
        @outputs
        class Outputs:
            first: Index
            second: Index

        def __init__(self) -> None:
            first = Index("first_dangling", None)
            second = Index("second_dangling", None)
            super().__init__("Bad", outputs=_Bad.Outputs(first=first, second=second))

    with pytest.raises(AbstractIndexNotInInputsError) as excinfo:
        _Bad()
    message = str(excinfo.value)
    assert "'first_dangling'" in message
    assert "'second_dangling'" in message


def test_abstract_index_not_in_inputs_no_error_when_declared():
    """No error when the abstract Output index is also declared in Inputs."""

    class _Good(Model, legacy=True):
        @inputs
        class Inputs:
            placeholder: Index

        @outputs
        class Outputs:
            placeholder: Index

        def __init__(self) -> None:
            placeholder = Index("resolved", None)
            super().__init__(
                "Good",
                inputs=_Good.Inputs(placeholder=placeholder),
                outputs=_Good.Outputs(placeholder=placeholder),
            )

    _Good()  # must not raise


# ---------------------------------------------------------------------------
# InputsContractError — convention check (hard error)
# ---------------------------------------------------------------------------


def test_inputs_contract_raises_for_undeclared_index():
    """Raise when a scalar Index parameter is not in Inputs."""

    class _Bad(Model, legacy=True):
        @outputs
        class Outputs:
            result: Index

        @expose
        class Expose:
            received: Index  # tracked so the model is valid, but not in Inputs

        def __init__(self, received: Index) -> None:
            result = Index("result", received + 1.0)
            # received is in Expose (so it's covered), but NOT in any Inputs dataclass
            super().__init__(
                "Bad",
                outputs=_Bad.Outputs(result=result),
                expose=_Bad.Expose(received=received),
            )

    received = Index("x", 1.0)
    with pytest.raises(InputsContractError, match="'received'"):
        _Bad(received)


def test_inputs_contract_raises_for_undeclared_timeseries():
    """Raise when a TimeseriesIndex parameter is not in Inputs."""

    class _Bad(Model, legacy=True):
        @outputs
        class Outputs:
            out: Index

        @expose
        class Expose:
            ts: TimeseriesIndex  # tracked so the model is valid, but not in Inputs

        def __init__(self, ts: TimeseriesIndex) -> None:
            out = Index("out", ts.sum())
            super().__init__("Bad", outputs=_Bad.Outputs(out=out), expose=_Bad.Expose(ts=ts))

    ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
    with pytest.raises(InputsContractError, match="'ts'"):
        _Bad(ts)


def test_inputs_contract_raises_for_undeclared_list():
    """Raise naming every item in a list[Index] parameter not in Inputs."""

    class _Bad(Model, legacy=True):
        @outputs
        class Outputs:
            total: Index

        @expose
        class Expose:
            costs: list  # tracked so the model is valid, but not in Inputs

        def __init__(self, costs: list[Index]) -> None:
            total = Index("total", costs[0] + costs[1])
            super().__init__("Bad", outputs=_Bad.Outputs(total=total), expose=_Bad.Expose(costs=costs))

    costs = [Index("c0", 1.0), Index("c1", 2.0)]
    with pytest.raises(InputsContractError) as excinfo:
        _Bad(costs)
    message = str(excinfo.value)
    assert "costs[0]" in message
    assert "costs[1]" in message


def test_inputs_contract_no_error_when_declared():
    """No error when all Index params are stored in Inputs."""

    class _Good(Model, legacy=True):
        @inputs
        class Inputs:
            received: Index

        @outputs
        class Outputs:
            result: Index

        def __init__(self, received: Index) -> None:
            inputs = _Good.Inputs(received=received)
            result = Index("result", received + 1.0)
            super().__init__(
                "Good",
                inputs=inputs,
                outputs=_Good.Outputs(result=result),
            )

    received = Index("x", 1.0)
    _Good(received)  # must not raise


def test_inputs_contract_no_error_for_non_index_params():
    """No error for str, float, or ndarray constructor parameters."""

    class _Good(Model, legacy=True):
        @outputs
        class Outputs:
            result: Index

        def __init__(self, label: str, scale: float, data: np.ndarray) -> None:
            result = Index("result", scale)
            super().__init__("Good", outputs=_Good.Outputs(result=result))

    _Good("hello", 3.14, np.array([1.0]))  # must not raise


def test_inputs_contract_no_error_for_base_model():
    """No error when constructing Model directly (only fires for subclasses)."""
    # Model itself has no __init__ parameter convention to check
    Model("base", outputs=None)  # must not raise


# ---------------------------------------------------------------------------
# InputsTypeMismatchError — wrong model's Inputs passed (hard error)
# ---------------------------------------------------------------------------


def test_inputs_type_mismatch_raises_for_wrong_model_inputs():
    """Raise when constructed with another model's Inputs instance."""

    class _Alpha(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        def __init__(self, inputs: "_Alpha.Inputs") -> None:
            super().__init__("Alpha", inputs=inputs)

    class _Beta(Model, legacy=True):
        @inputs
        class Inputs:
            y: Index

        def __init__(self, inputs: "_Beta.Inputs") -> None:
            super().__init__("Beta", inputs=inputs)

    wrong = _Beta.Inputs(y=Index("y", 1.0))
    with pytest.raises(InputsTypeMismatchError, match="Alpha"):
        _Alpha(wrong)  # type: ignore[arg-type]


def test_inputs_type_mismatch_raises_even_when_shapes_coincide():
    """Raise even when the wrong Inputs class happens to share the same field shape.

    Without an explicit class check, two structurally-identical but unrelated
    Inputs classes could be swapped with no error at all — the wrong data
    would silently flow into the model.
    """

    class _Alpha(Model, legacy=True):
        @inputs
        class Inputs:
            v: Index

        def __init__(self, inputs: "_Alpha.Inputs") -> None:
            super().__init__("Alpha", inputs=inputs)

    class _Gamma(Model, legacy=True):
        @inputs
        class Inputs:
            v: Index  # same field name and type as _Alpha.Inputs

        def __init__(self, inputs: "_Gamma.Inputs") -> None:
            super().__init__("Gamma", inputs=inputs)

    same_shape_wrong = _Gamma.Inputs(v=Index("v", 1.0))
    with pytest.raises(InputsTypeMismatchError):
        _Alpha(same_shape_wrong)  # type: ignore[arg-type]


def test_inputs_type_mismatch_no_error_when_correct():
    """No error when constructed with the model's own Inputs instance."""

    class _Alpha(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        def __init__(self, inputs: "_Alpha.Inputs") -> None:
            super().__init__("Alpha", inputs=inputs)

    _Alpha(_Alpha.Inputs(x=Index("x", 1.0)))  # must not raise


def test_inputs_type_mismatch_no_error_when_inputs_not_declared():
    """No error when the subclass declares no Inputs class at all (nothing to check against)."""

    @dataclasses.dataclass
    class _SomeInputs:
        x: Index

    class _NoInputs(Model, legacy=True):
        def __init__(self) -> None:
            super().__init__("NoInputs", inputs=_SomeInputs(x=Index("x", 1.0)))

    _NoInputs()  # must not raise


def test_inputs_type_mismatch_error_is_subclass_of_model_contract_error_not_warning():
    """InputsTypeMismatchError is a ModelContractError, not a ModelContractWarning."""
    assert issubclass(InputsTypeMismatchError, ModelContractError)
    assert not issubclass(InputsTypeMismatchError, ModelContractWarning)


def test_inputs_type_mismatch_error_shares_violation_base():
    """InputsTypeMismatchError is a ModelContractViolation, enabling a unified except clause."""
    assert issubclass(InputsTypeMismatchError, ModelContractViolation)


# ---------------------------------------------------------------------------
# FunctionsTypeMismatchError — wrong model's Functions passed (hard error)
# ---------------------------------------------------------------------------


def test_functions_type_mismatch_raises_for_wrong_model_functions():
    """Raise when constructed with another model's Functions instance."""

    class _AlphaF(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @functions
        class Functions:
            pass

        def __init__(self, inputs: "_AlphaF.Inputs", fns: "_AlphaF.Functions") -> None:
            super().__init__("AlphaF", inputs=inputs, functions=fns)

    class _BetaF(Model, legacy=True):
        @inputs
        class Inputs:
            y: Index

        @functions
        class Functions:
            pass

        def __init__(self, inputs: "_BetaF.Inputs", fns: "_BetaF.Functions") -> None:
            super().__init__("BetaF", inputs=inputs, functions=fns)

    wrong = _BetaF.Functions()
    with pytest.raises(FunctionsTypeMismatchError, match="AlphaF"):
        _AlphaF(_AlphaF.Inputs(x=Index("x", 1.0)), wrong)  # type: ignore[arg-type]


def test_functions_type_mismatch_raises_for_non_functions_object():
    """Raise — rather than silently drop — when passed an object that is not the declared Functions."""

    class _AlphaF(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @functions
        class Functions:
            pass

        def __init__(self, inputs: "_AlphaF.Inputs", fns: object) -> None:
            super().__init__("AlphaF", inputs=inputs, functions=fns)

    with pytest.raises(FunctionsTypeMismatchError):
        _AlphaF(_AlphaF.Inputs(x=Index("x", 1.0)), object())


def test_functions_type_mismatch_no_error_when_correct():
    """No error when constructed with the model's own Functions instance."""

    class _AlphaF(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @functions
        class Functions:
            pass

        def __init__(self, inputs: "_AlphaF.Inputs", fns: "_AlphaF.Functions") -> None:
            super().__init__("AlphaF", inputs=inputs, functions=fns)

    _AlphaF(_AlphaF.Inputs(x=Index("x", 1.0)), _AlphaF.Functions())  # must not raise


def test_functions_type_mismatch_no_error_when_functions_not_declared():
    """No error when the subclass declares no Functions class at all (nothing to check against)."""

    class _NoFns(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        def __init__(self, inputs: "_NoFns.Inputs") -> None:
            # A functions value is passed even though this class declares no
            # ``Functions``; with nothing to check against, it is left alone.
            super().__init__("NoFns", inputs=inputs, functions=object())

    _NoFns(_NoFns.Inputs(x=Index("x", 1.0)))  # must not raise


def test_functions_type_mismatch_error_shares_violation_base():
    """FunctionsTypeMismatchError is a ModelContractError/Violation, not a warning."""
    assert issubclass(FunctionsTypeMismatchError, ModelContractError)
    assert issubclass(FunctionsTypeMismatchError, ModelContractViolation)
    assert not issubclass(FunctionsTypeMismatchError, ModelContractWarning)


def test_inputs_contract_error_is_subclass_of_model_contract_error_not_warning():
    """InputsContractError is a ModelContractError, not a ModelContractWarning."""
    assert issubclass(InputsContractError, ModelContractError)
    assert not issubclass(InputsContractError, ModelContractWarning)


def test_inputs_contract_error_and_abstract_index_error_share_violation_base():
    """Both are ModelContractViolation, enabling a single unified except clause."""
    assert issubclass(InputsContractError, ModelContractViolation)
    assert issubclass(AbstractIndexNotInInputsError, ModelContractViolation)


def test_abstract_index_not_in_inputs_error_is_subclass_of_model_contract_error_not_warning():
    """AbstractIndexNotInInputsError is a ModelContractError, not a ModelContractWarning."""
    assert issubclass(AbstractIndexNotInInputsError, ModelContractError)
    assert not issubclass(AbstractIndexNotInInputsError, ModelContractWarning)


def test_model_contract_error_base_catches_inputs_contract_error():
    """Catching ModelContractError also catches InputsContractError."""

    class _Bad(Model, legacy=True):
        @outputs
        class Outputs:
            result: Index

        @expose
        class Expose:
            received: Index

        def __init__(self, received: Index) -> None:
            result = Index("result", received + 1.0)
            super().__init__("Bad", outputs=_Bad.Outputs(result=result), expose=_Bad.Expose(received=received))

    received = Index("x", 1.0)
    with pytest.raises(ModelContractError):
        _Bad(received)


def test_model_contract_violation_catches_inputs_contract_error():
    """Catching ModelContractViolation also catches the hard-error InputsContractError."""

    class _Bad(Model, legacy=True):
        @outputs
        class Outputs:
            result: Index

        @expose
        class Expose:
            received: Index

        def __init__(self, received: Index) -> None:
            result = Index("result", received + 1.0)
            super().__init__("Bad", outputs=_Bad.Outputs(result=result), expose=_Bad.Expose(received=received))

    received = Index("x", 1.0)
    with pytest.raises(ModelContractViolation):
        _Bad(received)


def test_model_contract_violation_catches_abstract_index_not_in_inputs_error():
    """Catching ModelContractViolation also catches the hard-error AbstractIndexNotInInputsError."""

    class _Bad(Model, legacy=True):
        @outputs
        class Outputs:
            placeholder: Index

        def __init__(self) -> None:
            placeholder = Index("dangling", None)
            super().__init__("Bad", outputs=_Bad.Outputs(placeholder=placeholder))

    with pytest.raises(ModelContractViolation):
        _Bad()


# ---------------------------------------------------------------------------
# _check_inputs_contract — dict-valued parameter path
# ---------------------------------------------------------------------------


def test_inputs_contract_raises_for_undeclared_dict():
    """InputsContractError names each missing key when the parameter is a dict."""

    class _Bad(Model, legacy=True):
        @dataclasses.dataclass
        class Outputs:
            result: Index

        def __init__(self, mapping: dict) -> None:
            result = Index("result", 1.0)
            super().__init__("Bad", outputs=_Bad.Outputs(result=result))

    x = Index("x", 1.0)
    y = Index("y", 2.0)
    with pytest.raises(InputsContractError) as excinfo:
        _Bad(mapping={"a": x, "b": y})

    message = str(excinfo.value)
    assert "mapping['a']" in message
    assert "mapping['b']" in message


def test_inputs_contract_no_error_for_declared_dict():
    """No InputsContractError when all dict-valued GenericIndex entries are in Inputs."""

    class _Good(Model, legacy=True):
        @dataclasses.dataclass
        class Inputs:
            mapping: dict

        @dataclasses.dataclass
        class Outputs:
            result: Index

        def __init__(self, mapping: dict) -> None:
            inputs = _Good.Inputs(mapping=mapping)
            result = Index("result", 1.0)
            super().__init__("Good", inputs=inputs, outputs=_Good.Outputs(result=result))

    x = Index("x", 1.0)
    _Good(mapping={"a": x})  # must not raise


# ---------------------------------------------------------------------------
# _check_inputs_contract — inspect.signature exception path
# ---------------------------------------------------------------------------


def test_inputs_contract_no_crash_when_signature_unavailable():
    """_check_inputs_contract silently returns when inspect.signature raises."""
    import unittest.mock

    class _Model(Model, legacy=True):
        @dataclasses.dataclass
        class Outputs:
            result: Index

        @dataclasses.dataclass
        class Expose:
            x: Index

        def __init__(self, x: Index) -> None:
            result = Index("result", x + 1.0)
            super().__init__("M", outputs=_Model.Outputs(result=result), expose=_Model.Expose(x=x))

    x = Index("x", 1.0)
    # Patch inspect.signature to raise TypeError, simulating a built-in or
    # C-extension __init__ whose signature cannot be introspected.
    with unittest.mock.patch("inspect.signature", side_effect=TypeError("no sig")):
        # Must not raise — the contract check should be silently skipped.
        _Model(x)


def test_inputs_contract_skips_params_absent_from_locals():
    """_check_inputs_contract silently skips parameters deleted before super().__init__().

    This covers the ``value is inspect.Parameter.empty`` branch — hit when a
    parameter declared in the signature is removed from local scope (via ``del``)
    before ``super().__init__()`` is called.
    """

    class _ModelWithDeletedParam(Model, legacy=True):
        @dataclasses.dataclass
        class Inputs:
            x: Index

        @dataclasses.dataclass
        class Outputs:
            result: Index

        def __init__(self, x: Index, y: Index) -> None:  # type: ignore[override]
            result = Index("result", x + 1.0)
            del y  # y is now absent from f_locals when super().__init__ inspects
            super().__init__(
                "M",
                inputs=_ModelWithDeletedParam.Inputs(x=x),
                outputs=_ModelWithDeletedParam.Outputs(result=result),
            )

    x = Index("x", 1.0)
    y = Index("y", 2.0)
    # Must not raise even though 'y' is absent from f_locals:
    # x is declared in Inputs; y was deleted before inspection.
    _ModelWithDeletedParam(x, y)


# ---------------------------------------------------------------------------
# Dropped concrete sub-model index detection (issue #195)
# ---------------------------------------------------------------------------


def test_dropped_concrete_submodel_index_raises_at_construction():
    """Model.__init__ raises ValueError when a sub-model's concrete Index is not in parent.indexes.

    The canonical failure mode: a parent model builds a sub-model with
    concrete-valued Index parameters (e.g. Index('k', 0.5)), stores the
    sub-model as self.sub, but does not include those concrete indexes in its
    own Inputs/Outputs/Expose.  Scenario.base_substitutions() would silently
    skip them, causing PlaceholderValueNotProvided at evaluation time.  The
    check here surfaces the problem at model construction.
    """
    from civic_digital_twins.dt_model.model.contracts import define, inputs, outputs

    @define("Inner")
    class InnerModel(Model, legacy=True):
        @inputs
        class Inputs:
            k: Index

        @outputs
        class Outputs:
            result: Index

        def compute(self, inp: Inputs) -> Outputs:
            return InnerModel.Outputs(result=Index("result", inp.k.node * 2.0))

    class OuterModel(Model, legacy=True):
        @inputs
        class Inputs:
            pass  # intentionally empty — k is NOT forwarded to the parent

        @outputs
        class Outputs:
            result: Index

        def __init__(self) -> None:
            k = Index("k", 0.5)  # concrete value, will be in inner.indexes
            self.inner = InnerModel(inputs=InnerModel.Inputs(k=k))
            super().__init__(
                "Outer",
                inputs=OuterModel.Inputs(),
                outputs=OuterModel.Outputs(result=self.inner.outputs.result),
            )

    with pytest.raises(ValueError, match="appear in the model's formulas but are not declared"):
        OuterModel()


def test_dropped_concrete_submodel_index_error_names_culprits():
    """Error message includes the names of the dropped concrete indexes."""
    from civic_digital_twins.dt_model.model.contracts import define, inputs, outputs

    @define("Inner")
    class _Inner(Model, legacy=True):
        @inputs
        class Inputs:
            alpha: Index
            beta: Index

        @outputs
        class Outputs:
            result: Index

        def compute(self, inp: Inputs) -> Outputs:
            return _Inner.Outputs(result=Index("result", inp.alpha.node + inp.beta.node))

    class _Outer(Model, legacy=True):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            result: Index

        def __init__(self) -> None:
            alpha = Index("alpha_param", 1.0)
            beta = Index("beta_param", 2.0)
            self.inner = _Inner(inputs=_Inner.Inputs(alpha=alpha, beta=beta))
            super().__init__(
                "Outer2",
                inputs=_Outer.Inputs(),
                outputs=_Outer.Outputs(result=self.inner.outputs.result),
            )

    with pytest.raises(ValueError, match="alpha_param"):
        _Outer()


def test_concrete_submodel_index_in_parent_expose_does_not_raise():
    """No error when all concrete sub-model indexes are included in parent Expose."""
    from civic_digital_twins.dt_model.model.contracts import define, expose, inputs, outputs

    @define("Inner")
    class _InnerOK(Model, legacy=True):
        @inputs
        class Inputs:
            k: Index

        @outputs
        class Outputs:
            result: Index

        def compute(self, inp: Inputs) -> Outputs:
            return _InnerOK.Outputs(result=Index("result", inp.k.node * 3.0))

    @define("Outer")
    class _OuterOK(Model, legacy=True):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            result: Index

        @expose
        class Expose:
            domain_indexes: list[GenericIndex]

        def compute(self, inp: Inputs) -> tuple[Outputs, Expose]:
            k = Index("k", 0.5)
            inner = _InnerOK(inputs=_InnerOK.Inputs(k=k))
            self.inner = inner
            return (
                _OuterOK.Outputs(result=inner.outputs.result),
                _OuterOK.Expose(domain_indexes=list(inner.indexes)),
            )

    # Should construct without error — k is tracked via domain_indexes in Expose.
    m = _OuterOK()
    assert any(getattr(i, "name", None) == "k" for i in m.expose.domain_indexes)


def test_const_index_in_submodel_does_not_raise():
    """ConstIndex in a sub-model is exempt: its value is baked into the graph."""
    from civic_digital_twins.dt_model import ConstIndex
    from civic_digital_twins.dt_model.model.contracts import define, inputs, outputs

    @define("Inner")
    class _InnerConst(Model, legacy=True):
        @inputs
        class Inputs:
            k: ConstIndex

        @outputs
        class Outputs:
            result: Index

        def compute(self, inp: Inputs) -> Outputs:
            return _InnerConst.Outputs(result=Index("result", inp.k.node * 1.0))

    class _OuterConst(Model, legacy=True):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            result: Index

        def __init__(self) -> None:
            k = ConstIndex("k", 0.5)  # value baked into graph — exempt from check
            self.inner = _InnerConst(inputs=_InnerConst.Inputs(k=k))
            super().__init__(
                "OuterConst",
                inputs=_OuterConst.Inputs(),
                outputs=_OuterConst.Outputs(result=self.inner.outputs.result),
            )

    # Should construct without error — ConstIndex values are baked into the graph.
    m = _OuterConst()
    assert m is not None


def test_inline_concrete_index_not_in_outputs_raises():
    """Single model: Index(name, scalar) used in a formula but not in Outputs/Expose raises.

    This is the single-model analogue of the sub-model dropped-index bug:
    Index('a', 0.2) creates a graph.placeholder whose value must be injected
    by Scenario.base_substitutions().  If the index is absent from model.indexes,
    the value is silently lost and evaluation raises PlaceholderValueNotProvided.
    The check in Model.__init__ must catch this at construction time.
    """
    from civic_digital_twins.dt_model.model.contracts import define, inputs, outputs

    @define("Single")
    class _Single(Model, legacy=True):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            b: Index

        def compute(self, inp: Inputs) -> Outputs:
            a = Index("a_factor", 0.2)  # concrete — NOT returned in Outputs or Expose
            b = Index("b", a.node * 2.0)
            return _Single.Outputs(b=b)

    with pytest.raises(ValueError, match="a_factor"):
        _Single()


def test_inline_abstract_index_not_in_outputs_raises():
    """Single model: Index(name, None) in a formula but absent from Inputs/Outputs raises."""
    from civic_digital_twins.dt_model.model.contracts import define, inputs, outputs

    @define("SingleAbstract")
    class _SingleAbstract(Model, legacy=True):
        @inputs
        class Inputs:
            pass  # 'x' intentionally omitted from Inputs

        @outputs
        class Outputs:
            result: Index

        def compute(self, inp: Inputs) -> Outputs:
            x = Index("x_abstract", None)  # abstract placeholder, NOT in Inputs
            result = Index("result", x.node + 1.0)
            return _SingleAbstract.Outputs(result=result)

    with pytest.raises(ValueError, match="x_abstract"):
        _SingleAbstract()


def test_orphan_check_visited_guard_diamond_dependency():
    """BFS visited-guard (line inside the while loop) is exercised by a diamond dependency.

    Two declared outputs both depend on the same intermediate formula node.
    The BFS adds that node to to_visit twice (once from each output); the
    second pop must hit the ``if node in visited: continue`` guard.
    The shared dependency itself has an orphaned concrete-valued input,
    so the check fires and names it correctly.
    """
    from civic_digital_twins.dt_model.model.contracts import define, inputs, outputs

    @define("Diamond")
    class _Diamond(Model, legacy=True):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            out_a: Index
            out_b: Index

        def compute(self, inp: Inputs) -> Outputs:
            # shared concrete-valued index (orphaned — not in Outputs)
            k = Index("k_shared", 0.5)
            # two outputs that both depend on k
            out_a = Index("out_a", k.node * 1.0)
            out_b = Index("out_b", k.node * 2.0)
            return _Diamond.Outputs(out_a=out_a, out_b=out_b)

    with pytest.raises(ValueError, match="k_shared"):
        _Diamond()


def test_orphan_check_visited_guard_formula_diamond():
    """BFS ``if node in visited: continue`` guard (L916) is hit by a formula diamond.

    When an uncovered formula node M is a transitive dependency of two separate
    formula nodes A and B, and A is itself a dependency of B, M is appended to
    ``to_visit`` twice before it is processed.  On the second pop the visited
    guard fires.

    Graph:  out.node = A + B
                A = shared + 0  (shares node with the first dep of out)
                B = A * 1       (also depends on A)
    → ``_iter_node_deps(out)`` returns [A, B]; processing B appends A again.
    """
    from civic_digital_twins.dt_model.model.contracts import define, inputs, outputs

    @define("FormulaTriangle")
    class _Tri(Model, legacy=True):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            out: Index

        def compute(self, inp: Inputs) -> Outputs:
            k = Index("k_tri", 0.5)  # orphaned concrete index
            shared = k.node + 0.0  # uncovered formula node A
            dep_b = shared * 1.0  # uncovered formula node B, depends on A
            out = Index("out", shared + dep_b)  # out depends on both A and B
            return _Tri.Outputs(out=out)

    with pytest.raises(ValueError, match="k_tri"):
        _Tri()


def test_orphan_check_no_false_positive_on_formula_backed_input():
    """Orphan detection must not flag nodes inside a formula-backed input as orphans.

    When sub-model A's output (a formula index, not a plain placeholder) is
    wired as an input to model B, model B's ``input_formula_nodes`` boundary
    stops the BFS traversal at A's output node.  Nodes inside A's formula are
    not B's concern and must not be reported as orphans.

    Regression test for the id()-keyed visited/covered sets introduced in
    ``_find_orphaned_placeholder_nodes`` (Minor 4 fix).
    """
    from civic_digital_twins.dt_model.model.contracts import define, inputs, outputs

    @define("Producer")
    class _Producer(Model, legacy=True):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            y: Index

        def compute(self, inp: Inputs) -> Outputs:
            return _Producer.Outputs(y=Index("y", inp.x.node * 2.0))

    @define("Consumer")
    class _Consumer(Model, legacy=True):
        @inputs
        class Inputs:
            y: Index  # will receive a formula-backed index from _Producer

        @outputs
        class Outputs:
            z: Index

        def compute(self, inp: Inputs) -> Outputs:
            return _Consumer.Outputs(z=Index("z", inp.y.node + 1.0))

    producer = _Producer(inputs=_Producer.Inputs(x=Index("x", 5.0)))
    # Wire Producer output as Consumer input — must not raise.
    consumer = _Consumer(inputs=_Consumer.Inputs(y=producer.outputs.y))
    assert consumer.outputs.z is not None
