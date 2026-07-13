"""Tests for civic_digital_twins.dt_model.model.Model."""

# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model.model.index import Distribution, DistributionIndex, Index, TimeseriesIndex
from civic_digital_twins.dt_model.model.model import InputsContractWarning, IOProxy, Model, ModelContractWarning

c1: Distribution = stats.norm(loc=2.0, scale=1.0)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Backward compatibility — existing Model behaviour unchanged
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_stores_name_and_indexes():
    """Model stores its name and index list."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", [a, b])
    assert m.name == "test"
    assert m.indexes == [a, b]


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_abstract_indexes_empty_when_all_concrete():
    """abstract_indexes() is empty when all indexes have concrete values."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", [a, b])
    assert m.abstract_indexes() == []
    assert m.is_instantiated()


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_abstract_indexes_includes_none_value():
    """abstract_indexes() includes placeholder (value=None) indexes."""
    a = Index("a", 1.0)
    p = Index("p", None)
    m = Model("test", [a, p])
    result = m.abstract_indexes()
    assert len(result) == 1 and result[0] is p
    assert not m.is_instantiated()


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_abstract_indexes_includes_distribution_value():
    """abstract_indexes() includes DistributionIndex instances."""
    a = Index("a", 1.0)
    d = DistributionIndex("d", stats.norm, {"loc": 2.0, "scale": 1.0})
    m = Model("test", [a, d])
    result = m.abstract_indexes()
    assert len(result) == 1 and result[0] is d
    assert not m.is_instantiated()


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_abstract_indexes_includes_timeseries_placeholder():
    """abstract_indexes() includes TimeseriesIndex placeholders."""
    ts = TimeseriesIndex("ts")
    tf = TimeseriesIndex("tf", np.array([1.0, 2.0]))
    m = Model("test", [ts, tf])
    result = m.abstract_indexes()
    assert len(result) == 1 and result[0] is ts
    assert not m.is_instantiated()


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_formula_index_is_concrete():
    """A formula-based Index (node wrapping) is not abstract."""
    from civic_digital_twins.dt_model.engine.frontend import graph

    n = graph.constant(3.0)
    idx = Index("formula", n)
    m = Model("test", [idx])
    assert m.abstract_indexes() == []
    assert m.is_instantiated()


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_no_inputs_outputs_backward_compat():
    """Model without inputs/outputs arguments works exactly as before."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", [a, b])
    assert len(m.inputs) == 0
    assert len(m.outputs) == 0


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
# Model inputs / outputs — access via self.* attribute names
# ---------------------------------------------------------------------------


class _TwoIndexModel(Model, legacy=True):
    """Minimal model with one input and one output.

    Deliberately uses the legacy flat ``indexes=`` constructor API (it is the
    subject under test for that API) rather than ``@define`` + ``compute()``.
    """

    def __init__(self):
        self.param = Index("Parameter A", None)  # level 1 input
        self.result = Index("Result B", 42.0)  # level 1 output
        self.internal = Index("Internal C", 1.0)  # level 2 — inspectable
        _hidden = Index("Hidden D", 0.0)  # level 3 — anonymous
        super().__init__(
            "Two Index",
            indexes=[self.param, self.result, self.internal, _hidden],
            inputs=[self.param],
            outputs=[self.result],
        )


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_outputs_accessible_by_attr_name():
    """Outputs accessible by the Python attribute name, not index.name."""
    m = _TwoIndexModel()
    assert m.outputs.result is m.result


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_inputs_accessible_by_attr_name():
    """Inputs accessible by the Python attribute name, not index.name."""
    m = _TwoIndexModel()
    assert m.inputs.param is m.param


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_index_name_irrelevant_to_proxy():
    """index.name is a display label — it plays no role in proxy access."""
    m = _TwoIndexModel()
    # m.result has index.name == "Result B", but proxy key is "result"
    assert m.outputs.result.name == "Result B"
    with pytest.raises(AttributeError):
        _ = m.outputs.Result_B  # type: ignore[attr-defined]


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_level2_index_directly_accessible():
    """Level-2 index (self.* but not in outputs) is directly accessible."""
    m = _TwoIndexModel()
    assert m.internal.value == 1.0


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_level2_index_not_in_outputs():
    """Level-2 index is not reachable via outputs."""
    m = _TwoIndexModel()
    assert m.internal not in m.outputs


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_level3_index_in_indexes():
    """Level-3 (anonymous) index is in indexes but not reachable by name."""
    m = _TwoIndexModel()
    # The hidden index is in indexes (engine needs it) but has no attr on m.
    hidden_values = [idx.value for idx in m.indexes if isinstance(idx, Index) and idx.value == 0.0]
    assert hidden_values == [0.0]


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_inputs_and_outputs_are_subsets_of_indexes():
    """Every input and output is also in indexes."""
    m = _TwoIndexModel()
    index_ids = {id(idx) for idx in m.indexes}
    for idx in m.inputs:
        assert id(idx) in index_ids
    for idx in m.outputs:
        assert id(idx) in index_ids


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_same_index_in_inputs_and_outputs():
    """The same index may appear in both inputs and outputs (pass-through)."""

    class _PassThrough(Model, legacy=True):
        def __init__(self):
            self.x = Index("x", None)
            super().__init__("pt", indexes=[self.x], inputs=[self.x], outputs=[self.x])

    m = _PassThrough()
    assert m.inputs.x is m.x
    assert m.outputs.x is m.x


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_outputs_not_in_indexes_raises():
    """Outputs entry not in indexes raises ValueError."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    with pytest.raises(ValueError, match="not in the model's indexes"):
        Model("test", [a], outputs=[b])


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_inputs_not_in_indexes_raises():
    """Inputs entry not in indexes raises ValueError."""
    a = Index("a", 1.0)
    b = Index("b", None)
    with pytest.raises(ValueError, match="not in the model's indexes"):
        Model("test", [a], inputs=[b])


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_outputs_not_assigned_to_self_raises():
    """Outputs entry not assigned to self.* raises ValueError."""

    class _Bad(Model, legacy=True):
        def __init__(self):
            local = Index("local", 1.0)
            super().__init__("bad", indexes=[local], outputs=[local])

    with pytest.raises(ValueError, match="not assigned to any attribute"):
        _Bad()


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_inputs_not_assigned_to_self_raises():
    """Inputs entry not assigned to self.* raises ValueError."""

    class _Bad(Model, legacy=True):
        def __init__(self):
            local = Index("local", None)
            super().__init__("bad", indexes=[local], inputs=[local])

    with pytest.raises(ValueError, match="not assigned to any attribute"):
        _Bad()


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_outputs_collision_raises():
    """Declaring the same self.* index twice in outputs raises ValueError."""

    class _Bad(Model, legacy=True):
        def __init__(self):
            self.x = Index("x", 1.0)
            # Declaring self.x twice — both entries resolve to attr name "x".
            super().__init__("bad", indexes=[self.x], outputs=[self.x, self.x])

    with pytest.raises(ValueError, match="collision"):
        _Bad()


# ---------------------------------------------------------------------------
# Subclass usage — typical real-world pattern
# ---------------------------------------------------------------------------


class _SubModel(Model, legacy=True):
    """Sub-model with one input, one output, one internal, one anonymous index.

    Deliberately uses the legacy flat ``indexes=`` constructor API (it is the
    subject under test for that API, together with ``_ParentModel`` below).
    """

    def __init__(self):
        self.inflow = Index("Total vehicle inflow", None)  # input
        self.traffic = Index("Reference traffic", 200.0)  # output
        self.ratio = Index("Traffic ratio", 0.5)  # level 2
        _base = Index("base constant", 1.0)  # level 3
        super().__init__(
            "Sub Model",
            indexes=[self.inflow, self.traffic, self.ratio, _base],
            inputs=[self.inflow],
            outputs=[self.traffic],
        )


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_submodel_inputs():
    """Sub-model inputs are accessible by the self.* attribute name."""
    m = _SubModel()
    assert m.inputs.inflow is m.inflow


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_submodel_outputs():
    """Sub-model outputs are accessible by the self.* attribute name."""
    m = _SubModel()
    assert m.outputs.traffic is m.traffic


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_submodel_level2_directly_accessible():
    """Level-2 index (self.* but not in outputs) is directly accessible on the model."""
    m = _SubModel()
    assert m.ratio.value == 0.5


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_submodel_indexes_contains_all():
    """Indexes contains all four indexes including the anonymous one."""
    m = _SubModel()
    assert len(m.indexes) == 4


class _ParentModel(Model, legacy=True):
    """Parent that wires a _SubModel output into its own computation.

    Deliberately uses the legacy flat ``indexes=`` constructor API (it is the
    subject under test for that API).
    """

    def __init__(self, sub: _SubModel):
        self.doubled = Index("doubled traffic", sub.outputs.traffic * 2)
        super().__init__(
            "Parent",
            indexes=[*sub.indexes, self.doubled],
            outputs=[self.doubled],
        )


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_submodel_wiring_into_parent():
    """Parent wires sub-model output as input to its own computation."""
    parent = _ParentModel(_SubModel())
    assert parent.outputs.doubled is parent.doubled


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_submodel_outputs_are_subset_of_parent_indexes():
    """Sub-model outputs referenced in parent indexes are findable."""
    parent = _ParentModel(_SubModel())
    index_ids = {id(idx) for idx in parent.indexes}
    for idx in parent.outputs:
        assert id(idx) in index_ids


# ---------------------------------------------------------------------------
# Legacy API — DeprecationWarning
# ---------------------------------------------------------------------------


def test_legacy_indexes_triggers_deprecation_warning():
    """Passing indexes= explicitly emits DeprecationWarning."""
    a = Index("a", 1.0)
    with pytest.warns(DeprecationWarning, match="indexes.*deprecated"):
        Model("test", indexes=[a])


# ---------------------------------------------------------------------------
# New dataclass-based API
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_outputs_only():
    """Model with only Outputs dataclass — inputs and expose are empty."""

    @dataclasses.dataclass
    class Outputs:
        traffic: Index

    a = Index("traffic", 1.0)
    m = Model("test", outputs=Outputs(traffic=a))
    assert m.outputs.traffic is a
    assert len(m.inputs) == 0
    assert len(m.expose) == 0


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_inputs_and_outputs():
    """Inputs and Outputs both declared — accessible by field name."""

    @dataclasses.dataclass
    class Inputs:
        inflow: Index

    @dataclasses.dataclass
    class Outputs:
        traffic: Index

    inflow = Index("inflow", None)
    traffic = Index("traffic", inflow * 2.0)
    m = Model("test", inputs=Inputs(inflow=inflow), outputs=Outputs(traffic=traffic))
    assert m.inputs.inflow is inflow
    assert m.outputs.traffic is traffic


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_expose():
    """Expose dataclass — accessible via m.expose."""

    @dataclasses.dataclass
    class Outputs:
        result: Index

    @dataclasses.dataclass
    class Expose:
        ratio: Index

    result = Index("result", 1.0)
    ratio = Index("ratio", 0.5)
    m = Model("test", outputs=Outputs(result=result), expose=Expose(ratio=ratio))
    assert m.expose.ratio is ratio


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_list_valued_field():
    """List-valued output field is returned as a list."""

    @dataclasses.dataclass
    class Outputs:
        items: list[Index]

    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", outputs=Outputs(items=[a, b]))
    result = m.outputs.items
    assert isinstance(result, list)
    assert result[0] is a
    assert result[1] is b


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_dict_valued_field():
    """Dict-valued output field is returned as a dict."""

    @dataclasses.dataclass
    class Outputs:
        by_class: dict[str, Index]

    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", outputs=Outputs(by_class={"x": a, "y": b}))
    result = m.outputs.by_class
    assert isinstance(result, dict)
    assert result["x"] is a
    assert result["y"] is b


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_indexes_derived_and_deduplicated():
    """Indexes is derived from inputs/outputs/expose; duplicates removed."""

    @dataclasses.dataclass
    class Inputs:
        inflow: Index

    @dataclasses.dataclass
    class Outputs:
        traffic: Index

    @dataclasses.dataclass
    class Expose:
        ratio: Index

    inflow = Index("inflow", None)
    traffic = Index("traffic", 1.0)
    ratio = Index("ratio", 0.5)
    m = Model(
        "test",
        inputs=Inputs(inflow=inflow),
        outputs=Outputs(traffic=traffic),
        expose=Expose(ratio=ratio),
    )
    assert set(id(i) for i in m.indexes) == {id(inflow), id(traffic), id(ratio)}
    assert len(m.indexes) == 3


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_indexes_dedup_shared_input():
    """An index appearing in both inputs and outputs is listed only once."""

    @dataclasses.dataclass
    class Inputs:
        x: Index

    @dataclasses.dataclass
    class Outputs:
        x: Index

    x = Index("x", None)
    m = Model("test", inputs=Inputs(x=x), outputs=Outputs(x=x))
    assert len(m.indexes) == 1
    assert m.indexes[0] is x


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_abstract_indexes():
    """abstract_indexes() works correctly with the new API."""

    @dataclasses.dataclass
    class Inputs:
        placeholder: Index

    @dataclasses.dataclass
    class Outputs:
        result: Index

    p = Index("p", None)
    r = Index("r", 1.0)
    m = Model("test", inputs=Inputs(placeholder=p), outputs=Outputs(result=r))
    result = m.abstract_indexes()
    assert len(result) == 1 and result[0] is p
    assert not m.is_instantiated()


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_proxy_iteration_flattens_list():
    """Iterating over a proxy with a list field yields individual indexes."""

    @dataclasses.dataclass
    class Outputs:
        items: list[Index]

    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", outputs=Outputs(items=[a, b]))
    assert list(m.outputs) == [a, b]


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_proxy_iteration_flattens_dict():
    """Iterating over a proxy with a dict field yields individual index values."""

    @dataclasses.dataclass
    class Outputs:
        by_class: dict[str, Index]

    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", outputs=Outputs(by_class={"x": a, "y": b}))
    assert list(m.outputs) == [a, b]


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_proxy_len_counts_scalars():
    """len(proxy) counts individual scalars across list and scalar fields."""

    @dataclasses.dataclass
    class Outputs:
        scalar: Index
        items: list[Index]

    a = Index("a", 1.0)
    b = Index("b", 2.0)
    c = Index("c", 3.0)
    m = Model("test", outputs=Outputs(scalar=a, items=[b, c]))
    assert len(m.outputs) == 3


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_proxy_contains_works_across_list():
    """In operator finds indexes inside list-valued fields."""

    @dataclasses.dataclass
    class Outputs:
        items: list[Index]

    a = Index("a", 1.0)
    b = Index("b", 2.0)
    outside = Index("x", 9.0)
    m = Model("test", outputs=Outputs(items=[a, b]))
    assert a in m.outputs
    assert b in m.outputs
    assert outside not in m.outputs


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_dataclass_proxy_unknown_attr_raises():
    """Accessing an undeclared field name on a dataclass proxy raises AttributeError."""

    @dataclasses.dataclass
    class Outputs:
        traffic: Index

    m = Model("test", outputs=Outputs(traffic=Index("t", 1.0)))
    with pytest.raises(AttributeError, match="No input/output"):
        _ = m.outputs.nonexistent  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# InputsContractWarning — convention check
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_inputs_contract_warning_fires_for_undeclared_index():
    """Warn when a scalar Index parameter is not in Inputs."""

    class _Bad(Model, legacy=True):
        @dataclasses.dataclass
        class Outputs:
            result: Index

        @dataclasses.dataclass
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
    with pytest.warns(InputsContractWarning, match="'received'"):
        _Bad(received)


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_inputs_contract_warning_fires_for_undeclared_timeseries():
    """Warn when a TimeseriesIndex parameter is not in Inputs."""

    class _Bad(Model, legacy=True):
        @dataclasses.dataclass
        class Outputs:
            out: Index

        @dataclasses.dataclass
        class Expose:
            ts: TimeseriesIndex  # tracked so the model is valid, but not in Inputs

        def __init__(self, ts: TimeseriesIndex) -> None:
            out = Index("out", ts.sum())
            super().__init__("Bad", outputs=_Bad.Outputs(out=out), expose=_Bad.Expose(ts=ts))

    ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
    with pytest.warns(InputsContractWarning, match="'ts'"):
        _Bad(ts)


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_inputs_contract_warning_fires_for_undeclared_list():
    """Warn for each item in a list[Index] parameter not in Inputs."""

    class _Bad(Model, legacy=True):
        @dataclasses.dataclass
        class Outputs:
            total: Index

        @dataclasses.dataclass
        class Expose:
            costs: list  # tracked so the model is valid, but not in Inputs

        def __init__(self, costs: list[Index]) -> None:
            total = Index("total", costs[0] + costs[1])
            super().__init__("Bad", outputs=_Bad.Outputs(total=total), expose=_Bad.Expose(costs=costs))

    costs = [Index("c0", 1.0), Index("c1", 2.0)]
    with pytest.warns(InputsContractWarning) as record:
        _Bad(costs)
    messages = [str(w.message) for w in record]
    assert any("costs[0]" in m for m in messages)
    assert any("costs[1]" in m for m in messages)


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_inputs_contract_no_warning_when_declared():
    """No warning when all Index params are stored in Inputs."""

    class _Good(Model, legacy=True):
        @dataclasses.dataclass
        class Inputs:
            received: Index

        @dataclasses.dataclass
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
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", InputsContractWarning)
        _Good(received)  # must not raise


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_inputs_contract_no_warning_for_non_index_params():
    """No warning for str, float, or ndarray constructor parameters."""

    class _Good(Model, legacy=True):
        @dataclasses.dataclass
        class Outputs:
            result: Index

        def __init__(self, label: str, scale: float, data: np.ndarray) -> None:
            result = Index("result", scale)
            super().__init__("Good", outputs=_Good.Outputs(result=result))

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", InputsContractWarning)
        _Good("hello", 3.14, np.array([1.0]))  # must not raise


def test_inputs_contract_no_warning_for_base_model():
    """No warning when constructing Model directly (only fires for subclasses)."""
    import warnings

    # Model itself has no __init__ parameter convention to check
    with warnings.catch_warnings():
        warnings.simplefilter("error", InputsContractWarning)
        Model("base", outputs=None)  # must not raise


def test_inputs_contract_warning_is_subclass_of_model_contract_warning():
    """InputsContractWarning is a subclass of ModelContractWarning."""
    assert issubclass(InputsContractWarning, ModelContractWarning)


@pytest.mark.xfail(
    reason="uses deprecated API, scheduled for removal (legacy-cleanup)", raises=DeprecationWarning, strict=True
)
def test_model_contract_warning_base_filter_catches_inputs_contract_warning():
    """Filtering on ModelContractWarning catches InputsContractWarning."""

    class _Bad(Model, legacy=True):
        @dataclasses.dataclass
        class Outputs:
            result: Index

        @dataclasses.dataclass
        class Expose:
            received: Index

        def __init__(self, received: Index) -> None:
            result = Index("result", received + 1.0)
            super().__init__("Bad", outputs=_Bad.Outputs(result=result), expose=_Bad.Expose(received=received))

    received = Index("x", 1.0)
    with pytest.warns(ModelContractWarning):
        _Bad(received)


# ---------------------------------------------------------------------------
# _check_inputs_contract — dict-valued parameter path
# ---------------------------------------------------------------------------


def test_inputs_contract_warning_fires_for_undeclared_dict():
    """InputsContractWarning names each missing key when the parameter is a dict."""
    import warnings

    class _Bad(Model, legacy=True):
        @dataclasses.dataclass
        class Outputs:
            result: Index

        def __init__(self, mapping: dict) -> None:
            result = Index("result", 1.0)
            super().__init__("Bad", outputs=_Bad.Outputs(result=result))

    x = Index("x", 1.0)
    y = Index("y", 2.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _Bad(mapping={"a": x, "b": y})

    contract_warnings = [w for w in caught if issubclass(w.category, InputsContractWarning)]
    assert len(contract_warnings) == 2
    messages = [str(w.message) for w in contract_warnings]
    assert any("mapping['a']" in m for m in messages)
    assert any("mapping['b']" in m for m in messages)


def test_inputs_contract_no_warning_for_declared_dict():
    """No InputsContractWarning when all dict-valued GenericIndex entries are in Inputs."""
    import warnings

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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _Good(mapping={"a": x})

    contract_warnings = [w for w in caught if issubclass(w.category, InputsContractWarning)]
    assert len(contract_warnings) == 0


# ---------------------------------------------------------------------------
# _check_inputs_contract — inspect.signature exception path
# ---------------------------------------------------------------------------


def test_inputs_contract_no_crash_when_signature_unavailable():
    """_check_inputs_contract silently returns when inspect.signature raises."""
    import unittest.mock
    import warnings

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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Must not raise — the contract check should be silently skipped.
            _Model(x)

    contract_warnings = [w for w in caught if issubclass(w.category, InputsContractWarning)]
    assert len(contract_warnings) == 0


def test_inputs_contract_skips_params_absent_from_locals():
    """_check_inputs_contract silently skips parameters deleted before super().__init__().

    This covers the ``value is inspect.Parameter.empty`` branch — hit when a
    parameter declared in the signature is removed from local scope (via ``del``)
    before ``super().__init__()`` is called.
    """
    import warnings

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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Must not raise even though 'y' is absent from f_locals.
        _ModelWithDeletedParam(x, y)

    # No warning fires: x is declared in Inputs; y was deleted before inspection.
    contract_warnings = [w for w in caught if issubclass(w.category, InputsContractWarning)]
    assert len(contract_warnings) == 0


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
            self.inner = InnerModel(InnerModel.Inputs(k=k))
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
            self.inner = _Inner(_Inner.Inputs(alpha=alpha, beta=beta))
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
            domain_indexes: list[Index]

        def compute(self, inp: Inputs) -> tuple[Outputs, Expose]:
            k = Index("k", 0.5)
            inner = _InnerOK(_InnerOK.Inputs(k=k))
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
            self.inner = _InnerConst(_InnerConst.Inputs(k=k))
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
