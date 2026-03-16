"""Tests for civic_digital_twins.dt_model.model.Model."""

# SPDX-License-Identifier: Apache-2.0

from typing import cast

import numpy as np
import pytest
from scipy import stats

from civic_digital_twins.dt_model.model.index import Distribution, Index, TimeseriesIndex
from civic_digital_twins.dt_model.model.model import IOProxy, Model

c1 = cast(Distribution, stats.norm(loc=2.0, scale=1.0))


# ---------------------------------------------------------------------------
# Backward compatibility — existing Model behaviour unchanged
# ---------------------------------------------------------------------------


def test_model_stores_name_and_indexes():
    """Model stores its name and index list."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", [a, b])
    assert m.name == "test"
    assert m.indexes == [a, b]


def test_model_abstract_indexes_empty_when_all_concrete():
    """abstract_indexes() is empty when all indexes have concrete values."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    m = Model("test", [a, b])
    assert m.abstract_indexes() == []
    assert m.is_instantiated()


def test_model_abstract_indexes_includes_none_value():
    """abstract_indexes() includes placeholder (value=None) indexes."""
    a = Index("a", 1.0)
    p = Index("p", None)
    m = Model("test", [a, p])
    assert m.abstract_indexes() == [p]
    assert not m.is_instantiated()


def test_model_abstract_indexes_includes_distribution_value():
    """abstract_indexes() includes distribution-backed indexes."""
    a = Index("a", 1.0)
    d = Index("d", c1)
    m = Model("test", [a, d])
    assert m.abstract_indexes() == [d]
    assert not m.is_instantiated()


def test_model_abstract_indexes_includes_timeseries_placeholder():
    """abstract_indexes() includes TimeseriesIndex placeholders."""
    ts = TimeseriesIndex("ts")
    tf = TimeseriesIndex("tf", np.array([1.0, 2.0]))
    m = Model("test", [ts, tf])
    assert m.abstract_indexes() == [ts]
    assert not m.is_instantiated()


def test_model_formula_index_is_concrete():
    """A formula-based Index (node wrapping) is not abstract."""
    from civic_digital_twins.dt_model.engine.frontend import graph

    n = graph.constant(3.0)
    idx = Index("formula", n)
    m = Model("test", [idx])
    assert m.abstract_indexes() == []
    assert m.is_instantiated()


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


class _TwoIndexModel(Model):
    """Minimal model with one input and one output."""

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


def test_model_outputs_accessible_by_attr_name():
    """Outputs accessible by the Python attribute name, not index.name."""
    m = _TwoIndexModel()
    assert m.outputs.result is m.result


def test_model_inputs_accessible_by_attr_name():
    """Inputs accessible by the Python attribute name, not index.name."""
    m = _TwoIndexModel()
    assert m.inputs.param is m.param


def test_model_index_name_irrelevant_to_proxy():
    """index.name is a display label — it plays no role in proxy access."""
    m = _TwoIndexModel()
    # m.result has index.name == "Result B", but proxy key is "result"
    assert cast(Index, m.outputs.result).name == "Result B"
    with pytest.raises(AttributeError):
        _ = m.outputs.Result_B  # type: ignore[attr-defined]


def test_model_level2_index_directly_accessible():
    """Level-2 index (self.* but not in outputs) is directly accessible."""
    m = _TwoIndexModel()
    assert cast(Index, m.internal).value == 1.0


def test_model_level2_index_not_in_outputs():
    """Level-2 index is not reachable via outputs."""
    m = _TwoIndexModel()
    assert m.internal not in m.outputs


def test_model_level3_index_in_indexes():
    """Level-3 (anonymous) index is in indexes but not reachable by name."""
    m = _TwoIndexModel()
    # The hidden index is in indexes (engine needs it) but has no attr on m.
    hidden_values = [cast(Index, idx).value for idx in m.indexes if cast(Index, idx).value == 0.0]
    assert hidden_values == [0.0]


def test_model_inputs_and_outputs_are_subsets_of_indexes():
    """Every input and output is also in indexes."""
    m = _TwoIndexModel()
    index_ids = {id(idx) for idx in m.indexes}
    for idx in m.inputs:
        assert id(idx) in index_ids
    for idx in m.outputs:
        assert id(idx) in index_ids


def test_model_same_index_in_inputs_and_outputs():
    """The same index may appear in both inputs and outputs (pass-through)."""

    class _PassThrough(Model):
        def __init__(self):
            self.x = Index("x", None)
            super().__init__("pt", indexes=[self.x], inputs=[self.x], outputs=[self.x])

    m = _PassThrough()
    assert m.inputs.x is m.x
    assert m.outputs.x is m.x


def test_model_inputs_and_outputs_independent():
    """Inputs and outputs are independent views; overlap is allowed."""

    class _IO(Model):
        def __init__(self):
            self.a = Index("a", None)
            self.b = Index("b", 1.0)
            super().__init__("io", indexes=[self.a, self.b], inputs=[self.a], outputs=[self.b])

    m = _IO()
    assert m.inputs.a is m.a
    assert m.outputs.b is m.b


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_model_outputs_not_in_indexes_raises():
    """Outputs entry not in indexes raises ValueError."""
    a = Index("a", 1.0)
    b = Index("b", 2.0)
    with pytest.raises(ValueError, match="not in the model's indexes"):
        Model("test", [a], outputs=[b])


def test_model_inputs_not_in_indexes_raises():
    """Inputs entry not in indexes raises ValueError."""
    a = Index("a", 1.0)
    b = Index("b", None)
    with pytest.raises(ValueError, match="not in the model's indexes"):
        Model("test", [a], inputs=[b])


def test_model_outputs_not_assigned_to_self_raises():
    """Outputs entry not assigned to self.* raises ValueError."""

    class _Bad(Model):
        def __init__(self):
            local = Index("local", 1.0)
            super().__init__("bad", indexes=[local], outputs=[local])

    with pytest.raises(ValueError, match="not assigned to any attribute"):
        _Bad()


def test_model_inputs_not_assigned_to_self_raises():
    """Inputs entry not assigned to self.* raises ValueError."""

    class _Bad(Model):
        def __init__(self):
            local = Index("local", None)
            super().__init__("bad", indexes=[local], inputs=[local])

    with pytest.raises(ValueError, match="not assigned to any attribute"):
        _Bad()


def test_model_outputs_collision_raises():
    """Declaring the same self.* index twice in outputs raises ValueError."""

    class _Bad(Model):
        def __init__(self):
            self.x = Index("x", 1.0)
            # Declaring self.x twice — both entries resolve to attr name "x".
            super().__init__("bad", indexes=[self.x], outputs=[self.x, self.x])

    with pytest.raises(ValueError, match="collision"):
        _Bad()


# ---------------------------------------------------------------------------
# Subclass usage — typical real-world pattern
# ---------------------------------------------------------------------------


class _SubModel(Model):
    """Sub-model with one input, one output, one internal, one anonymous index."""

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


def test_submodel_inputs():
    """Sub-model inputs are accessible by the self.* attribute name."""
    m = _SubModel()
    assert m.inputs.inflow is m.inflow


def test_submodel_outputs():
    """Sub-model outputs are accessible by the self.* attribute name."""
    m = _SubModel()
    assert m.outputs.traffic is m.traffic


def test_submodel_level2_directly_accessible():
    """Level-2 index (self.* but not in outputs) is directly accessible on the model."""
    m = _SubModel()
    assert m.ratio.value == 0.5


def test_submodel_indexes_contains_all():
    """Indexes contains all four indexes including the anonymous one."""
    m = _SubModel()
    assert len(m.indexes) == 4


def test_submodel_wiring_into_parent():
    """Parent wires sub-model output as input to its own computation."""
    sub = _SubModel()

    class _Parent(Model):
        def __init__(self, sub: _SubModel):
            # Wire sub output into parent computation.
            self.doubled = Index("doubled traffic", sub.outputs.traffic * 2)
            super().__init__(
                "Parent",
                indexes=[*sub.indexes, self.doubled],
                outputs=[self.doubled],
            )

    parent = _Parent(sub)
    assert parent.outputs.doubled is parent.doubled


def test_submodel_outputs_are_subset_of_parent_indexes():
    """Sub-model outputs referenced in parent indexes are findable."""
    sub = _SubModel()

    class _Parent(Model):
        def __init__(self, sub: _SubModel):
            self.doubled = Index("doubled", sub.outputs.traffic * 2)
            super().__init__(
                "Parent",
                indexes=[*sub.indexes, self.doubled],
                outputs=[self.doubled],
            )

    parent = _Parent(sub)
    index_ids = {id(idx) for idx in parent.indexes}
    for idx in parent.outputs:
        assert id(idx) in index_ids
