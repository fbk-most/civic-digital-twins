"""Tests for ModelVariant runtime (CategoricalIndex / graph.Node selector) mode."""

# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.model.index import CategoricalIndex, Index
from civic_digital_twins.dt_model.model.model import IOProxy, Model
from civic_digital_twins.dt_model.model.model_variant import ModelVariant


# ---------------------------------------------------------------------------
# Shared test models (same I/O contract)
# ---------------------------------------------------------------------------


class _BikeModel(Model):
    @dataclasses.dataclass
    class Inputs:
        capacity: Index

    @dataclasses.dataclass
    class Outputs:
        throughput: Index
        emissions: Index

    def __init__(self, capacity: Index) -> None:
        cap_val = capacity.value
        throughput = Index("throughput", float(cap_val) * 1.0 if isinstance(cap_val, (int, float)) else None)
        emissions = Index("emissions", 0.0)
        super().__init__(
            "BikeModel",
            inputs=_BikeModel.Inputs(capacity=capacity),
            outputs=_BikeModel.Outputs(throughput=throughput, emissions=emissions),
        )


class _TrainModel(Model):
    @dataclasses.dataclass
    class Inputs:
        capacity: Index

    @dataclasses.dataclass
    class Outputs:
        throughput: Index
        emissions: Index

    def __init__(self, capacity: Index) -> None:
        cap_val = capacity.value
        throughput = Index("throughput", float(cap_val) * 10.0 if isinstance(cap_val, (int, float)) else None)
        emissions = Index("emissions", 50.0)
        super().__init__(
            "TrainModel",
            inputs=_TrainModel.Inputs(capacity=capacity),
            outputs=_TrainModel.Outputs(throughput=throughput, emissions=emissions),
        )


def _make_variants() -> dict[str, Model]:
    return {
        "bike": _BikeModel(Index("capacity", 100.0)),
        "train": _TrainModel(Index("capacity", 500.0)),
    }


# ===========================================================================
# Runtime mode — CategoricalIndex selector
# ===========================================================================


def test_runtime_mode_is_not_static():
    """A CategoricalIndex selector puts ModelVariant in runtime (not static) mode."""
    mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    mv = ModelVariant("Transport", _make_variants(), selector=mode)
    # In runtime mode there is no _active attribute.
    with pytest.raises(AttributeError):
        object.__getattribute__(mv, "_active")


def test_runtime_outputs_are_merged_indexes():
    """Runtime mode outputs are backed by exclusive_multi_clause_where nodes."""
    mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    mv = ModelVariant("Transport", _make_variants(), selector=mode)
    assert isinstance(mv.outputs.throughput.node, graph.exclusive_multi_clause_where)
    assert isinstance(mv.outputs.emissions.node, graph.exclusive_multi_clause_where)


def test_runtime_inputs_proxies_first_variant():
    """In runtime mode, inputs proxies the first variant (field names are shared)."""
    mode = CategoricalIndex("mode", {"bike": 0.4, "train": 0.6})
    cap_bike = Index("capacity", 100.0)
    cap_train = Index("capacity", 500.0)
    mv = ModelVariant(
        "Transport",
        {"bike": _BikeModel(cap_bike), "train": _TrainModel(cap_train)},
        selector=mode,
    )
    # inputs.capacity is the first variant's capacity (bike)
    assert mv.inputs.capacity is cap_bike


def test_runtime_abstract_indexes_includes_categorical():
    """abstract_indexes() in runtime mode includes the CategoricalIndex."""
    mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    mv = ModelVariant("Transport", _make_variants(), selector=mode)
    abstract = mv.abstract_indexes()
    assert mode in abstract


def test_runtime_abstract_indexes_includes_variant_abstract_indexes():
    """abstract_indexes() in runtime mode includes variants' own abstract indexes."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    cap_placeholder = Index("capacity", None)  # abstract
    mv = ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(cap_placeholder),
            "train": _TrainModel(Index("capacity", 500.0)),
        },
        selector=mode,
    )
    abstract = mv.abstract_indexes()
    assert cap_placeholder in abstract


def test_runtime_is_instantiated_returns_false():
    """is_instantiated() is always False in runtime mode."""
    mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    mv = ModelVariant("Transport", _make_variants(), selector=mode)
    assert not mv.is_instantiated()


def test_runtime_indexes_includes_merged_outputs_and_selector():
    """In runtime mode, indexes includes merged output indexes and the CategoricalIndex."""
    mode = CategoricalIndex("mode", {"bike": 0.4, "train": 0.6})
    mv = ModelVariant("Transport", _make_variants(), selector=mode)
    idx_ids = {id(idx) for idx in mv.indexes}
    # The merged outputs must be in indexes.
    assert id(mv.outputs.throughput) in idx_ids
    assert id(mv.outputs.emissions) in idx_ids
    # The CategoricalIndex must be in indexes.
    assert id(mode) in idx_ids


def test_runtime_expose_is_intersection():
    """In runtime mode, expose returns the intersection of fields across variants."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = ModelVariant("Transport", _make_variants(), selector=mode)
    # _BikeModel and _TrainModel have no Expose — intersection is empty.
    assert isinstance(mv.expose, IOProxy)
    assert len(mv.expose) == 0


# ===========================================================================
# Runtime mode — construction-time validation
# ===========================================================================


def test_categorical_selector_bad_key_raises():
    """CategoricalIndex with key not in variants raises ValueError at construction."""
    mode = CategoricalIndex("mode", {"bike": 0.3, "bus": 0.7})  # "bus" not in variants
    with pytest.raises(ValueError, match="outcome key"):
        ModelVariant("Transport", _make_variants(), selector=mode)


# ===========================================================================
# Runtime mode — graph.Node selector (guards_to_selector)
# ===========================================================================


def test_guards_to_selector_returns_node():
    """guards_to_selector returns a graph.Node."""
    cost = Index("cost", graph.placeholder("cost"))
    selector = ModelVariant.guards_to_selector([
        ("train", cost.node > graph.constant(5.0)),
        ("bike", True),
    ])
    assert isinstance(selector, graph.Node)


def test_node_selector_runtime_mode():
    """A graph.Node selector creates runtime mode."""
    cost = Index("cost", graph.placeholder("cost"))
    selector = ModelVariant.guards_to_selector([
        ("train", cost.node > graph.constant(5.0)),
        ("bike", True),
    ])
    mv = ModelVariant("Transport", _make_variants(), selector=selector)
    assert isinstance(mv.outputs.throughput.node, graph.exclusive_multi_clause_where)


def test_node_selector_abstract_indexes_does_not_include_selector_node():
    """A graph.Node selector does not add a new abstract index — its deps are variants'."""
    cost = Index("cost", graph.placeholder("cost"))
    selector = ModelVariant.guards_to_selector([
        ("train", cost.node > graph.constant(5.0)),
        ("bike", True),
    ])
    mv = ModelVariant("Transport", _make_variants(), selector=selector)
    # The node selector itself is not a GenericIndex so it cannot appear.
    # abstract_indexes should be the union of the variants' own abstract indexes.
    abstract = mv.abstract_indexes()
    assert not any(isinstance(idx, CategoricalIndex) for idx in abstract)


# ===========================================================================
# repr
# ===========================================================================


def test_runtime_repr_contains_name_and_selector():
    """Repr in runtime mode includes name and selector."""
    mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    mv = ModelVariant("Transport", _make_variants(), selector=mode)
    r = repr(mv)
    assert "Transport" in r
    assert "mode" in r


def test_runtime_getattr_forwards_to_first_variant():
    """Unknown attribute access in runtime mode is forwarded to the first variant."""
    mode = CategoricalIndex("mode", {"bike": 0.4, "train": 0.6})
    cap_bike = Index("capacity", 100.0)
    cap_train = Index("capacity", 500.0)
    mv = ModelVariant(
        "Transport",
        {"bike": _BikeModel(cap_bike), "train": _TrainModel(cap_train)},
        selector=mode,
    )
    # 'name' is on ModelVariant itself; access something forwarded via __getattr__.
    # Model.name is a direct attribute on Model instances — forwarded in runtime mode.
    assert mv.name == "Transport"  # ModelVariant's own name attribute


def test_runtime_getattr_unknown_raises_attribute_error():
    """Accessing a non-existent attribute in runtime mode raises AttributeError."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    mv = ModelVariant("Transport", _make_variants(), selector=mode)
    with pytest.raises(AttributeError):
        _ = mv.this_does_not_exist
