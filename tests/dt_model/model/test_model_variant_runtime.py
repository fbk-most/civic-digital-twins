"""Tests for ModelVariant runtime (CategoricalIndex / graph.Node selector) mode."""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from civic_digital_twins.dt_model import DistributionEnsemble, Evaluation, Scenario, define, expose, inputs, outputs
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.model.index import CategoricalIndex, ConstIndex, Index
from civic_digital_twins.dt_model.model.model import IOProxy, Model
from civic_digital_twins.dt_model.model.model_variant import ModelVariant

# ---------------------------------------------------------------------------
# Shared test models (same I/O contract)
# ---------------------------------------------------------------------------


@define("BikeModel")
class _BikeModel(Model):
    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        throughput: Index
        emissions: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute throughput/emissions for the bike variant."""
        throughput = Index("throughput", inputs.capacity * 1.0)
        emissions = Index("emissions", 0.0)
        return _BikeModel.Outputs(throughput=throughput, emissions=emissions)


@define("TrainModel")
class _TrainModel(Model):
    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        throughput: Index
        emissions: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute throughput/emissions for the train variant."""
        throughput = Index("throughput", inputs.capacity * 10.0)
        emissions = Index("emissions", 50.0)
        return _TrainModel.Outputs(throughput=throughput, emissions=emissions)


def _make_variants() -> dict[str, Model]:
    return {
        "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=Index("capacity", 100.0))),
        "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=Index("capacity", 500.0))),
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


def test_runtime_inputs_is_disjoint_union_keyed_by_variant():
    """In runtime mode too, inputs is a dict keyed by variant, never merged by field name."""
    mode = CategoricalIndex("mode", {"bike": 0.4, "train": 0.6})
    cap_bike = Index("capacity", 100.0)
    cap_train = Index("capacity", 500.0)
    bike = _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike))
    train = _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train))
    mv = ModelVariant(
        "Transport",
        {"bike": bike, "train": train},
        selector=mode,
    )
    assert set(mv.inputs) == {"bike", "train"}
    assert mv.inputs["bike"].capacity is cap_bike
    assert mv.inputs["train"].capacity is cap_train


def test_runtime_abstract_indexes_includes_categorical():
    """abstract_indexes() in runtime mode includes the CategoricalIndex."""
    mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})
    mv = ModelVariant("Transport", _make_variants(), selector=mode)
    abstract = mv.abstract_indexes()
    assert any(idx is mode for idx in abstract)


def test_runtime_abstract_indexes_includes_variant_abstract_indexes():
    """abstract_indexes() in runtime mode includes variants' own abstract indexes."""
    mode = CategoricalIndex("mode", {"bike": 0.5, "train": 0.5})
    cap_placeholder = Index("capacity", None)  # abstract
    mv = ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_placeholder)),
            "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=Index("capacity", 500.0))),
        },
        selector=mode,
    )
    abstract = mv.abstract_indexes()
    assert any(idx is cap_placeholder for idx in abstract)


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


def _make_variants_with_expose() -> tuple[dict[str, Model], dict[str, float]]:
    """Two single-output variants, each exposing a distinct constant via `diag`."""

    @define("A")
    class _A(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            y: Index

        @expose
        class Expose:
            diag: Index

        def compute(self, inputs: "_A.Inputs") -> tuple["_A.Outputs", "_A.Expose"]:
            return _A.Outputs(y=Index("y", 1.0)), _A.Expose(diag=ConstIndex("diag", 111.0))

    @define("B")
    class _B(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            y: Index

        @expose
        class Expose:
            diag: Index

        def compute(self, inputs: "_B.Inputs") -> tuple["_B.Outputs", "_B.Expose"]:
            return _B.Outputs(y=Index("y", 2.0)), _B.Expose(diag=ConstIndex("diag", 999.0))

    return {"a": _A(inputs=_A.Inputs()), "b": _B(inputs=_B.Inputs())}, {"a": 111.0, "b": 999.0}


def test_runtime_expose_field_backed_by_merged_node():
    """Regression test for #252: mv.expose.field is a merged node, like mv.outputs.field.

    Before the fix, the runtime-mode `expose` property returned an arbitrary
    variant's own Index directly — not backed by any dispatch node at all.
    """
    variants, _ = _make_variants_with_expose()
    mode = CategoricalIndex("mode", {"a": 0.5, "b": 0.5})
    mv = ModelVariant("Group", variants, selector=mode)
    assert isinstance(mv.expose.diag.node, graph.exclusive_multi_clause_where)


def test_runtime_expose_included_in_indexes():
    """Regression test for #252: the merged expose index is reachable via mv.indexes."""
    variants, _ = _make_variants_with_expose()
    mode = CategoricalIndex("mode", {"a": 0.5, "b": 0.5})
    mv = ModelVariant("Group", variants, selector=mode)
    idx_ids = {id(idx) for idx in mv.indexes}
    assert id(mv.expose.diag) in idx_ids


def test_runtime_expose_dispatches_per_scenario():
    """Regression test for #252: mv.expose.field tracks the selector, per scenario.

    Reproduces the original bug report: two variants exposing distinct
    constants via a common `diag` field; a genuinely mixed CategoricalIndex
    selector across 20 scenarios. Before the fix, `result[mv.expose.diag]`
    was constant (one arbitrary variant's value) regardless of selection;
    after the fix it must track `result[mv._selector_index]` exactly.
    """
    variants, expected_by_key = _make_variants_with_expose()
    mode = CategoricalIndex("mode", {"a": 0.5, "b": 0.5})
    mv = ModelVariant("Group", variants, selector=mode)

    scenario = Scenario(mv)
    ensemble = DistributionEnsemble(scenario, size=20, rng=np.random.default_rng(0))
    result = Evaluation(scenario).evaluate(ensemble=ensemble)

    selected_keys = result[mv._selector_index].ravel()
    diag_values = result[mv.expose.diag].ravel()

    # Guard against a degenerate sample that happens to pick only one variant.
    assert set(selected_keys) == {"a", "b"}

    expected = np.array([expected_by_key[key] for key in selected_keys])
    np.testing.assert_array_equal(diag_values, expected)


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
    selector = ModelVariant.guards_to_selector(
        [
            ("train", cost.node > graph.constant(5.0)),
            ("bike", True),
        ]
    )
    assert isinstance(selector, graph.Node)


def test_node_selector_runtime_mode():
    """A graph.Node selector creates runtime mode."""
    cost = Index("cost", graph.placeholder("cost"))
    selector = ModelVariant.guards_to_selector(
        [
            ("train", cost.node > graph.constant(5.0)),
            ("bike", True),
        ]
    )
    mv = ModelVariant("Transport", _make_variants(), selector=selector)
    assert isinstance(mv.outputs.throughput.node, graph.exclusive_multi_clause_where)


def test_node_selector_abstract_indexes_does_not_include_selector_node():
    """A graph.Node selector does not add a new abstract index — its deps are variants'."""
    cost = Index("cost", graph.placeholder("cost"))
    selector = ModelVariant.guards_to_selector(
        [
            ("train", cost.node > graph.constant(5.0)),
            ("bike", True),
        ]
    )
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
    bike = _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike))
    train = _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train))
    mv = ModelVariant(
        "Transport",
        {"bike": bike, "train": train},
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


def test_runtime_merged_outputs_nestable_as_bulk_field():
    """Runtime mode's merged mv.outputs can be nested as a whole field too.

    Regression test: the merged-graph IOProxy built in ModelVariant.__init__ previously had
    no `dc=`, so contract validation rejected nesting it in a parent's own @expose field
    ("Surfacing sub-model diagnostics in bulk" — dd-cdt-modularity.md). Uses single-output,
    ConstIndex-only variants (rather than _BikeModel/_TrainModel) so the only thing under test
    is the nesting itself, not unrelated orphaned-placeholder bookkeeping for a deeper composite.
    """
    from civic_digital_twins.dt_model.model.index import ConstIndex  # noqa: PLC0415

    @define("A")
    class _A(Model):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            y: Index

        def compute(self, inputs: "_A.Inputs") -> "_A.Outputs":
            return _A.Outputs(y=Index("y", inputs.x * 1.0))

    @define("B")
    class _B(Model):
        @inputs
        class Inputs:
            x: Index

        @outputs
        class Outputs:
            y: Index

        def compute(self, inputs: "_B.Inputs") -> "_B.Outputs":
            return _B.Outputs(y=Index("y", inputs.x * 10.0))

    mode = CategoricalIndex("mode", {"a": 0.5, "b": 0.5})
    x = ConstIndex("x", 1.0)
    mv = ModelVariant(
        "Group",
        {"a": _A(inputs=_A.Inputs(x=x)), "b": _B(inputs=_B.Inputs(x=x))},
        selector=mode,
    )

    @define("RootBulk")
    class _RootBulk(Model):
        @inputs
        class Inputs:
            mode: CategoricalIndex
            x: Index

        @outputs
        class Outputs:
            placeholder: Index

        @expose
        class Expose:
            sub_out: IOProxy

        def compute(self, inputs: "_RootBulk.Inputs") -> tuple["_RootBulk.Outputs", "_RootBulk.Expose"]:
            return _RootBulk.Outputs(placeholder=Index("p", 0.0)), _RootBulk.Expose(sub_out=mv.outputs)

    root = _RootBulk(inputs=_RootBulk.Inputs(mode=mode, x=x))
    assert root.expose.sub_out.y is mv.outputs.y
