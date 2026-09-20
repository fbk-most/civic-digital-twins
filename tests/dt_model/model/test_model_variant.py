"""Tests for civic_digital_twins.dt_model.model.ModelVariant."""

# SPDX-License-Identifier: Apache-2.0

import pytest

from civic_digital_twins.dt_model import NumpyBackend, define, expose, inputs, outputs
from civic_digital_twins.dt_model.model.index import Index
from civic_digital_twins.dt_model.model.model import IOProxy, Model
from civic_digital_twins.dt_model.model.model_variant import ModelVariant
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from civic_digital_twins.dt_model.simulation.scenario import Scenario

# ---------------------------------------------------------------------------
# Shared fixtures — two concrete Model subclasses with the same I/O contract
# ---------------------------------------------------------------------------


@define("BikeModel")
class _BikeModel(Model):
    """Variant A — bicycle-mode transport model."""

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
    """Variant B — rail-mode transport model."""

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
    """Build a fresh variants dict with independent Index objects."""
    cap_bike = Index("capacity", 100.0)
    cap_train = Index("capacity", 500.0)
    return {
        "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike)),
        "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train)),
    }


# ---------------------------------------------------------------------------
# Helper: Model with mismatched I/O field names
# ---------------------------------------------------------------------------


@define("OtherOutputsModel")
class _OtherOutputsModel(Model):
    """Model whose outputs field names differ from _BikeModel / _TrainModel."""

    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        flow: Index  # renamed field
        co2: Index  # renamed field

    def compute(self, inputs: Inputs) -> Outputs:
        """Return fixed flow/co2 outputs."""
        return _OtherOutputsModel.Outputs(flow=Index("flow", 1.0), co2=Index("co2", 2.0))


@define("ExtraInputModel")
class _ExtraInputModel(Model):
    """Model with an additional input field not present in _BikeModel / _TrainModel."""

    @inputs
    class Inputs:
        capacity: Index
        bonus: Index  # extra field absent from _BikeModel / _TrainModel

    @outputs
    class Outputs:
        throughput: Index
        emissions: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Return fixed throughput/emissions outputs."""
        return _ExtraInputModel.Outputs(throughput=Index("t", 1.0), emissions=Index("e", 0.0))


# ===========================================================================
# Static string selector
# ===========================================================================


def test_static_selector_instantiates_correct_variant():
    """String selector picks the correct variant."""
    mv = ModelVariant("Transport", _make_variants(), selector="bike")
    assert mv.name == "Transport"
    assert isinstance(mv.variants["bike"], _BikeModel)
    assert isinstance(mv.variants["train"], _TrainModel)


def test_static_selector_bike_outputs():
    """Active variant's outputs are accessible through ModelVariant."""
    mv = ModelVariant("Transport", _make_variants(), selector="bike")
    # BikeModel sets emissions=0.0
    assert mv.outputs.emissions.concrete_default == 0.0


def test_static_selector_train_outputs():
    """Static 'train' selector delegates to TrainModel."""
    mv = ModelVariant("Transport", _make_variants(), selector="train")
    # TrainModel sets emissions=50.0
    assert mv.outputs.emissions.concrete_default == 50.0


def test_static_selector_unknown_key_raises():
    """Unknown string selector key raises ValueError."""
    with pytest.raises(ValueError, match="does not match any"):
        ModelVariant("Transport", _make_variants(), selector="bus")


def test_non_string_selector_raises_value_error():
    """Passing an invalid selector type raises ValueError."""
    with pytest.raises(ValueError, match="selector must be a str"):
        ModelVariant("Transport", _make_variants(), selector=42)  # type: ignore[arg-type]


# ===========================================================================
# Attribute proxying — inputs, outputs, expose, indexes
# ===========================================================================


def test_inputs_is_disjoint_union_keyed_by_variant():
    """mv.inputs is a dict keyed by variant, never merged by field name."""
    cap_bike = Index("capacity", 100.0)
    cap_train = Index("capacity", 500.0)
    variants = {
        "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike)),
        "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train)),
    }
    mv = ModelVariant("Transport", variants, selector="bike")
    assert set(mv.inputs) == {"bike", "train"}
    # Both variants' own values are reachable, unmerged — including the inactive one.
    assert mv.inputs["bike"].capacity is cap_bike
    assert mv.inputs["train"].capacity is cap_train
    assert mv.inputs["train"] is mv.variants["train"].inputs


def test_outputs_proxy_delegates_to_active_variant():
    """Outputs proxy delegates to the active variant."""
    variants = _make_variants()
    mv = ModelVariant("Transport", variants, selector="train")
    # TrainModel throughput = capacity * 10 = 500 * 10 = 5000
    result = Evaluation(Scenario(mv)).evaluate(backend=NumpyBackend)
    assert float(result[mv.outputs.throughput]) == pytest.approx(5000.0)


def test_expose_proxy_delegates_to_active_variant():
    """Expose proxy is empty when the variant declares no Expose."""
    mv = ModelVariant("Transport", _make_variants(), selector="bike")
    from civic_digital_twins.dt_model.model.model import IOProxy

    assert isinstance(mv.expose, IOProxy)
    assert len(mv.expose) == 0


def test_indexes_delegates_to_active_variant_only():
    """Indexes list contains only the active variant's indexes."""
    cap_bike = Index("capacity", 100.0)
    cap_train = Index("capacity", 500.0)
    bike = _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike))
    train = _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train))
    mv = ModelVariant("Transport", {"bike": bike, "train": train}, selector="bike")

    # The bike model's indexes must all appear (identity check — __eq__ returns a Node).
    mv_index_ids = {id(idx) for idx in mv.indexes}
    for idx in bike.indexes:
        assert id(idx) in mv_index_ids

    # The train model's indexes must NOT appear (they belong to the inactive variant).
    for idx in train.indexes:
        assert id(idx) not in mv_index_ids


def test_inactive_variant_indexes_accessible_via_variants_key():
    """Inactive variant's indexes are reachable via variants["key"]."""
    cap_bike = Index("capacity", 100.0)
    cap_train = Index("capacity", 500.0)
    train = _TrainModel(inputs=_TrainModel.Inputs(capacity=cap_train))
    bike = _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike))
    mv = ModelVariant("Transport", {"bike": bike, "train": train}, selector="bike")

    # The train capacity is NOT in mv.indexes (identity check — __eq__ returns a Node).
    mv_index_ids = {id(idx) for idx in mv.indexes}
    assert id(cap_train) not in mv_index_ids
    # ... but is accessible via the explicit path.
    assert mv.variants["train"].inputs.capacity is cap_train


def test_abstract_indexes_delegates_to_active_variant():
    """abstract_indexes() delegates to the active variant."""
    cap_placeholder = Index("capacity", None)
    variants = {
        "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_placeholder)),
        "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=Index("capacity", 500.0))),
    }
    mv = ModelVariant("Transport", variants, selector="bike")
    abstract = mv.abstract_indexes()
    # cap_placeholder has value=None, so it is abstract.
    assert any(idx is cap_placeholder for idx in abstract)


def test_is_instantiated_delegates_to_active_variant():
    """is_instantiated() delegates to the active variant."""
    cap_concrete = Index("capacity", 100.0)
    variants = {
        "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_concrete)),
        "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=Index("capacity", 500.0))),
    }
    mv = ModelVariant("Transport", variants, selector="bike")
    assert mv.is_instantiated()


def test_is_not_instantiated_when_active_has_placeholder():
    """is_instantiated() returns False when the active variant has a placeholder."""
    cap_placeholder = Index("capacity", None)
    variants = {
        "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_placeholder)),
        "train": _TrainModel(inputs=_TrainModel.Inputs(capacity=Index("capacity", 500.0))),
    }
    mv = ModelVariant("Transport", variants, selector="bike")
    assert not mv.is_instantiated()


# ===========================================================================
# Direct attribute access (transparency)
# ===========================================================================


def test_direct_attribute_access_forwards_to_active_variant():
    """Attribute access for unknown names is forwarded to the active Model."""
    cap_bike = Index("capacity", 100.0)
    bike = _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike))
    train = _TrainModel(inputs=_TrainModel.Inputs(capacity=Index("capacity", 500.0)))
    mv = ModelVariant("Transport", {"bike": bike, "train": train}, selector="bike")
    # 'name' is defined directly on ModelVariant, not proxied.
    assert mv.name == "Transport"
    # outputs is a property on ModelVariant — check field-level forwarding via proxy.
    assert mv.outputs.throughput is not None


def test_unknown_attribute_raises_attribute_error():
    """Accessing a non-existent attribute raises AttributeError."""
    mv = ModelVariant("Transport", _make_variants(), selector="bike")
    with pytest.raises(AttributeError):
        _ = mv.nonexistent_field


# ===========================================================================
# variants dict
# ===========================================================================


def test_variants_dict_contains_all_keys():
    """All declared variant keys are present in variants."""
    mv = ModelVariant("Transport", _make_variants(), selector="bike")
    assert set(mv.variants.keys()) == {"bike", "train"}


def test_variants_dict_gives_access_to_model_instances():
    """variants["key"] returns the original Model instance."""
    cap_bike = Index("capacity", 100.0)
    bike = _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike))
    train = _TrainModel(inputs=_TrainModel.Inputs(capacity=Index("c", 1.0)))
    mv = ModelVariant("Transport", {"bike": bike, "train": train}, selector="bike")
    assert mv.variants["bike"] is bike


# ===========================================================================
# Interface validation errors
# ===========================================================================


def test_mismatched_outputs_raises_value_error():
    """Variants with different outputs field names raise ValueError."""
    cap = Index("capacity", 100.0)
    with pytest.raises(ValueError, match="outputs.*field names differ"):
        ModelVariant(
            "Transport",
            {
                "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap)),
                "other": _OtherOutputsModel(inputs=_OtherOutputsModel.Inputs(capacity=Index("capacity", 200.0))),
            },
            selector="bike",
        )


def test_mismatched_inputs_is_allowed():
    """Variants with different inputs field names are allowed; inputs union is exposed."""
    cap_bike = Index("capacity", 100.0)
    cap_extra = Index("capacity", 200.0)
    bonus = Index("bonus", 5.0)
    mv = ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike)),
            "extra": _ExtraInputModel(inputs=_ExtraInputModel.Inputs(capacity=cap_extra, bonus=bonus)),
        },
        selector="bike",
    )
    assert mv is not None


def test_runtime_inputs_stay_disjoint_by_variant():
    """Runtime mode inputs stay a disjoint union too — never merged by field name."""
    from civic_digital_twins.dt_model.model.index import CategoricalIndex

    cap_bike = Index("capacity", 100.0)
    cap_extra = Index("capacity", 200.0)
    bonus = Index("bonus", 5.0)
    mode = CategoricalIndex("mode", {"bike": 0.5, "extra": 0.5})
    mv = ModelVariant(
        "Transport",
        {
            "bike": _BikeModel(inputs=_BikeModel.Inputs(capacity=cap_bike)),
            "extra": _ExtraInputModel(inputs=_ExtraInputModel.Inputs(capacity=cap_extra, bonus=bonus)),
        },
        selector=mode,
    )
    assert set(mv.inputs) == {"bike", "extra"}
    assert mv.inputs["bike"].capacity is cap_bike
    assert mv.inputs["extra"].capacity is cap_extra
    assert mv.inputs["extra"].bonus is bonus
    # "bike" never declared "bonus" — not merged in from "extra".
    assert not hasattr(mv.inputs["bike"], "bonus")


def test_empty_variants_raises_value_error():
    """Empty variants dict raises ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        ModelVariant("Transport", {}, selector="bike")


# ===========================================================================
# repr
# ===========================================================================


def test_repr_contains_name_and_active_key():
    """Repr includes the variant group name and active key."""
    mv = ModelVariant("Transport", _make_variants(), selector="train")
    r = repr(mv)
    assert "Transport" in r
    assert "train" in r


# ===========================================================================
# Expose-bearing model — extra coverage
# ===========================================================================


class _ExposeModel(Model, legacy=True):
    """Model that uses an Expose dataclass for intermediate results.

    Takes a ``label`` constructor parameter (used for both the model name and
    the exposed ratio index's name) that does not fit the ``@define`` +
    ``compute()`` contract, which only accepts ``inputs=`` (and ``fns=``).
    """

    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        throughput: Index
        emissions: Index

    @expose
    class Expose:
        ratio: Index

    def __init__(self, capacity: Index, label: str) -> None:
        Inputs = _ExposeModel.Inputs
        Outputs = _ExposeModel.Outputs
        Expose = _ExposeModel.Expose

        throughput = Index("throughput", capacity * 1.0)
        emissions = Index("emissions", 0.0)
        ratio = Index("ratio_" + label, 1.0)

        super().__init__(
            f"ExposeModel-{label}",
            inputs=Inputs(capacity=capacity),
            outputs=Outputs(throughput=throughput, emissions=emissions),
            expose=Expose(ratio=ratio),
        )


def test_expose_proxy_common_field_reads_active_variant_value():
    """expose.<field> common to all variants reads the active variant's value, in static mode."""
    cap_a = Index("capacity", 100.0)
    cap_b = Index("capacity", 200.0)
    variants: dict[str, Model] = {
        "a": _ExposeModel(cap_a, "a"),
        "b": _ExposeModel(cap_b, "b"),
    }
    mv = ModelVariant("ExposeGroup", variants, selector="a")
    # ratio is common to both variants — accessible through mv.expose, value from "a" (active).
    assert mv.expose.ratio.concrete_default == 1.0


class _ExposeModelExtra(Model, legacy=True):
    """Like ``_ExposeModel``, but its ``Expose`` has one additional field."""

    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        throughput: Index
        emissions: Index

    @expose
    class Expose:
        ratio: Index
        extra: Index

    def __init__(self, capacity: Index, label: str) -> None:
        Inputs = _ExposeModelExtra.Inputs
        Outputs = _ExposeModelExtra.Outputs
        Expose = _ExposeModelExtra.Expose

        throughput = Index("throughput", capacity * 1.0)
        emissions = Index("emissions", 0.0)
        ratio = Index("ratio_" + label, 1.0)
        extra = Index("extra_" + label, 2.0)

        super().__init__(
            f"ExposeModelExtra-{label}",
            inputs=Inputs(capacity=capacity),
            outputs=Outputs(throughput=throughput, emissions=emissions),
            expose=Expose(ratio=ratio, extra=extra),
        )


def test_expose_static_mode_narrows_to_intersection_across_variants():
    """mv.expose in static mode only exposes fields common to *all* declared variants.

    A field present only on the active variant (not on an inactive sibling) is not reachable
    through mv.expose — even though static mode otherwise proxies the active variant directly for
    outputs/inputs/indexes. This keeps mv.expose's accessible field set independent of which
    selector value is chosen. mv.variants[key].expose remains available, unrestricted, for a
    specific variant's full expose shape.
    """
    cap_a = Index("capacity", 100.0)
    cap_b = Index("capacity", 200.0)
    variants: dict[str, Model] = {
        "a": _ExposeModelExtra(cap_a, "a"),  # has ratio + extra
        "b": _ExposeModel(cap_b, "b"),  # has ratio only
    }
    mv = ModelVariant("ExposeGroup", variants, selector="a")

    # ratio is common to both — accessible, value from the active variant ("a").
    assert mv.expose.ratio.concrete_default == 1.0

    # extra only exists on "a" (the active variant!) but not on "b" — narrowed out regardless.
    with pytest.raises(AttributeError):
        _ = mv.expose.extra

    # The full, unrestricted shape of a specific variant remains reachable explicitly.
    assert mv.variants["a"].expose.extra.concrete_default == 2.0


def test_expose_indexes_not_in_inactive_variant():
    """Expose indexes of inactive variant are not in mv.indexes."""
    cap_a = Index("capacity", 100.0)
    cap_b = Index("capacity", 200.0)
    model_a = _ExposeModel(cap_a, "a")
    model_b = _ExposeModel(cap_b, "b")
    variants: dict[str, Model] = {"a": model_a, "b": model_b}
    mv = ModelVariant("ExposeGroup", variants, selector="a")

    # ratio of "b" must not appear in mv.indexes (identity check — __eq__ returns a Node).
    ratio_b = model_b.expose.ratio
    mv_index_ids = {id(idx) for idx in mv.indexes}
    assert id(ratio_b) not in mv_index_ids


def test_modelvariant_outputs_and_expose_nestable_as_bulk_field():
    """mv.outputs/mv.expose can be nested as a whole field, same as a plain Model's.

    Regression test: ModelVariant.outputs/.expose build a synthesized IOProxy with no
    backing dataclass instance, which previously had no `dc=`, so contract validation
    (`_check_index_field_value`) rejected nesting them in a parent's own @outputs/@expose
    field ("Surfacing sub-model diagnostics in bulk" — dd-cdt-modularity.md), even though
    the identical pattern already worked for a plain Model's own .outputs/.expose.
    """
    cap_a = Index("capacity", 100.0)
    cap_b = Index("capacity", 200.0)
    variants: dict[str, Model] = {
        "a": _ExposeModel(cap_a, "a"),
        "b": _ExposeModel(cap_b, "b"),
    }
    mv = ModelVariant("ExposeGroup", variants, selector="a")

    @define("RootBulk")
    class _RootBulk(Model):
        @inputs
        class Inputs:
            capacity: Index

        @outputs
        class Outputs:
            total: Index

        @expose
        class Expose:
            sub_out: _ExposeModel.Outputs
            sub_exp: _ExposeModel.Expose

        def compute(self, inputs: "_RootBulk.Inputs") -> tuple["_RootBulk.Outputs", "_RootBulk.Expose"]:
            return (
                _RootBulk.Outputs(total=Index("total", mv.outputs.throughput)),
                _RootBulk.Expose(sub_out=mv.outputs, sub_exp=mv.expose),
            )

    root = _RootBulk(inputs=_RootBulk.Inputs(capacity=cap_a))
    assert root.expose.sub_out.throughput is mv.outputs.throughput
    assert root.expose.sub_exp.ratio is mv.expose.ratio


def test_modelvariant_outputs_cannot_be_nested_in_expose_direction_violation():
    """An @outputs field must not hold a ModelVariant's .expose, same rule as for a plain Model."""
    variants = _make_variants()
    mv = ModelVariant("Transport", variants, selector="bike")

    @define("RootBad")
    class _RootBad(Model):
        @inputs
        class Inputs:
            pass

        @outputs
        class Outputs:
            sub_out: IOProxy  # would hold mv.outputs (an @expose value) — forbidden

        def compute(self, inputs: "_RootBad.Inputs") -> "_RootBad.Outputs":
            return _RootBad.Outputs(sub_out=mv.expose)

    with pytest.raises(TypeError, match="must not hold an .*@expose"):
        _RootBad(inputs=_RootBad.Inputs())


# ===========================================================================
# Export smoke tests
# ===========================================================================


def test_model_variant_importable_from_dt_model():
    """ModelVariant is importable from civic_digital_twins.dt_model."""
    from civic_digital_twins.dt_model import ModelVariant as MV  # noqa: PLC0415

    assert MV is ModelVariant


def test_model_variant_importable_from_model_subpackage():
    """ModelVariant is importable from civic_digital_twins.dt_model.model."""
    from civic_digital_twins.dt_model.model import ModelVariant as MV  # noqa: PLC0415

    assert MV is ModelVariant
