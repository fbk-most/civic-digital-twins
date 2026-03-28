"""Tests for civic_digital_twins.dt_model.model.ModelVariant."""

# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from civic_digital_twins.dt_model.model.index import Index
from civic_digital_twins.dt_model.model.model import Model
from civic_digital_twins.dt_model.model.model_variant import ModelVariant

# ---------------------------------------------------------------------------
# Shared fixtures — two concrete Model subclasses with the same I/O contract
# ---------------------------------------------------------------------------


class _BikeModel(Model):
    """Variant A — bicycle-mode transport model."""

    @dataclasses.dataclass
    class Inputs:
        capacity: Index

    @dataclasses.dataclass
    class Outputs:
        throughput: Index
        emissions: Index

    def __init__(self, capacity: Index) -> None:
        Inputs = _BikeModel.Inputs
        Outputs = _BikeModel.Outputs

        cap_val = capacity.value
        throughput = Index("throughput", float(cap_val) * 1.0 if isinstance(cap_val, (int, float)) else None)
        emissions = Index("emissions", 0.0)

        super().__init__(
            "BikeModel",
            inputs=Inputs(capacity=capacity),
            outputs=Outputs(throughput=throughput, emissions=emissions),
        )


class _TrainModel(Model):
    """Variant B — rail-mode transport model."""

    @dataclasses.dataclass
    class Inputs:
        capacity: Index

    @dataclasses.dataclass
    class Outputs:
        throughput: Index
        emissions: Index

    def __init__(self, capacity: Index) -> None:
        Inputs = _TrainModel.Inputs
        Outputs = _TrainModel.Outputs

        cap_val = capacity.value
        throughput = Index("throughput", float(cap_val) * 10.0 if isinstance(cap_val, (int, float)) else None)
        emissions = Index("emissions", 50.0)

        super().__init__(
            "TrainModel",
            inputs=Inputs(capacity=capacity),
            outputs=Outputs(throughput=throughput, emissions=emissions),
        )


def _make_variants() -> dict[str, Model]:
    """Build a fresh variants dict with independent Index objects."""
    cap_bike = Index("capacity", 100.0)
    cap_train = Index("capacity", 500.0)
    return {
        "bike": _BikeModel(cap_bike),
        "train": _TrainModel(cap_train),
    }


# ---------------------------------------------------------------------------
# Helper: Model with mismatched I/O field names
# ---------------------------------------------------------------------------


class _OtherOutputsModel(Model):
    """Model whose outputs field names differ from _BikeModel / _TrainModel."""

    @dataclasses.dataclass
    class Inputs:
        capacity: Index

    @dataclasses.dataclass
    class Outputs:
        flow: Index  # renamed field
        co2: Index  # renamed field

    def __init__(self, capacity: Index) -> None:
        Inputs = _OtherOutputsModel.Inputs
        Outputs = _OtherOutputsModel.Outputs

        super().__init__(
            "OtherOutputsModel",
            inputs=Inputs(capacity=capacity),
            outputs=Outputs(flow=Index("flow", 1.0), co2=Index("co2", 2.0)),
        )


class _OtherInputsModel(Model):
    """Model whose inputs field names differ from _BikeModel / _TrainModel."""

    @dataclasses.dataclass
    class Inputs:
        volume: Index  # renamed field

    @dataclasses.dataclass
    class Outputs:
        throughput: Index
        emissions: Index

    def __init__(self, volume: Index) -> None:
        Inputs = _OtherInputsModel.Inputs
        Outputs = _OtherInputsModel.Outputs

        super().__init__(
            "OtherInputsModel",
            inputs=Inputs(volume=volume),
            outputs=Outputs(throughput=Index("t", 1.0), emissions=Index("e", 0.0)),
        )


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
    assert mv.outputs.emissions.value == 0.0


def test_static_selector_train_outputs():
    """Static 'train' selector delegates to TrainModel."""
    mv = ModelVariant("Transport", _make_variants(), selector="train")
    # TrainModel sets emissions=50.0
    assert mv.outputs.emissions.value == 50.0


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


def test_inputs_proxy_delegates_to_active_variant():
    """Inputs proxy delegates to the active variant."""
    cap_bike = Index("capacity", 100.0)
    cap_train = Index("capacity", 500.0)
    variants = {"bike": _BikeModel(cap_bike), "train": _TrainModel(cap_train)}
    mv = ModelVariant("Transport", variants, selector="bike")
    # inputs.capacity should be the bike model's capacity index (same object)
    assert mv.inputs.capacity is cap_bike


def test_outputs_proxy_delegates_to_active_variant():
    """Outputs proxy delegates to the active variant."""
    variants = _make_variants()
    mv = ModelVariant("Transport", variants, selector="train")
    # TrainModel throughput = capacity * 10 = 500 * 10 = 5000
    assert mv.outputs.throughput.value == 5000.0


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
    bike = _BikeModel(cap_bike)
    train = _TrainModel(cap_train)
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
    train = _TrainModel(cap_train)
    mv = ModelVariant("Transport", {"bike": _BikeModel(cap_bike), "train": train}, selector="bike")

    # The train capacity is NOT in mv.indexes (identity check — __eq__ returns a Node).
    mv_index_ids = {id(idx) for idx in mv.indexes}
    assert id(cap_train) not in mv_index_ids
    # ... but is accessible via the explicit path.
    assert mv.variants["train"].inputs.capacity is cap_train


def test_abstract_indexes_delegates_to_active_variant():
    """abstract_indexes() delegates to the active variant."""
    cap_placeholder = Index("capacity", None)
    variants = {
        "bike": _BikeModel(cap_placeholder),
        "train": _TrainModel(Index("capacity", 500.0)),
    }
    mv = ModelVariant("Transport", variants, selector="bike")
    abstract = mv.abstract_indexes()
    # cap_placeholder has value=None, so it is abstract.
    assert cap_placeholder in abstract


def test_is_instantiated_delegates_to_active_variant():
    """is_instantiated() delegates to the active variant."""
    cap_concrete = Index("capacity", 100.0)
    variants = {
        "bike": _BikeModel(cap_concrete),
        "train": _TrainModel(Index("capacity", 500.0)),
    }
    mv = ModelVariant("Transport", variants, selector="bike")
    assert mv.is_instantiated()


def test_is_not_instantiated_when_active_has_placeholder():
    """is_instantiated() returns False when the active variant has a placeholder."""
    cap_placeholder = Index("capacity", None)
    variants = {
        "bike": _BikeModel(cap_placeholder),
        "train": _TrainModel(Index("capacity", 500.0)),
    }
    mv = ModelVariant("Transport", variants, selector="bike")
    assert not mv.is_instantiated()


# ===========================================================================
# Direct attribute access (transparency)
# ===========================================================================


def test_direct_attribute_access_forwards_to_active_variant():
    """Attribute access for unknown names is forwarded to the active Model."""
    cap_bike = Index("capacity", 100.0)
    bike = _BikeModel(cap_bike)
    mv = ModelVariant("Transport", {"bike": bike, "train": _TrainModel(Index("capacity", 500.0))}, selector="bike")
    # 'name' is defined directly on ModelVariant, not proxied.
    assert mv.name == "Transport"
    # inputs is a property on ModelVariant — check field-level forwarding via proxy.
    assert mv.inputs.capacity is cap_bike


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
    bike = _BikeModel(cap_bike)
    mv = ModelVariant("Transport", {"bike": bike, "train": _TrainModel(Index("c", 1.0))}, selector="bike")
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
                "bike": _BikeModel(cap),
                "other": _OtherOutputsModel(Index("capacity", 200.0)),
            },
            selector="bike",
        )


def test_mismatched_inputs_raises_value_error():
    """Variants with different inputs field names raise ValueError."""
    cap = Index("capacity", 100.0)
    with pytest.raises(ValueError, match="inputs.*field names differ"):
        ModelVariant(
            "Transport",
            {
                "bike": _BikeModel(cap),
                "other": _OtherInputsModel(Index("volume", 200.0)),
            },
            selector="bike",
        )


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


class _ExposeModel(Model):
    """Model that uses an Expose dataclass for intermediate results."""

    @dataclasses.dataclass
    class Inputs:
        capacity: Index

    @dataclasses.dataclass
    class Outputs:
        throughput: Index
        emissions: Index

    @dataclasses.dataclass
    class Expose:
        ratio: Index

    def __init__(self, capacity: Index, label: str) -> None:
        Inputs = _ExposeModel.Inputs
        Outputs = _ExposeModel.Outputs
        Expose = _ExposeModel.Expose

        cap_val = capacity.value
        throughput = Index("throughput", float(cap_val) if isinstance(cap_val, (int, float)) else None)
        emissions = Index("emissions", 0.0)
        ratio = Index("ratio_" + label, 1.0)

        super().__init__(
            f"ExposeModel-{label}",
            inputs=Inputs(capacity=capacity),
            outputs=Outputs(throughput=throughput, emissions=emissions),
            expose=Expose(ratio=ratio),
        )


def test_expose_proxy_field_accessible_on_active_variant():
    """expose.<field> on active variant is accessible through ModelVariant."""
    cap_a = Index("capacity", 100.0)
    cap_b = Index("capacity", 200.0)
    variants: dict[str, Model] = {
        "a": _ExposeModel(cap_a, "a"),
        "b": _ExposeModel(cap_b, "b"),
    }
    mv = ModelVariant("ExposeGroup", variants, selector="a")
    # The expose.ratio of variant "a" should be accessible.
    assert mv.expose.ratio.value == 1.0


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
