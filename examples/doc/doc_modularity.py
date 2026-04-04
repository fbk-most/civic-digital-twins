"""Runnable snippets from docs/design/dd-cdt-modularity.md."""

# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    DistributionIndex,
    Index,
    InputsContractWarning,
    Model,
    ModelContractWarning,
    ModelVariant,
    TimeseriesIndex,
)
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.model.index import ConstIndex, GenericIndex


def _id_in(idx: GenericIndex, seq: Sequence[GenericIndex]) -> bool:
    """Return ``True`` if *idx* is present in *seq* by identity.

    Parameters
    ----------
    idx:
        The index to search for.
    seq:
        The sequence to search in.

    Notes
    -----
    ``GenericIndex.__eq__`` returns a lazy graph node rather than a ``bool``,
    so the built-in ``in`` operator always evaluates as truthy.  This helper
    uses ``is`` instead, matching the same contract used by
    ``IOProxy.__contains__``.
    """
    return any(idx is item for item in seq)


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §2 — Basic submodel pattern (Inputs + Outputs)
# ---------------------------------------------------------------------------


class TrafficModel(Model):
    """Minimal model demonstrating the Inputs / Outputs dataclass pattern."""

    @dataclass
    class Inputs:
        """Inputs of :class:`TrafficModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex

    @dataclass
    class Outputs:
        """Outputs of :class:`TrafficModel`."""

        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        total: Index

    def __init__(self, ts_inflow: TimeseriesIndex, ts_starting: TimeseriesIndex) -> None:
        Inputs = TrafficModel.Inputs
        Outputs = TrafficModel.Outputs

        inputs = Inputs(ts_inflow=ts_inflow, ts_starting=ts_starting)

        traffic = TimeseriesIndex("traffic", inputs.ts_inflow + inputs.ts_starting)
        modified_traffic = TimeseriesIndex("modified_traffic", inputs.ts_inflow * 0.9 + inputs.ts_starting)
        total = Index("total", traffic.sum())

        super().__init__(
            "Traffic",
            inputs=inputs,
            outputs=Outputs(traffic=traffic, modified_traffic=modified_traffic, total=total),
        )


ts_in = TimeseriesIndex("inflow", np.array([10.0, 20.0, 30.0]))
ts_st = TimeseriesIndex("starting", np.array([5.0, 10.0, 15.0]))
m = TrafficModel(ts_in, ts_st)

assert m.inputs.ts_inflow is ts_in
assert m.inputs.ts_starting is ts_st
assert m.outputs.traffic is not None
assert m.outputs.modified_traffic is not None
assert m.outputs.total is not None
# indexes are derived automatically from inputs + outputs
assert _id_in(ts_in, m.indexes)
assert _id_in(ts_st, m.indexes)


# ---------------------------------------------------------------------------
# Block 02: dd-cdt-modularity.md — Level 1 contractual access
# ---------------------------------------------------------------------------


def _demo_02_level1_access() -> None:
    """Block 02: Level 1 contractual attribute access."""
    ts_i = TimeseriesIndex("ts_inflow_demo", np.array([10.0, 20.0, 30.0]))
    ts_s = TimeseriesIndex("ts_starting_demo", np.array([5.0, 10.0, 15.0]))
    traffic = TrafficModel(ts_inflow=ts_i, ts_starting=ts_s)
    ts = traffic.outputs.traffic  # contractual output — stable
    mod = traffic.outputs.modified_traffic  # contractual output — stable
    inp = traffic.inputs.ts_inflow  # contractual input  — stable
    assert ts is not None
    assert mod is not None
    assert inp is ts_i


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §3 — Three-level access model
# ---------------------------------------------------------------------------


class ThreeLevelModel(Model):
    """Demonstrates all three access levels: Inputs, Outputs, and Expose."""

    @dataclass
    class Inputs:
        """Inputs of :class:`ThreeLevelModel`."""

        base: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`ThreeLevelModel`."""

        result: Index

    @dataclass
    class Expose:
        """Inspectable intermediate indexes of :class:`ThreeLevelModel`."""

        intermediate: Index

    def __init__(self, base: Index) -> None:
        Inputs = ThreeLevelModel.Inputs
        Outputs = ThreeLevelModel.Outputs
        Expose = ThreeLevelModel.Expose

        inputs = Inputs(base=base)
        intermediate = Index("intermediate", inputs.base * 2)
        result = Index("result", intermediate + 1)

        super().__init__(
            "ThreeLevel",
            inputs=inputs,
            outputs=Outputs(result=result),
            expose=Expose(intermediate=intermediate),
        )


b = Index("base", 5.0)
m3 = ThreeLevelModel(b)

assert m3.inputs.base is b
assert m3.outputs.result is not None
assert m3.expose.intermediate is not None
# all three levels contribute to the flat indexes list
assert _id_in(m3.inputs.base, m3.indexes)
assert _id_in(m3.outputs.result, m3.indexes)
assert _id_in(m3.expose.intermediate, m3.indexes)

# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §4 — Wiring sub-models via constructor (pipeline)
# ---------------------------------------------------------------------------


class StageAModel(Model):
    """Stage-A sub-model: processes raw data into two outputs."""

    @dataclass
    class Inputs:
        """Inputs of :class:`StageAModel`."""

        raw_data: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`StageAModel`."""

        processed: Index
        ratio: Index

    def __init__(self, raw_data: Index) -> None:
        inputs = StageAModel.Inputs(raw_data=raw_data)
        processed = Index("processed", inputs.raw_data * 2.0)
        ratio = Index("ratio", inputs.raw_data * 0.1)
        super().__init__("StageA", inputs=inputs, outputs=StageAModel.Outputs(processed=processed, ratio=ratio))


class StageBModel(Model):
    """Stage-B sub-model: combines processed data and ratio."""

    @dataclass
    class Inputs:
        """Inputs of :class:`StageBModel`."""

        processed: Index
        ratio: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`StageBModel`."""

        result: Index

    def __init__(self, processed: Index, ratio: Index) -> None:
        inputs = StageBModel.Inputs(processed=processed, ratio=ratio)
        result = Index("result", inputs.processed + inputs.ratio)
        super().__init__("StageB", inputs=inputs, outputs=StageBModel.Outputs(result=result))


class PipelineModel(Model):
    """Root model that wires StageAModel → StageBModel as a pipeline."""

    @dataclass
    class Outputs:
        """Outputs of :class:`PipelineModel`."""

        result: Index

    @dataclass
    class Expose:
        """Inspectable intermediate indexes of :class:`PipelineModel`."""

        stage_a_indexes: list[GenericIndex]
        stage_b_indexes: list[GenericIndex]

    def __init__(self) -> None:
        raw_data = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})

        Outputs = PipelineModel.Outputs
        Expose = PipelineModel.Expose

        stage_a = StageAModel(raw_data=raw_data)
        stage_b = StageBModel(
            processed=stage_a.outputs.processed,
            ratio=stage_a.outputs.ratio,
        )

        super().__init__(
            "Pipeline",
            outputs=Outputs(result=stage_b.outputs.result),
            expose=Expose(
                stage_a_indexes=list(stage_a.indexes),
                stage_b_indexes=list(stage_b.indexes),
            ),
        )


pipeline = PipelineModel()

assert pipeline.outputs.result is not None
# raw_data is a DistributionIndex (abstract) → model is not fully instantiated
assert pipeline.is_instantiated() is False
# The wired output is reachable through the pipeline's index list
assert _id_in(pipeline.outputs.result, pipeline.indexes)
# Sub-model indexes are surfaced via Expose
assert len(pipeline.expose.stage_a_indexes) > 0
assert len(pipeline.expose.stage_b_indexes) > 0

# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §5 — Inputs contract convention / InputsContractWarning
# ---------------------------------------------------------------------------


class GoodModel(Model):
    """Model that correctly declares its GenericIndex parameter in Inputs."""

    @dataclass
    class Inputs:
        """Inputs of :class:`GoodModel` — 'inflow' declared here ..."""

        inflow: TimeseriesIndex

    @dataclass
    class Outputs:
        """Outputs of :class:`GoodModel`."""

        total: Index

    def __init__(self, inflow: TimeseriesIndex) -> None:
        Inputs = GoodModel.Inputs
        inputs = Inputs(inflow=inflow)  # ... and forwarded here
        total = Index("total_good", inputs.inflow.sum())
        super().__init__("Good", inputs=inputs, outputs=GoodModel.Outputs(total=total))


class BadModel(Model):
    """Model that deliberately omits 'inflow' from its Inputs dataclass."""

    @dataclass
    class Inputs:
        """Inputs of :class:`BadModel` — intentionally empty to trigger the warning."""

        pass  # inflow is missing

    def __init__(self, inflow: TimeseriesIndex) -> None:
        # InputsContractWarning fires here: 'inflow' holds a GenericIndex
        # that is not declared in Inputs.
        total = Index("total_bad", inflow.sum())
        super().__init__("Bad", inputs=BadModel.Inputs())


ts_inflow_gs = TimeseriesIndex("inflow_gs", np.array([10.0, 20.0, 30.0]))

good = GoodModel(ts_inflow_gs)
assert good.inputs.inflow is ts_inflow_gs
assert good.outputs.total is not None

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    BadModel(ts_inflow_gs)

assert any(issubclass(w.category, InputsContractWarning) for w in caught), (
    "Expected an InputsContractWarning when a GenericIndex parameter is absent from Inputs"
)


# ---------------------------------------------------------------------------
# Block 08: dd-cdt-modularity.md — InputsContractWarning filterwarnings
# ---------------------------------------------------------------------------


def _demo_08_filterwarnings() -> None:
    """Block 08: Escalate contract warnings to errors."""
    import warnings
    from civic_digital_twins.dt_model import ModelContractWarning, InputsContractWarning

    with warnings.catch_warnings():
        # Escalate all contract warnings to errors (recommended for CI)
        warnings.filterwarnings("error", category=ModelContractWarning)

        # Or target only the inputs-specific warning
        warnings.filterwarnings("error", category=InputsContractWarning)


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §6 — ModelVariant transport models
# ---------------------------------------------------------------------------


class BikeModel(Model):
    """Variant: bike transport emissions model."""

    @dataclass
    class Inputs:
        """Inputs of :class:`BikeModel`."""

        capacity: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`BikeModel`."""

        emissions: Index

    def __init__(self, capacity: float = 100.0) -> None:
        cap = ConstIndex("bike_capacity", float(capacity))
        inputs = BikeModel.Inputs(capacity=cap)
        emissions = Index("bike_emissions", cap * 3.0)
        super().__init__("Bike", inputs=inputs, outputs=BikeModel.Outputs(emissions=emissions))


class TrainModel(Model):
    """Variant: train transport emissions model."""

    @dataclass
    class Inputs:
        """Inputs of :class:`TrainModel`."""

        capacity: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`TrainModel`."""

        emissions: Index

    def __init__(self, capacity: float = 500.0) -> None:
        cap = ConstIndex("train_capacity", float(capacity))
        inputs = TrainModel.Inputs(capacity=cap)
        emissions = Index("train_emissions", cap * 1.0)
        super().__init__("Train", inputs=inputs, outputs=TrainModel.Outputs(emissions=emissions))


# ---------------------------------------------------------------------------
# Block 09: dd-cdt-modularity.md — ModelVariant static mode
# ---------------------------------------------------------------------------


def _demo_09_static_variant() -> None:
    """Block 09: ModelVariant with static string selector and capacity kwargs."""
    from civic_digital_twins.dt_model import ModelVariant

    mv = ModelVariant(
        "TransportModel",
        variants={
            "bike": BikeModel(capacity=100),
            "train": TrainModel(capacity=500),
        },
        selector="bike",
    )
    assert mv.outputs.emissions is not None
    assert mv.is_instantiated() is True


# ---------------------------------------------------------------------------
# Block 10: dd-cdt-modularity.md — Transparent proxy attributes
# ---------------------------------------------------------------------------


def _demo_10_proxy_attributes() -> None:
    """Block 10: ModelVariant proxy attribute delegation."""
    mv = ModelVariant(
        "TransportModel",
        variants={
            "bike": BikeModel(capacity=100),
            "train": TrainModel(capacity=500),
        },
        selector="bike",
    )
    mv.outputs.emissions        # delegates to BikeModel.outputs.emissions
    mv.inputs.capacity          # delegates to BikeModel.inputs.capacity
    mv.indexes                  # index list of the active (BikeModel) variant only
    mv.abstract_indexes()       # delegates to BikeModel.abstract_indexes()
    mv.is_instantiated()        # delegates to BikeModel.is_instantiated()


# ---------------------------------------------------------------------------
# Block 11: dd-cdt-modularity.md — Accessing inactive variants
# ---------------------------------------------------------------------------


def _demo_11_inactive_variants() -> None:
    """Block 11: Accessing inactive variants via mv.variants."""
    mv = ModelVariant(
        "TransportModel",
        variants={
            "bike": BikeModel(),
            "train": TrainModel(),
        },
        selector="bike",
    )
    mv.variants["train"].outputs.emissions   # explicit — reaches inactive variant
    mv.variants["train"].indexes             # index list of TrainModel only

    # Active variant's emissions IS in mv.indexes; inactive's is NOT
    assert _id_in(mv.variants["bike"].outputs.emissions, mv.indexes)
    assert not _id_in(mv.variants["train"].outputs.emissions, mv.indexes)


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md — Runtime ModelVariant with CategoricalIndex selector
# ---------------------------------------------------------------------------

mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})

mv_runtime = ModelVariant(
    "TransportModel",
    variants={"bike": BikeModel(), "train": TrainModel()},
    selector=mode,  # runtime: resolved per scenario via DistributionEnsemble
)

# In runtime mode the selector index is abstract — must appear in abstract_indexes()
assert mode in mv_runtime.abstract_indexes()
# The merged output node is a real Index backed by a combined graph node
assert mv_runtime.outputs.emissions is not None
# is_instantiated() always returns False in runtime mode (selector is abstract)
assert mv_runtime.is_instantiated() is False


# ---------------------------------------------------------------------------
# Block 13: dd-cdt-modularity.md — CategoricalIndex selector (no-arg variants)
# ---------------------------------------------------------------------------


def _demo_13_categorical_selector() -> None:
    """Block 13: ModelVariant with CategoricalIndex selector and no-arg variant construction."""
    from civic_digital_twins.dt_model import CategoricalIndex, ModelVariant

    mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})

    mv = ModelVariant(
        "TransportModel",
        variants={
            "bike":  BikeModel(),
            "train": TrainModel(),
        },
        selector=mode,
    )
    assert mode in mv.abstract_indexes()
    assert mv.outputs.emissions is not None
    assert mv.is_instantiated() is False


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md — piecewise formula with mode CategoricalIndex
# ---------------------------------------------------------------------------

peak_factor = Index(
    "peak_factor",
    graph.piecewise((1.8, mode == "bike"), (1.0, True)),  # default: non-bike
)

assert peak_factor.value is not None


# ---------------------------------------------------------------------------
# Block 15: dd-cdt-modularity.md — CategoricalIndex as a formula guard
# ---------------------------------------------------------------------------


def _demo_15_piecewise_categorical() -> None:
    """Block 15: CategoricalIndex season guard with four-clause graph.piecewise."""
    from civic_digital_twins.dt_model import CategoricalIndex
    from civic_digital_twins.dt_model.engine.frontend import graph

    season = CategoricalIndex("season", {"summer": 0.25, "spring": 0.25,
                                          "autumn": 0.25, "winter": 0.25})

    peak_factor = Index("peak_factor", graph.piecewise(
        (1.8, season == "summer"),
        (1.2, season == "spring"),
        (1.0, season == "autumn"),
        (0.7, True),              # winter — default
    ))

    assert peak_factor.value is not None


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md — End-to-End evaluation with marginalize
# ---------------------------------------------------------------------------

from civic_digital_twins.dt_model import DistributionEnsemble, Evaluation  # noqa: E402

# PipelineModel has a DistributionIndex (raw_data), so DistributionEnsemble works.
_pipeline_eval = PipelineModel()
_ensemble_eval = DistributionEnsemble(_pipeline_eval, size=50)
_result_eval = Evaluation(_pipeline_eval).evaluate(_ensemble_eval)

mean_pipeline_result = _result_eval.marginalize(_pipeline_eval.outputs.result)
assert mean_pipeline_result > 0


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md — function_call in a formula node
# ---------------------------------------------------------------------------

_mv_base = Index("base_fc", 10.0)
_smoothed_node = graph.function_call("smooth", _mv_base.node)
assert _smoothed_node is not None


# ---------------------------------------------------------------------------
# Block 29: dd-cdt-modularity.md — Warning classes API reference
# ---------------------------------------------------------------------------


def _demo_29_filterwarnings_api() -> None:
    """Block 29: API reference — escalate contract warnings."""
    import warnings
    from civic_digital_twins.dt_model import ModelContractWarning, InputsContractWarning

    with warnings.catch_warnings():
        # Recommended for CI — escalate all contract warnings to errors
        warnings.filterwarnings("error", category=ModelContractWarning)

        # Fine-grained — only escalate the inputs-specific warning
        warnings.filterwarnings("error", category=InputsContractWarning)


# ---------------------------------------------------------------------------
# Run all demo functions
# ---------------------------------------------------------------------------

_demo_02_level1_access()
_demo_08_filterwarnings()
_demo_09_static_variant()
_demo_10_proxy_attributes()
_demo_11_inactive_variants()
_demo_13_categorical_selector()
_demo_15_piecewise_categorical()
_demo_29_filterwarnings_api()


if __name__ == "__main__":
    print("doc_modularity.py: all snippets OK")
