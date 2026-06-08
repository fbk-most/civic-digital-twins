# SPDX-License-Identifier: Apache-2.0
"""Runnable snippets from docs/design/dd-cdt-modularity.md."""

import warnings
from collections.abc import Sequence

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    ConstIndex,
    DistributionEnsemble,
    DistributionIndex,
    Evaluation,
    Functor,
    GenericIndex,
    Index,
    InputsContractWarning,
    Model,
    ModelVariant,
    NumpyBackend,
    Scenario,
    TimeseriesIndex,
    define,
    expose,
    functions,
    graph,
    inputs,
    outputs,
)


def _id_in(idx: GenericIndex, seq: Sequence[GenericIndex]) -> bool:
    """Return ``True`` if *idx* is present in *seq* by identity.

    Notes
    -----
    ``GenericIndex.__eq__`` returns a lazy graph node rather than a ``bool``,
    so the built-in ``in`` operator always evaluates as truthy.  This helper
    uses ``is`` instead, matching the same contract used by
    ``IOProxy.__contains__``.
    """
    return any(idx is item for item in seq)


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md TL;DR — @define with @inputs/@outputs/@expose
# ---------------------------------------------------------------------------


@define("Traffic")
class TrafficModel(Model):

    @inputs
    class Inputs:
        ts_inflow:         TimeseriesIndex
        ts_starting:       TimeseriesIndex
        modified_inflow:   Index
        modified_starting: Index

    @outputs
    class Outputs:
        traffic:                TimeseriesIndex
        modified_traffic:       TimeseriesIndex
        total_modified_traffic: Index
        inflow_ratio:           Index
        starting_ratio:         Index
        traffic_ratio:          Index

    def compute(self, inputs: Inputs) -> Outputs:
        traffic = TimeseriesIndex("reference traffic", inputs.ts_inflow + inputs.ts_starting)
        modified_traffic = TimeseriesIndex("modified traffic", inputs.modified_inflow + inputs.modified_starting)
        total_modified_traffic = Index("total modified traffic", modified_traffic.sum())
        inflow_ratio     = Index("inflow ratio", inputs.ts_inflow / inputs.modified_inflow)
        starting_ratio   = Index("starting ratio", inputs.ts_starting / inputs.modified_starting)
        traffic_ratio    = Index("traffic ratio", traffic / modified_traffic)
        return TrafficModel.Outputs(
            traffic=traffic,
            modified_traffic=modified_traffic,
            total_modified_traffic=total_modified_traffic,
            inflow_ratio=inflow_ratio,
            starting_ratio=starting_ratio,
            traffic_ratio=traffic_ratio,
        )


ts_in = TimeseriesIndex("inflow", np.array([10.0, 20.0, 30.0]))
ts_st = TimeseriesIndex("starting", np.array([5.0, 10.0, 15.0]))
mod_in = Index("modified_inflow", 0.9)
mod_st = Index("modified_starting", 0.95)
m = TrafficModel(inputs=TrafficModel.Inputs(
    ts_inflow=ts_in,
    ts_starting=ts_st,
    modified_inflow=mod_in,
    modified_starting=mod_st,
))

assert m.inputs.ts_inflow is ts_in
assert m.inputs.ts_starting is ts_st
assert m.outputs.traffic is not None
assert m.outputs.modified_traffic is not None
assert m.outputs.total_modified_traffic is not None
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
    mod_i = Index("mod_inflow_demo", 0.9)
    mod_s = Index("mod_starting_demo", 0.95)
    traffic = TrafficModel(inputs=TrafficModel.Inputs(
        ts_inflow=ts_i,
        ts_starting=ts_s,
        modified_inflow=mod_i,
        modified_starting=mod_s,
    ))
    ts = traffic.outputs.traffic           # contractual output — stable
    mod = traffic.outputs.modified_traffic  # contractual output — stable
    inp = traffic.inputs.ts_inflow         # contractual input  — stable
    assert ts is not None
    assert mod is not None
    assert inp is ts_i


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §3 — Three-level access model
# ---------------------------------------------------------------------------


@define("ThreeLevel")
class ThreeLevelModel(Model):

    @inputs
    class Inputs:
        base: Index

    @outputs
    class Outputs:
        result: Index

    @expose
    class Expose:
        intermediate: Index

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        intermediate = Index("intermediate", inputs.base * 2)
        result = Index("result", intermediate + 1)
        return (
            ThreeLevelModel.Outputs(result=result),
            ThreeLevelModel.Expose(intermediate=intermediate),
        )


b = Index("base", 5.0)
m3 = ThreeLevelModel(inputs=ThreeLevelModel.Inputs(base=b))

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


@define("StageA")
class StageAModel(Model):

    @inputs
    class Inputs:
        raw_data: Index

    @outputs
    class Outputs:
        processed: Index
        ratio: Index

    def compute(self, inputs: Inputs) -> Outputs:
        processed = Index("processed", inputs.raw_data * 2.0)
        ratio = Index("ratio", inputs.raw_data * 0.1)
        return StageAModel.Outputs(processed=processed, ratio=ratio)


@define("StageB")
class StageBModel(Model):

    @inputs
    class Inputs:
        processed: Index
        ratio: Index

    @outputs
    class Outputs:
        result: Index

    def compute(self, inputs: Inputs) -> Outputs:
        result = Index("result", inputs.processed + inputs.ratio)
        return StageBModel.Outputs(result=result)


@define("Pipeline")
class PipelineModel(Model):

    @inputs
    class Inputs:
        raw_data: DistributionIndex

    @outputs
    class Outputs:
        result: Index

    def compute(self, inputs: Inputs) -> Outputs:
        stage_a = StageAModel(inputs=StageAModel.Inputs(raw_data=inputs.raw_data))
        stage_b = StageBModel(inputs=StageBModel.Inputs(
            processed=stage_a.outputs.processed,
            ratio=stage_a.outputs.ratio,
        ))
        return PipelineModel.Outputs(result=stage_b.outputs.result)


_raw_data = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
pipeline = PipelineModel(inputs=PipelineModel.Inputs(raw_data=_raw_data))

assert pipeline.outputs.result is not None
# raw_data is a DistributionIndex (abstract) → model is not fully instantiated
assert pipeline.is_instantiated() is False
# The wired output is reachable through the pipeline's index list
assert _id_in(pipeline.outputs.result, pipeline.indexes)


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §5 — Inputs contract convention / InputsContractWarning
# (legacy=True models demonstrating the warning for hand-written __init__)
# ---------------------------------------------------------------------------


class GoodModel(Model, legacy=True):
    """Model that correctly declares its GenericIndex parameter in Inputs."""

    @inputs
    class Inputs:
        inflow: TimeseriesIndex

    @outputs
    class Outputs:
        total: Index

    def __init__(self, inflow: TimeseriesIndex) -> None:
        Inputs = GoodModel.Inputs
        inputs_ = Inputs(inflow=inflow)       # ... and forwarded here
        total_idx = Index("total_good", inputs_.inflow.sum())
        super().__init__("Good", inputs=inputs_, outputs=GoodModel.Outputs(total=total_idx))


class BadModel(Model, legacy=True):
    """Model that deliberately omits 'inflow' from its Inputs to trigger the warning."""

    @inputs
    class Inputs:
        pass   # inflow is missing

    def __init__(self, inflow: TimeseriesIndex) -> None:
        # InputsContractWarning fires here: 'inflow' holds a GenericIndex
        # that is not declared in Inputs.
        total = Index("total_bad", inflow.sum())  # noqa: F841
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

    from civic_digital_twins.dt_model import InputsContractWarning, ModelContractWarning

    with warnings.catch_warnings():
        # Escalate all contract warnings to errors (recommended for CI)
        warnings.filterwarnings("error", category=ModelContractWarning)

        # Or target only the inputs-specific warning
        warnings.filterwarnings("error", category=InputsContractWarning)


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §6 — ModelVariant transport models
# ---------------------------------------------------------------------------


@define("Bike")
class BikeModel(Model):

    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        emissions: Index

    @classmethod
    def default_inputs(cls, capacity: float = 100.0) -> Inputs:
        return cls.Inputs(capacity=ConstIndex("bike_capacity", capacity))

    def compute(self, inputs: Inputs) -> Outputs:
        emissions = Index("bike_emissions", inputs.capacity * 3.0)
        return BikeModel.Outputs(emissions=emissions)


@define("Train")
class TrainModel(Model):

    @inputs
    class Inputs:
        capacity: Index

    @outputs
    class Outputs:
        emissions: Index

    @classmethod
    def default_inputs(cls, capacity: float = 500.0) -> Inputs:
        return cls.Inputs(capacity=ConstIndex("train_capacity", capacity))

    def compute(self, inputs: Inputs) -> Outputs:
        emissions = Index("train_emissions", inputs.capacity * 1.0)
        return TrainModel.Outputs(emissions=emissions)


# ---------------------------------------------------------------------------
# Block 09: dd-cdt-modularity.md — ModelVariant static mode
# ---------------------------------------------------------------------------


def _demo_09_static_variant() -> None:
    """Block 09: ModelVariant with static string selector."""
    mv = ModelVariant(
        "TransportModel",
        variants={
            "bike":  BikeModel(inputs=BikeModel.default_inputs(100)),
            "train": TrainModel(inputs=TrainModel.default_inputs(500)),
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
            "bike":  BikeModel(inputs=BikeModel.default_inputs(100)),
            "train": TrainModel(inputs=TrainModel.default_inputs(500)),
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
            "bike":  BikeModel(inputs=BikeModel.default_inputs()),
            "train": TrainModel(inputs=TrainModel.default_inputs()),
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
    variants={
        "bike":  BikeModel(inputs=BikeModel.default_inputs()),
        "train": TrainModel(inputs=TrainModel.default_inputs()),
    },
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
    """Block 13: ModelVariant with CategoricalIndex selector."""
    mode = CategoricalIndex("mode", {"bike": 0.3, "train": 0.7})

    mv = ModelVariant(
        "TransportModel",
        variants={
            "bike":  BikeModel(inputs=BikeModel.default_inputs()),
            "train": TrainModel(inputs=TrainModel.default_inputs()),
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
    season = CategoricalIndex("season", {"summer": 0.25, "spring": 0.25, "autumn": 0.25, "winter": 0.25})

    peak_factor = Index(
        "peak_factor",
        graph.piecewise(
            (1.8, season == "summer"),
            (1.2, season == "spring"),
            (1.0, season == "autumn"),
            (0.7, True),  # winter — default
        ),
    )

    assert peak_factor.value is not None


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md — Presence-aware variant models for 2-D parameter sweep
# ---------------------------------------------------------------------------


@define("BikePres")
class BikeModelPres(Model):

    @inputs
    class Inputs:
        presence: Index

    @outputs
    class Outputs:
        emissions: Index

    def compute(self, inputs: Inputs) -> Outputs:
        emissions = Index("bike_pres_emissions", inputs.presence * 3.0)
        return BikeModelPres.Outputs(emissions=emissions)


@define("TrainPres")
class TrainModelPres(Model):

    @inputs
    class Inputs:
        presence: Index

    @outputs
    class Outputs:
        emissions: Index

    def compute(self, inputs: Inputs) -> Outputs:
        emissions = Index("train_pres_emissions", inputs.presence * 1.0)
        return TrainModelPres.Outputs(emissions=emissions)


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md — CategoricalIndex as PARAMETER axis (1-D sweep)
# ---------------------------------------------------------------------------


def _demo_catidx_param_axis_1d() -> None:
    """1-D deterministic sweep: mode ∈ {bike, train}, constant capacity variants."""
    mode_param = CategoricalIndex("mode_param", {"bike": 0.5, "train": 0.5})
    mv_param = ModelVariant(
        "TransportParam",
        variants={
            "bike":  BikeModel(inputs=BikeModel.default_inputs()),
            "train": TrainModel(inputs=TrainModel.default_inputs()),
        },
        selector=mode_param,
    )

    result = Evaluation(Scenario(mv_param)).evaluate(
        ensemble=None,
        parameters={mode_param: np.array(["bike", "train"])},
    )
    # result.expected_value(mv_param.outputs.emissions) → shape (2,)
    # index 0 = bike emissions, index 1 = train emissions
    arr = result.expected_value(mv_param.outputs.emissions)
    assert arr.shape == (2,)
    assert np.isclose(arr[0], 100.0 * 3.0)  # BikeModel: capacity=100, factor=3
    assert np.isclose(arr[1], 500.0 * 1.0)  # TrainModel: capacity=500, factor=1


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md — CategoricalIndex as PARAMETER axis (2-D grid)
# ---------------------------------------------------------------------------


def _demo_catidx_param_axis_2d() -> None:
    """2-D deterministic grid: mode × presence, presence-aware variants."""
    mode_param = CategoricalIndex("mode_param", {"bike": 0.5, "train": 0.5})
    presence = Index("presence", None)  # abstract — swept by the grid
    mv_grid = ModelVariant(
        "TransportGrid",
        variants={
            "bike":  BikeModelPres(inputs=BikeModelPres.Inputs(presence=presence)),
            "train": TrainModelPres(inputs=TrainModelPres.Inputs(presence=presence)),
        },
        selector=mode_param,
    )

    result = Evaluation(Scenario(mv_grid)).evaluate(
        ensemble=None,
        parameters={
            mode_param: np.array(["bike", "train"]),
            presence:   np.array([100.0, 200.0, 300.0]),
        },
    )
    # result.expected_value(mv_grid.outputs.emissions) → shape (2, 3)
    # row 0 = bike emissions for each presence level
    # row 1 = train emissions for each presence level
    arr = result.expected_value(mv_grid.outputs.emissions)
    assert arr.shape == (2, 3)
    assert np.allclose(arr[0], [100.0 * 3, 200.0 * 3, 300.0 * 3])  # bike: presence * 3
    assert np.allclose(arr[1], [100.0 * 1, 200.0 * 1, 300.0 * 1])  # train: presence * 1


# ---------------------------------------------------------------------------
# Block 20: dd-cdt-modularity.md — @functions contract
# ---------------------------------------------------------------------------


def _demo_20_functions_contract() -> None:
    """Block 20: @functions typed functor contract."""

    @define("Smoother")
    class SmootherModel(Model):

        @inputs
        class Inputs:
            signal: TimeseriesIndex

        @functions
        class Functions:
            smooth: Functor

        @outputs
        class Outputs:
            smoothed: TimeseriesIndex

        def compute(self, inputs: Inputs, *, fns: Functions) -> Outputs:
            smoothed = TimeseriesIndex("smoothed", graph.function_call("smooth", inputs.signal))
            return SmootherModel.Outputs(smoothed=smoothed)

    signal = TimeseriesIndex("signal", np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
    m = SmootherModel(
        inputs=SmootherModel.Inputs(signal=signal),
        fns=SmootherModel.Functions(smooth=NumpyBackend.adapt(lambda x: x)),
    )
    assert m.outputs.smoothed is not None


# ---------------------------------------------------------------------------
# dd-cdt-modularity.md — End-to-end evaluation with Scenario
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# dd-cdt-modularity.md — function_call in a formula node
# ---------------------------------------------------------------------------

_mv_base = Index("base_fc", 10.0)
_smoothed_node = graph.function_call("smooth", _mv_base)
assert _smoothed_node is not None


# ---------------------------------------------------------------------------
# Block 29: dd-cdt-modularity.md — Warning classes API reference
# ---------------------------------------------------------------------------


def _demo_29_filterwarnings_api() -> None:
    """Block 29: API reference — escalate contract warnings."""
    import warnings

    from civic_digital_twins.dt_model import InputsContractWarning, ModelContractWarning

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
_demo_20_functions_contract()
_demo_catidx_param_axis_1d()
_demo_catidx_param_axis_2d()
_demo_29_filterwarnings_api()


if __name__ == "__main__":
    print("doc_modularity.py: all snippets OK")
