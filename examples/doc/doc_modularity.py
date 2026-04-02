"""Runnable snippets from docs/design/dd-cdt-modularity.md."""

# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    DistributionIndex,
    Index,
    InputsContractWarning,
    Model,
    ModelVariant,
    TimeseriesIndex,
)
from civic_digital_twins.dt_model.model.index import GenericIndex


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
        total: Index

    def __init__(self, ts_inflow: TimeseriesIndex, ts_starting: TimeseriesIndex) -> None:
        Inputs = TrafficModel.Inputs
        Outputs = TrafficModel.Outputs

        inputs = Inputs(ts_inflow=ts_inflow, ts_starting=ts_starting)

        traffic = TimeseriesIndex("traffic", inputs.ts_inflow + inputs.ts_starting)
        total = Index("total", traffic.sum())

        super().__init__(
            "Traffic",
            inputs=inputs,
            outputs=Outputs(traffic=traffic, total=total),
        )


ts_in = TimeseriesIndex("inflow", np.array([10.0, 20.0, 30.0]))
ts_st = TimeseriesIndex("starting", np.array([5.0, 10.0, 15.0]))
m = TrafficModel(ts_in, ts_st)

assert m.inputs.ts_inflow is ts_in
assert m.inputs.ts_starting is ts_st
assert m.outputs.traffic is not None
assert m.outputs.total is not None
# indexes are derived automatically from inputs + outputs
assert _id_in(ts_in, m.indexes)
assert _id_in(ts_st, m.indexes)

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

        stage_a_indexes: list
        stage_b_indexes: list

    def __init__(self) -> None:
        raw_data = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})

        stage_a = StageAModel(raw_data=raw_data)
        stage_b = StageBModel(
            processed=stage_a.outputs.processed,
            ratio=stage_a.outputs.ratio,
        )

        super().__init__(
            "Pipeline",
            outputs=PipelineModel.Outputs(result=stage_b.outputs.result),
            expose=PipelineModel.Expose(
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

    @dataclass
    class Outputs:
        """Outputs of :class:`BadModel`."""

        total: Index

    def __init__(self, inflow: TimeseriesIndex) -> None:
        # InputsContractWarning fires here: 'inflow' holds a GenericIndex
        # that is not declared in Inputs.
        total = Index("total_bad", inflow.sum())
        super().__init__("Bad", inputs=BadModel.Inputs(), outputs=BadModel.Outputs(total=total))


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
# dd-cdt-modularity.md §6 — ModelVariant
# ---------------------------------------------------------------------------


class BikeModel(Model):
    """Variant: bike transport emissions model."""

    @dataclass
    class Inputs:
        """Inputs of :class:`BikeModel`."""

        base: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`BikeModel`."""

        emissions: Index

    def __init__(self, base: Index) -> None:
        inputs = BikeModel.Inputs(base=base)
        emissions = Index("bike_emissions", inputs.base * 3.0)
        super().__init__("Bike", inputs=inputs, outputs=BikeModel.Outputs(emissions=emissions))


class TrainModel(Model):
    """Variant: train transport emissions model."""

    @dataclass
    class Inputs:
        """Inputs of :class:`TrainModel`."""

        base: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`TrainModel`."""

        emissions: Index

    def __init__(self, base: Index) -> None:
        inputs = TrainModel.Inputs(base=base)
        emissions = Index("train_emissions", inputs.base * 1.0)
        super().__init__("Train", inputs=inputs, outputs=TrainModel.Outputs(emissions=emissions))


mv_base = Index("base", 10.0)
mv = ModelVariant(
    "TransportModel",
    variants={"bike": BikeModel(mv_base), "train": TrainModel(mv_base)},
    selector="bike",
)

# Active variant is "bike" — proxy delegates to BikeModel
assert mv.outputs.emissions is not None
# Inactive variant is still reachable via mv.variants
assert mv.variants["train"] is not None
assert mv.variants["train"].outputs.emissions is not None
# Inactive variant's emissions index must NOT be in mv.indexes (identity check)
assert not _id_in(mv.variants["train"].outputs.emissions, mv.indexes)
# Active variant's emissions index IS in mv.indexes (identity check)
assert _id_in(mv.variants["bike"].outputs.emissions, mv.indexes)
# is_instantiated delegates to active variant (all indexes are concrete)
assert mv.is_instantiated() is True


if __name__ == "__main__":
    print("doc_modularity.py: all snippets OK")
