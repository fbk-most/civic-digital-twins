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


class FactorModel(Model):
    """Stage-A sub-model: doubles its input."""

    @dataclass
    class Inputs:
        """Inputs of :class:`FactorModel`."""

        x: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`FactorModel`."""

        y: Index

    def __init__(self, x: Index) -> None:
        inputs = FactorModel.Inputs(x=x)
        y = Index("y", inputs.x * 2.0)
        super().__init__("Factor", inputs=inputs, outputs=FactorModel.Outputs(y=y))


class SumModel(Model):
    """Stage-B sub-model: sums two inputs."""

    @dataclass
    class Inputs:
        """Inputs of :class:`SumModel`."""

        y: Index
        z: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`SumModel`."""

        total: Index

    def __init__(self, y: Index, z: Index) -> None:
        inputs = SumModel.Inputs(y=y, z=z)
        total = Index("total", inputs.y + inputs.z)
        super().__init__("Sum", inputs=inputs, outputs=SumModel.Outputs(total=total))


class RootModel(Model):
    """Root model that wires FactorModel → SumModel as a pipeline."""

    @dataclass
    class Outputs:
        """Outputs of :class:`RootModel`."""

        total: Index

    @dataclass
    class Expose:
        """Inspectable intermediate indexes of :class:`RootModel`."""

        sub_indexes: list

    def __init__(self) -> None:
        x = DistributionIndex("x", stats.uniform, {"loc": 0.0, "scale": 10.0})
        z = Index("z", 3.0)

        _factor = FactorModel(x=x)
        _sum = SumModel(y=_factor.outputs.y, z=z)

        super().__init__(
            "Root",
            outputs=RootModel.Outputs(total=_sum.outputs.total),
            expose=RootModel.Expose(sub_indexes=list(_factor.indexes) + list(_sum.indexes)),
        )


root = RootModel()

assert root.outputs.total is not None
# x is a DistributionIndex (abstract) → model is not fully instantiated
assert root.is_instantiated() is False
# The wired output is reachable through the root's index list
assert _id_in(root.outputs.total, root.indexes)
# Sub-model indexes are surfaced via Expose
assert len(root.expose.sub_indexes) > 0

# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §5 — Inputs contract convention / InputsContractWarning
# ---------------------------------------------------------------------------


class UndeclaredModel(Model):
    """Model that deliberately omits 'x' from its Inputs dataclass."""

    @dataclass
    class Inputs:
        """Inputs of :class:`UndeclaredModel` — intentionally empty to trigger the warning."""

        pass  # does NOT declare 'x'

    @dataclass
    class Outputs:
        """Outputs of :class:`UndeclaredModel`."""

        y: Index

    def __init__(self, x: Index) -> None:
        y = Index("y", x * 2)
        super().__init__("Undeclared", inputs=UndeclaredModel.Inputs(), outputs=UndeclaredModel.Outputs(y=y))


x_contract = Index("x_contract", 5.0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    UndeclaredModel(x_contract)

assert any(issubclass(w.category, InputsContractWarning) for w in caught), (
    "Expected an InputsContractWarning when a GenericIndex parameter is absent from Inputs"
)

# ---------------------------------------------------------------------------
# dd-cdt-modularity.md §6 — ModelVariant
# ---------------------------------------------------------------------------


class HighCostModel(Model):
    """Variant with a cost multiplier of 3."""

    @dataclass
    class Inputs:
        """Inputs of :class:`HighCostModel`."""

        base: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`HighCostModel`."""

        cost: Index

    def __init__(self, base: Index) -> None:
        inputs = HighCostModel.Inputs(base=base)
        cost = Index("cost", inputs.base * 3.0)
        super().__init__("HighCost", inputs=inputs, outputs=HighCostModel.Outputs(cost=cost))


class LowCostModel(Model):
    """Variant with a cost multiplier of 1."""

    @dataclass
    class Inputs:
        """Inputs of :class:`LowCostModel`."""

        base: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`LowCostModel`."""

        cost: Index

    def __init__(self, base: Index) -> None:
        inputs = LowCostModel.Inputs(base=base)
        cost = Index("cost", inputs.base * 1.0)
        super().__init__("LowCost", inputs=inputs, outputs=LowCostModel.Outputs(cost=cost))


mv_base = Index("base", 10.0)
mv = ModelVariant(
    "CostModel",
    variants={"high": HighCostModel(mv_base), "low": LowCostModel(mv_base)},
    selector="high",
)

# Active variant is "high" — proxy delegates to HighCostModel
assert mv.outputs.cost is not None
# Inactive variant is still reachable via mv.variants
assert mv.variants["low"] is not None
assert mv.variants["low"].outputs.cost is not None
# Inactive variant's cost index must NOT be in mv.indexes (identity check)
assert not _id_in(mv.variants["low"].outputs.cost, mv.indexes)
# Active variant's cost index IS in mv.indexes (identity check)
assert _id_in(mv.variants["high"].outputs.cost, mv.indexes)
# is_instantiated delegates to active variant (all indexes are concrete)
assert mv.is_instantiated() is True


if __name__ == "__main__":
    print("doc_modularity.py: all snippets OK")
