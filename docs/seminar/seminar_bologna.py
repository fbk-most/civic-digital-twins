"""Companion script for the CDT seminar — Bologna mobility case study.

Sections mirror the slide deck (docs/seminar/seminar.md):

  §1  A first end-to-end run            (CDT basics)
  §2  IO Contracts                       (Inputs / Outputs / Expose)
  §3  Modularity                         (constructor wiring, pipeline)
  §4a ModelVariant — model formulation   (SimpleTrafficModel vs IterativeTrafficModel)
  §4b ModelVariant — engine variant      (IterativeTrafficModel vs ExternalSimulatorModel)

Run from the repo root:

    uv run python docs/seminar/seminar_bologna.py

All plots are saved to docs/seminar/ as PNG files.
"""

# SPDX-License-Identifier: Apache-2.0

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Make sure the examples/ packages are importable when the script is run
# from the repo root (the same way pytest does via its pythonpath setting).
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).parent.parent.parent
_examples_dir = _repo_root / "examples"
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from mobility_bologna.mobility_bologna import (
    BolognaModel,
    EmissionsModel,
    InflowModel,
    TrafficModel,
    _ts_solve,
    compute_kpis,
    evaluate,
)
from mobility_bologna.mobility_bologna_data import vehicle_inflow, vehicle_starting

from civic_digital_twins.dt_model import (
    DistributionEnsemble,
    DistributionIndex,
    Index,
    Model,
    ModelVariant,
    TimeseriesIndex,
)
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation

_SEMINAR_DIR = Path(__file__).parent


def _section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ===========================================================================
# §1  A FIRST END-TO-END RUN
# ===========================================================================
_section("§1  A first end-to-end run")

print("\nBuilding BolognaModel …")
model = BolognaModel()

print(f"  Abstract indexes  : {len(model.abstract_indexes())}")
print(f"  Is instantiated?  : {model.is_instantiated()}")

print("\nRunning ensemble (size=200) …")
evals = evaluate(model, size=200)

kpis = compute_kpis(model, evals)
print("\nKPIs:")
for label, value in kpis.items():
    print(f"  {label:<40s} {value:>12,}")

# Quick bar-chart of KPI deltas
fig, ax = plt.subplots(figsize=(8, 3.5))
labels = ["Base\ninflow", "Modified\ninflow", "Shifted\nvehicles", "Paying\nvehicles"]
values = [
    kpis["Base inflow [veh/day]"],
    kpis["Modified inflow [veh/day]"],
    kpis["Shifted inflow [veh/day]"],
    kpis["Paying inflow [veh/day]"],
]
colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
ax.bar(labels, values, color=colors)
ax.set_ylabel("vehicles / day")
ax.set_title("Bologna ZTL pricing — inflow KPIs (200-sample ensemble mean)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
fig.tight_layout()
out = _SEMINAR_DIR / "kpi_inflow.png"
fig.savefig(out, dpi=150)
print(f"\nPlot saved → {out}")
plt.close(fig)

# ===========================================================================
# §2  IO CONTRACTS
# ===========================================================================
_section("§2  IO Contracts — Inputs / Outputs / Expose")

# ── 2a. Inspecting the contract of InflowModel ──────────────────────────────
print("\nInflowModel.Inputs fields:")
for f in InflowModel.Inputs.__dataclass_fields__:
    print(f"  inputs.{f}")

print("\nInflowModel.Outputs fields:")
for f in InflowModel.Outputs.__dataclass_fields__:
    print(f"  outputs.{f}")

print("\nInflowModel.Expose fields (diagnostics — not contractual):")
for f in InflowModel.Expose.__dataclass_fields__:
    print(f"  expose.{f}")

# ── 2b. Accessing a sub-model's outputs by name ──────────────────────────────
# Build a minimal InflowModel to show attribute access
import pandas as pd
from mobility_bologna.mobility_bologna_data import euro_class_split

ts = TimeseriesIndex(
    "time range",
    np.array(
        [
            (t - pd.Timestamp("00:00:00")).total_seconds()
            for t in pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")
        ]
    ),
)
ts_inflow = TimeseriesIndex("inflow", vehicle_inflow)
ts_starting = TimeseriesIndex("starting", vehicle_starting)

inflow_model = InflowModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    ts=ts,
    i_p_start_time=Index("start time", (pd.Timestamp("07:30:00") - pd.Timestamp("00:00:00")).total_seconds()),
    i_p_end_time=Index("end time", (pd.Timestamp("19:30:00") - pd.Timestamp("00:00:00")).total_seconds()),
    i_p_cost=[Index(f"cost euro {e}", 5.00 - e * 0.25) for e in range(7)],
    i_p_fraction_exempted=Index("exempted %", 0.15),
    i_b_p50_cost=DistributionIndex("cost threshold", stats.uniform, {"loc": 4.00, "scale": 7.00}),
    i_b_p50_anticipating=Index("anticipation likelihood", 0.5),
    i_b_p50_anticipation=Index("anticipation threshold", 0.25),
    i_b_p50_postponing=Index("postponement likelihood", 0.8),
    i_b_p50_postponement=Index("postponement threshold", 0.50),
    i_b_starting_modified_factor=Index("starting factor", 1.00),
)

print(f"\nAccessing outputs by name:")
print(f"  inflow_model.outputs.modified_inflow  → {inflow_model.outputs.modified_inflow.name!r}")
print(f"  inflow_model.outputs.total_paying     → {inflow_model.outputs.total_paying.name!r}")
print(f"  inflow_model.expose.i_fraction_anticipating → {inflow_model.expose.i_fraction_anticipating.name!r}")

# ── 2c. InputsContractWarning demo ───────────────────────────────────────────
print("\nInputsContractWarning demo:")

from civic_digital_twins.dt_model import InputsContractWarning


class _BadModel(Model):
    """Deliberately omits 'inflow' from Inputs — triggers the warning."""

    @dataclass
    class Inputs:
        pass  # inflow is missing

    @dataclass
    class Outputs:
        total: Index

    def __init__(self, inflow: TimeseriesIndex) -> None:
        total = Index("total", inflow.sum())
        super().__init__("Bad", inputs=_BadModel.Inputs(), outputs=_BadModel.Outputs(total=total))


with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    _BadModel(ts_inflow)

w = next(w for w in caught if issubclass(w.category, InputsContractWarning))
print(f"  ⚠️  {w.category.__name__}: {w.message}")


# ===========================================================================
# §3  MODULARITY — constructor wiring, pipeline
# ===========================================================================
_section("§3  Modularity — pipeline wiring")

print("\nPipeline: InflowModel → TrafficModel → EmissionsModel → BolognaModel")
print()

# Show the wiring in BolognaModel's __init__ explicitly (mirror the real code)
i_p_start_time = Index("start time", (pd.Timestamp("07:30:00") - pd.Timestamp("00:00:00")).total_seconds())
i_p_end_time = Index("end time", (pd.Timestamp("19:30:00") - pd.Timestamp("00:00:00")).total_seconds())
i_p_cost = [Index(f"cost euro {e}", 5.00 - e * 0.25) for e in range(7)]
i_p_fraction_exempted = Index("exempted %", 0.15)
i_b_p50_cost = DistributionIndex("cost threshold", stats.uniform, {"loc": 4.00, "scale": 7.00})
i_b_p50_anticipating = Index("anticipation likelihood", 0.5)
i_b_p50_anticipation = Index("anticipation threshold", 0.25)
i_b_p50_postponing = Index("postponement likelihood", 0.8)
i_b_p50_postponement = Index("postponement threshold", 0.50)
i_b_starting_factor = Index("starting factor", 1.00)

_inflow = InflowModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    ts=ts,
    i_p_start_time=i_p_start_time,
    i_p_end_time=i_p_end_time,
    i_p_cost=i_p_cost,
    i_p_fraction_exempted=i_p_fraction_exempted,
    i_b_p50_cost=i_b_p50_cost,
    i_b_p50_anticipating=i_b_p50_anticipating,
    i_b_p50_anticipation=i_b_p50_anticipation,
    i_b_p50_postponing=i_b_p50_postponing,
    i_b_p50_postponement=i_b_p50_postponement,
    i_b_starting_modified_factor=i_b_starting_factor,
)
print(f"  InflowModel built  — {len(list(_inflow.indexes))} indexes")

# Level-1 wiring: _inflow.outputs.* → TrafficModel constructor
_traffic = TrafficModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    modified_inflow=_inflow.outputs.modified_inflow,  # ← Level-1 wire
    modified_starting=_inflow.outputs.modified_starting,  # ← Level-1 wire
)
print(f"  TrafficModel built — {len(list(_traffic.indexes))} indexes")

_emissions = EmissionsModel(
    ts=ts,
    i_p_start_time=i_p_start_time,
    i_p_end_time=i_p_end_time,
    traffic=_traffic.outputs.traffic,  # ← Level-1 wire
    modified_traffic=_traffic.outputs.modified_traffic,  # ← Level-1 wire
    modified_euro_class_split=_inflow.outputs.modified_euro_class_split,  # ← Level-1 wire
)
print(f"  EmissionsModel built — {len(list(_emissions.indexes))} indexes")

print("\nSub-models are independent objects — test them in isolation:")
print(f"  traffic.outputs.traffic_ratio.name   = {_traffic.outputs.traffic_ratio.name!r}")
print(f"  emissions.outputs.total_emissions.name = {_emissions.outputs.total_emissions.name!r}")


# ===========================================================================
# §4a  ModelVariant — CASE A: same phenomenon, two model formulations
# ===========================================================================
_section("§4a  ModelVariant — case a: same phenomenon, two model formulations")

# ── SimpleTrafficModel: direct sum, no iterative solver ─────────────────────


class SimpleTrafficModel(Model):
    """Approximation: traffic ≈ inflow + starting (no iterative solve).

    Fast to evaluate; useful for sensitivity sweeps where solver accuracy
    is less important than speed.
    """

    @dataclass
    class Inputs:
        """Inputs of :class:`SimpleTrafficModel` — identical to :class:`TrafficModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: Index
        modified_starting: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`SimpleTrafficModel` — identical to :class:`TrafficModel`."""

        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        total_modified_traffic: Index
        inflow_ratio: Index
        starting_ratio: Index
        traffic_ratio: Index

    def __init__(
        self,
        ts_inflow: TimeseriesIndex,
        ts_starting: TimeseriesIndex,
        modified_inflow: Index,
        modified_starting: Index,
    ) -> None:
        Inputs = SimpleTrafficModel.Inputs
        Outputs = SimpleTrafficModel.Outputs

        inputs = Inputs(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            modified_inflow=modified_inflow,
            modified_starting=modified_starting,
        )

        # Direct sum — no iterative steady-state solve
        traffic = TimeseriesIndex("reference traffic", inputs.ts_inflow + inputs.ts_starting)
        modified_traffic = TimeseriesIndex("modified traffic", inputs.modified_inflow + inputs.modified_starting)

        total_modified_traffic = Index("total modified traffic", modified_traffic.sum())
        inflow_ratio = Index("ratio modified/base inflow", inputs.ts_inflow / inputs.modified_inflow)
        starting_ratio = Index("ratio modified/base starting", inputs.ts_starting / inputs.modified_starting)
        traffic_ratio = Index("ratio modified/base traffic", traffic / modified_traffic)

        super().__init__(
            "SimpleTraffic",
            inputs=inputs,
            outputs=Outputs(
                traffic=traffic,
                modified_traffic=modified_traffic,
                total_modified_traffic=total_modified_traffic,
                inflow_ratio=inflow_ratio,
                starting_ratio=starting_ratio,
                traffic_ratio=traffic_ratio,
            ),
        )


# ── Build both variants and wrap in a ModelVariant ───────────────────────────

_simple_traffic = SimpleTrafficModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    modified_inflow=_inflow.outputs.modified_inflow,
    modified_starting=_inflow.outputs.modified_starting,
)

_iterative_traffic = TrafficModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    modified_inflow=_inflow.outputs.modified_inflow,
    modified_starting=_inflow.outputs.modified_starting,
)

traffic_variant_a = ModelVariant(
    "TrafficModel",
    variants={
        "simple": _simple_traffic,
        "iterative": _iterative_traffic,
    },
    selector="iterative",
)

print(f"\nModelVariant selector = 'iterative'")
print(f"  active traffic output  : {traffic_variant_a.outputs.traffic.name!r}")
print(f"  inactive (simple) reachable via variants dict:")
print(
    f"    traffic_variant_a.variants['simple'].outputs.traffic.name = "
    f"{traffic_variant_a.variants['simple'].outputs.traffic.name!r}"
)

# Demonstrate contract enforcement
print("\nContract enforcement demo (mismatched Outputs → ValueError):")


class _WrongOutputsModel(Model):
    @dataclass
    class Outputs:
        result: Index  # 'result' ≠ 'traffic', 'modified_traffic', …

    def __init__(self) -> None:
        r = Index("r", 0.0)
        super().__init__("Wrong", outputs=_WrongOutputsModel.Outputs(result=r))


try:
    ModelVariant(
        "Bad",
        variants={"iterative": _iterative_traffic, "wrong": _WrongOutputsModel()},
        selector="iterative",
    )
except ValueError as exc:
    print(f"  ✅ ValueError caught: {exc}")


# ===========================================================================
# §4b  ModelVariant — CASE B: same interface, different engine
# ===========================================================================
_section("§4b  ModelVariant — case b: same interface, different engine")


class ExternalSimulatorTrafficModel(Model):
    """Traffic model that delegates to an external simulator.

    The computation graph passes the raw inflow and starting timeseries
    directly to the 'sumo_simulate' function node, which is registered at
    evaluation time.  The CDT engine does not perform the iterative
    convergence internally — the external process handles it entirely.
    """

    @dataclass
    class Inputs:
        """Inputs — identical to :class:`TrafficModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: Index
        modified_starting: Index

    @dataclass
    class Outputs:
        """Outputs — identical to :class:`TrafficModel`."""

        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        total_modified_traffic: Index
        inflow_ratio: Index
        starting_ratio: Index
        traffic_ratio: Index

    def __init__(
        self,
        ts_inflow: TimeseriesIndex,
        ts_starting: TimeseriesIndex,
        modified_inflow: Index,
        modified_starting: Index,
    ) -> None:
        Inputs = ExternalSimulatorTrafficModel.Inputs
        Outputs = ExternalSimulatorTrafficModel.Outputs

        inputs = Inputs(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            modified_inflow=modified_inflow,
            modified_starting=modified_starting,
        )

        # Raw inflow and starting are passed to the external simulator as
        # separate arguments — the simulator combines them internally.
        # .node extracts the underlying graph.Node from each GenericIndex.
        traffic = TimeseriesIndex(
            "reference traffic",
            graph.function_call("sumo_simulate", inputs.ts_inflow.node, inputs.ts_starting.node),
        )
        modified_traffic = TimeseriesIndex(
            "modified traffic",
            graph.function_call("sumo_simulate", inputs.modified_inflow.node, inputs.modified_starting.node),
        )

        total_modified_traffic = Index("total modified traffic", modified_traffic.sum())
        inflow_ratio = Index("ratio modified/base inflow", inputs.ts_inflow / inputs.modified_inflow)
        starting_ratio = Index("ratio modified/base starting", inputs.ts_starting / inputs.modified_starting)
        traffic_ratio = Index("ratio modified/base traffic", traffic / modified_traffic)

        super().__init__(
            "ExternalSimulatorTraffic",
            inputs=inputs,
            outputs=Outputs(
                traffic=traffic,
                modified_traffic=modified_traffic,
                total_modified_traffic=total_modified_traffic,
                inflow_ratio=inflow_ratio,
                starting_ratio=starting_ratio,
                traffic_ratio=traffic_ratio,
            ),
        )


def _sumo_simulate_stub(inflow: np.ndarray, starting: np.ndarray) -> np.ndarray:
    """Stand-in for a real SUMO call.

    In production this would serialize the arrays, invoke the SUMO process,
    and deserialize the result.  Here we reuse the same iterative solver so
    the output is numerically identical to the built-in variant.
    """
    return _ts_solve(inflow + starting)


_external_traffic = ExternalSimulatorTrafficModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    modified_inflow=_inflow.outputs.modified_inflow,
    modified_starting=_inflow.outputs.modified_starting,
)

traffic_variant_b = ModelVariant(
    "TrafficModel",
    variants={
        "builtin": _iterative_traffic,
        "external": _external_traffic,
    },
    selector="builtin",
)

print(f"\nModelVariant selector = 'builtin'")
print(f"  proxy delegates to : {traffic_variant_b.variants['builtin'].__class__.__name__}")
print(f"  external variant   : {traffic_variant_b.variants['external'].__class__.__name__}")
print(f"  same Outputs field names? ", end="")
builtin_fields = set(TrafficModel.Outputs.__dataclass_fields__)
external_fields = set(ExternalSimulatorTrafficModel.Outputs.__dataclass_fields__)
print("✅ yes" if builtin_fields == external_fields else f"❌ differ: {builtin_fields ^ external_fields}")

# ── Evaluate both variants and compare results ───────────────────────────────
print("\nEvaluating both variants (size=50) and comparing total_modified_traffic …")

from civic_digital_twins.dt_model.model.index import GenericIndex


def _run_variant(traffic_mv: ModelVariant, functions: dict) -> float:
    """Build a root model around *traffic_mv* and return mean total modified traffic."""

    @dataclass
    class _RootOutputs:
        total_modified_traffic: Index
        total_emissions: Index
        total_modified_emissions: Index

    @dataclass
    class _RootExpose:
        inflow_indexes: list
        traffic_indexes: list
        emissions_indexes: list

    _em = EmissionsModel(
        ts=ts,
        i_p_start_time=i_p_start_time,
        i_p_end_time=i_p_end_time,
        traffic=traffic_mv.outputs.traffic,
        modified_traffic=traffic_mv.outputs.modified_traffic,
        modified_euro_class_split=_inflow.outputs.modified_euro_class_split,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        root = Model(
            "root",
            outputs=_RootOutputs(
                total_modified_traffic=traffic_mv.outputs.total_modified_traffic,
                total_emissions=_em.outputs.total_emissions,
                total_modified_emissions=_em.outputs.total_modified_emissions,
            ),
            expose=_RootExpose(
                inflow_indexes=list(_inflow.indexes),
                traffic_indexes=list(traffic_mv.indexes),
                emissions_indexes=list(_em.indexes),
            ),
        )

    ensemble = DistributionEnsemble(root, size=50)
    result = Evaluation(root).evaluate(ensemble, functions=functions)
    return float(result[root.outputs.total_modified_traffic].mean())


builtin_functions = {"ts_solve": executor.LambdaAdapter(_ts_solve)}
external_functions = {"sumo_simulate": executor.LambdaAdapter(_sumo_simulate_stub)}

mean_builtin = _run_variant(traffic_variant_b, builtin_functions)
mean_external = _run_variant(
    ModelVariant("T", variants={"external": _external_traffic, "builtin": _iterative_traffic}, selector="external"),
    external_functions,
)

print(f"  builtin  variant — mean total modified traffic : {mean_builtin:,.0f} veh")
print(f"  external variant — mean total modified traffic : {mean_external:,.0f} veh")
print(f"  difference (stub ≈ builtin, so should be ~0)  : {abs(mean_builtin - mean_external):,.0f} veh")


# ===========================================================================
# Done
# ===========================================================================
_section("Done")
print("\nAll sections completed successfully.")
print(f"Plots written to: {_SEMINAR_DIR}")
