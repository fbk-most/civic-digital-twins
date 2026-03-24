"""Companion script for the CDT seminar — Bologna mobility case study.

Sections mirror the slide deck (docs/seminar/seminar.md):

  §1  A first end-to-end run             (CDT basics)
  §2  IO Contracts                       (Inputs / Outputs / Expose)
  §3  Modularity                         (Constructor wiring, pipeline)
  §4a ModelVariant — model formulation   (LinearTrafficModel vs SolverTrafficModel)
  §4b ModelVariant — engine variant      (SolverTrafficModel vs FtsTrafficModel)

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
    plot_field_graph,
    roundup,
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

# ── Per-scenario statistics ──────────────────────────────────────────────────
_nox_reduction = evals[model.outputs.total_emissions][:, 0] - evals[model.outputs.total_modified_emissions][:, 0]

_scalar_kpis = [
    ("Base inflow", evals[model.outputs.total_base_inflow][:, 0], "veh/day", True),
    ("Modified inflow", evals[model.outputs.total_modified_inflow][:, 0], "veh/day", False),
    ("Shifted inflow", evals[model.outputs.total_shifted][:, 0], "veh/day", False),
    ("Paying vehicles", evals[model.outputs.total_paying][:, 0], "veh/day", False),
    ("Fees collected", evals[model.outputs.total_payed][:, 0], "EUR/day", False),
    ("NOx reduction", _nox_reduction, "g/day", False),
]

print("\nKPI statistics (mean, std, 5th–95th percentile):")
_col = "  {:<22s}  {:>10s}  {:>8s}  {:>10s}  {:>10s}  {}"
print(_col.format("KPI", "Mean", "Std", "5th pct", "95th pct", "Unit"))
print("  " + "─" * 75)
_kpi_stats = []
for label, arr, unit, no_uncertainty in _scalar_kpis:
    mean = float(arr.mean())
    std = float(arr.std())
    p5, p95 = (float(x) for x in np.percentile(arr, [5, 95]))
    print(_col.format(label, f"{mean:,.0f}", f"{std:,.0f}", f"{p5:,.0f}", f"{p95:,.0f}", unit))
    _kpi_stats.append((label, mean, std, p5, p95, unit, no_uncertainty))


# ── Write kpi_summary.txt (slide-ready format) ───────────────────────────────
def _slide_fmt(mean: float, p5: float, p95: float, unit: str, no_uncertainty: bool) -> tuple[str, str]:
    """Format mean and range as they appear in the slide's KPI table."""
    if unit == "veh/day":
        mean_s = f"{mean:,.0f} veh/d"
        range_s = "—" if no_uncertainty else f"{p5 / 1000:.0f} k – {p95 / 1000:.0f} k"
    elif unit == "g/day":
        mean_s = f"{mean:,.0f} g/d"
        range_s = "—" if no_uncertainty else f"{p5 / 1000:.1f} – {p95 / 1000:.1f} kg/d"
    elif unit == "EUR/day":
        mean_s = f"{mean:,.0f} €/d"
        range_s = "—" if no_uncertainty else f"{p5 / 1000:.0f} k – {p95 / 1000:.0f} k €"
    else:
        mean_s = f"{mean:,.0f} {unit}"
        range_s = "—" if no_uncertainty else f"{p5:,.0f} – {p95:,.0f}"
    return mean_s, range_s


_summary_path = _SEMINAR_DIR / "kpi_summary.txt"
with open(_summary_path, "w") as _fh:
    _fh.write("Bologna ZTL pricing — KPI summary\n")
    _fh.write("200-scenario ensemble · cost_threshold ~ Uniform(4 €, 11 €)\n")
    _fh.write("=" * 66 + "\n")
    _fh.write(f"{'KPI':<24}  {'Mean':>16}  {'5th – 95th pct':>20}\n")
    _fh.write("─" * 66 + "\n")
    for label, mean, std, p5, p95, unit, no_unc in _kpi_stats:
        mean_s, range_s = _slide_fmt(mean, p5, p95, unit, no_unc)
        _fh.write(f"{label:<24}  {mean_s:>16}  {range_s:>20}\n")
print(f"\nKPI summary written → {_summary_path}")
print("  (copy these numbers into the 'Bologna: what the model tells us' slide before the PR)")

# ── Uncertainty range chart (kpi_uncertainty.png) ────────────────────────────
_t_stats = [(l, m, s, p5, p95, u) for l, m, s, p5, p95, u, no_u in _kpi_stats if u == "veh/day" and not no_u]
_n_stats = [(l, m, s, p5, p95, u) for l, m, s, p5, p95, u, no_u in _kpi_stats if u == "g/day"]

fig2, (ax_t, ax_n) = plt.subplots(1, 2, figsize=(11, 3.5))

_colors_t = ["#55A868", "#DD8452", "#C44E52"]
for i, (label, mean, std, p5, p95, unit) in enumerate(_t_stats):
    ax_t.barh(i, mean, color=_colors_t[i % len(_colors_t)], alpha=0.75, height=0.55)
    ax_t.errorbar(mean, i, xerr=[[mean - p5], [p95 - mean]], fmt="none", color="#333333", capsize=5, linewidth=1.5)
ax_t.set_yticks(range(len(_t_stats)))
ax_t.set_yticklabels([s[0] for s in _t_stats])
ax_t.set_xlabel("vehicles / day")
ax_t.set_title("Traffic KPIs")
ax_t.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f} k"))

label, mean, std, p5, p95, unit = _n_stats[0]
ax_n.barh(0, mean, color="#8172B2", alpha=0.75, height=0.55)
ax_n.errorbar(mean, 0, xerr=[[mean - p5], [p95 - mean]], fmt="none", color="#333333", capsize=5, linewidth=1.5)
ax_n.set_yticks([0])
ax_n.set_yticklabels(["NOx reduction"])
ax_n.set_xlabel("g / day")
ax_n.set_title("Emissions KPI")

fig2.suptitle(
    "Bologna ZTL pricing — KPI uncertainty  (200 scenarios, whiskers = 5th–95th percentile)",
    fontsize=9,
)
fig2.tight_layout()
out_u = _SEMINAR_DIR / "kpi_uncertainty.png"
fig2.savefig(out_u, dpi=150)
print(f"Uncertainty plot saved → {out_u}")
plt.close(fig2)

# ── Timeseries uncertainty plots ─────────────────────────────────────────────
_ts_traffic = evals[model.expose.traffic]
_ts_mod_traffic = evals[model.expose.modified_traffic]
_ts_emissions = evals[model.expose.emissions]

fig_t = plot_field_graph(
    _ts_mod_traffic,
    "Time of day",
    "Circulating traffic (veh)",
    vertical_size=roundup(float(np.max(_ts_traffic))),
    reference_line=_ts_traffic.mean(axis=0),
)
out_tt = _SEMINAR_DIR / "kpi_traffic_ts.png"
fig_t.savefig(out_tt, dpi=150)
print(f"Traffic timeseries plot saved → {out_tt}")
plt.close(fig_t)

fig_e = plot_field_graph(
    _ts_emissions,
    "Time of day",
    "NOx emissions (g)",
    vertical_size=roundup(float(np.max(_ts_emissions))),
)
out_et = _SEMINAR_DIR / "kpi_emissions_ts.png"
fig_e.savefig(out_et, dpi=150)
print(f"Emissions timeseries plot saved → {out_et}")
plt.close(fig_e)

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
    modified_inflow=_inflow.outputs.modified_inflow,  # ← contractual output
    modified_starting=_inflow.outputs.modified_starting,  # ← contractual output
)
print(f"  TrafficModel built — {len(list(_traffic.indexes))} indexes")

_emissions = EmissionsModel(
    ts=ts,
    i_p_start_time=i_p_start_time,
    i_p_end_time=i_p_end_time,
    traffic=_traffic.outputs.traffic,  # ← contractual output
    modified_traffic=_traffic.outputs.modified_traffic,  # ← contractual output
    modified_euro_class_split=_inflow.outputs.modified_euro_class_split,  # ← contractual output
)
print(f"  EmissionsModel built — {len(list(_emissions.indexes))} indexes")

print("\nSub-models are independent objects — test them in isolation:")
print(f"  traffic.outputs.traffic_ratio.name   = {_traffic.outputs.traffic_ratio.name!r}")
print(f"  emissions.outputs.total_emissions.name = {_emissions.outputs.total_emissions.name!r}")


# ===========================================================================
# §4a  ModelVariant — CASE A: same phenomenon, two model formulations
# ===========================================================================
_section("§4a  ModelVariant — case a: same phenomenon, two model formulations")

# ── LinearTrafficModel: direct sum, no iterative solver ──────────────────────
# NOTE: in the slides this class is called TrafficModel for simplicity;
# it is renamed here to avoid shadowing the imported Bologna TrafficModel.


class LinearTrafficModel(Model):
    """Approximation: traffic ≈ inflow + starting (direct sum, no iterative solve).

    Corresponds to the TrafficModel shown in Part 3 of the slides.
    Used here as Variant A (the baseline) in the ModelVariant demo.
    """

    @dataclass
    class Inputs:
        """Inputs of :class:`LinearTrafficModel` — identical to :class:`TrafficModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: Index
        modified_starting: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`LinearTrafficModel` — identical to :class:`TrafficModel`."""

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
        Inputs = LinearTrafficModel.Inputs
        Outputs = LinearTrafficModel.Outputs

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
            "LinearTraffic",
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

_linear_traffic = LinearTrafficModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    modified_inflow=_inflow.outputs.modified_inflow,
    modified_starting=_inflow.outputs.modified_starting,
)

# SolverTrafficModel in the slides = the real Bologna TrafficModel (uses ts_solve)
_solver_traffic = TrafficModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    modified_inflow=_inflow.outputs.modified_inflow,
    modified_starting=_inflow.outputs.modified_starting,
)

traffic_variant_a = ModelVariant(
    "TrafficModel",
    variants={
        "linear": _linear_traffic,
        "solver": _solver_traffic,
    },
    selector="solver",
)

print(f"\nModelVariant selector = 'solver'")
print(f"  active traffic output  : {traffic_variant_a.outputs.traffic.name!r}")
print(f"  inactive (linear) reachable via variants dict:")
print(
    f"    traffic_variant_a.variants['linear'].outputs.traffic.name = "
    f"{traffic_variant_a.variants['linear'].outputs.traffic.name!r}"
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
        variants={"solver": _solver_traffic, "wrong": _WrongOutputsModel()},
        selector="solver",
    )
except ValueError as exc:
    print(f"  ✅ ValueError caught: {exc}")


# ===========================================================================
# §4b  ModelVariant — CASE B: same interface, different engine
# ===========================================================================
_section("§4b  ModelVariant — case b: same interface, different engine")


class FtsTrafficModel(Model):
    """Traffic model that delegates to FTS — FBK's Fast Traffic Simulator.

    The computation graph passes the raw inflow and starting timeseries
    directly to the 'fts_simulate' function node, which is registered at
    evaluation time.  The CDT engine does not perform the iterative
    convergence internally — the external process handles it entirely.
    """

    @dataclass
    class Inputs:
        """Inputs of :class:`FtsTrafficModel` — identical to :class:`TrafficModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: Index
        modified_starting: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`FtsTrafficModel` — identical to :class:`TrafficModel`."""

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
        Inputs = FtsTrafficModel.Inputs
        Outputs = FtsTrafficModel.Outputs

        inputs = Inputs(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            modified_inflow=modified_inflow,
            modified_starting=modified_starting,
        )

        # Raw inflow and starting are passed to FTS as separate arguments —
        # the simulator combines them internally.
        # .node extracts the underlying graph.Node from each GenericIndex.
        traffic = TimeseriesIndex(
            "reference traffic",
            graph.function_call("fts_simulate", inputs.ts_inflow.node, inputs.ts_starting.node),
        )
        modified_traffic = TimeseriesIndex(
            "modified traffic",
            graph.function_call("fts_simulate", inputs.modified_inflow.node, inputs.modified_starting.node),
        )

        total_modified_traffic = Index("total modified traffic", modified_traffic.sum())
        inflow_ratio = Index("ratio modified/base inflow", inputs.ts_inflow / inputs.modified_inflow)
        starting_ratio = Index("ratio modified/base starting", inputs.ts_starting / inputs.modified_starting)
        traffic_ratio = Index("ratio modified/base traffic", traffic / modified_traffic)

        super().__init__(
            "FtsTraffic",
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


def _fts_simulate_stub(inflow: np.ndarray, starting: np.ndarray) -> np.ndarray:
    """Stand-in for a real FTS (FBK's Fast Traffic Simulator) call.

    In production this would serialize the arrays, invoke the FTS process,
    and deserialize the result.  Here we reuse the same iterative solver so
    the output is numerically identical to the built-in variant.
    """
    return _ts_solve(inflow + starting)


_fts_traffic = FtsTrafficModel(
    ts_inflow=ts_inflow,
    ts_starting=ts_starting,
    modified_inflow=_inflow.outputs.modified_inflow,
    modified_starting=_inflow.outputs.modified_starting,
)

traffic_variant_b = ModelVariant(
    "TrafficModel",
    variants={
        "solver": _solver_traffic,
        "fts": _fts_traffic,
    },
    selector="solver",
)

print(f"\nModelVariant selector = 'solver'")
print(f"  proxy delegates to : {traffic_variant_b.variants['solver'].__class__.__name__}")
print(f"  fts variant        : {traffic_variant_b.variants['fts'].__class__.__name__}")
print(f"  same Outputs field names? ", end="")
solver_fields = set(TrafficModel.Outputs.__dataclass_fields__)
fts_fields = set(FtsTrafficModel.Outputs.__dataclass_fields__)
print("✅ yes" if solver_fields == fts_fields else f"❌ differ: {solver_fields ^ fts_fields}")

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


solver_functions = {"ts_solve": executor.LambdaAdapter(_ts_solve)}
fts_functions = {"fts_simulate": executor.LambdaAdapter(_fts_simulate_stub)}

mean_solver = _run_variant(traffic_variant_b, solver_functions)
mean_fts = _run_variant(
    ModelVariant("T", variants={"fts": _fts_traffic, "solver": _solver_traffic}, selector="fts"),
    fts_functions,
)

print(f"  solver variant — mean total modified traffic : {mean_solver:,.0f} veh")
print(f"  fts    variant — mean total modified traffic : {mean_fts:,.0f} veh")
print(f"  difference (stub ≈ solver, so should be ~0)  : {abs(mean_solver - mean_fts):,.0f} veh")


# ===========================================================================
# Done
# ===========================================================================
_section("Done")
print("\nAll sections completed successfully.")
print(f"Plots written to: {_SEMINAR_DIR}")
