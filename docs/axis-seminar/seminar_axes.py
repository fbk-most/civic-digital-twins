# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D101, D102, D103, D106  (slide code: models are documented in the deck)

"""Companion script for the CDT seminar on axes (v0.12.0) — city air quality.

A city split into three zones.  An entry fee, graduated by Euro class, changes
how many vehicles enter each zone; the resulting NOx emissions are spread over
a spatial grid and diffused.  The model carries five DOMAIN axes:

  zone        SetType       3   centre / north / south
  euro_class  SequenceType  7   Euro 0 … Euro 6
  time        TimeType     24   hourly steps (TIME_AXIS)
  x, y        SpaceType    16   0.5 km cells, clean air beyond the border

Sections mirror the slide deck (docs/axis-seminar/seminar.md):

  §1  Defining axes
  §2  Indexes with a shape
  §3  Computing along axes
  §4  Reading results
  §5  @config

Run from the repo root:

    uv run python docs/axis-seminar/seminar_axes.py

All plots are saved to docs/axis-seminar/ as PNG files.
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy import stats

from civic_digital_twins.dt_model import (
    DOMAIN,
    TIME_AXIS,
    AxesInferenceWarning,
    Axis,
    BoundaryCondition,
    ConfigTypeMismatchError,
    Constant,
    DistributionEnsemble,
    DistributionIndex,
    DomainAxis,
    Evaluation,
    GenericIndex,
    Index,
    Model,
    NumpyBackend,
    Reflect,
    Scenario,
    SequenceType,
    SetType,
    SpaceType,
    TimeseriesIndex,
    config,
    define,
    expose,
    inputs,
    outputs,
)
from civic_digital_twins.dt_model.engine.frontend import graph

_SEMINAR_DIR = Path(__file__).parent


def _section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0][:150]}"


# ===========================================================================
# §1  DEFINING AXES
# ===========================================================================
_section("§1  Defining axes")

# ── axes.py: define every axis ONCE, import it everywhere ───────────────────
ZONE = DomainAxis("zone", type=SetType())
EURO = DomainAxis("euro_class", type=SequenceType())
X = DomainAxis("x", type=SpaceType(spacing=0.5, boundary=Constant(0.0)))
Y = DomainAxis("y", type=SpaceType(spacing=0.5, boundary=Constant(0.0)))
# TIME_AXIS is ready-made: DomainAxis("time", type=TimeType())

ZONES = ("centre", "north", "south")
N_EURO, N_HOURS, N_CELLS = 7, 24, 16

for ax in (ZONE, EURO, TIME_AXIS, X, Y):
    print(f"  {ax!r}")

# Identity is (name, role): the type is metadata, not part of equality.
plain_x = Axis("x", DOMAIN)
print(f"\n  Axis('x', DOMAIN) == X          : {plain_x == X}")
print(f"  hash(Axis('x', DOMAIN)) == hash(X): {hash(plain_x) == hash(X)}")
print(f"  ... but its type                : {getattr(plain_x, 'type', None)!r}")


# ===========================================================================
# §2  INDEXES WITH A SHAPE
# ===========================================================================
_section("§2  Indexes with a shape")


# ── Named shapes: subclass Index and fix FIXED_AXES ──────────────────────────
class ShapedIndex(Index):
    """An Index whose axes are fixed by the subclass's FIXED_AXES."""

    FIXED_AXES: ClassVar[tuple[Axis, ...]] = ()

    def __init__(
        self,
        name: str,
        value: np.ndarray | graph.Node | GenericIndex | None = None,
        *,
        axes: tuple[Axis, ...] | None = None,  # ignored: the class fixes the axes
    ) -> None:
        super().__init__(name, value, axes=self.FIXED_AXES)


class ZoneIndex(ShapedIndex):
    FIXED_AXES = (ZONE,)


class EuroIndex(ShapedIndex):
    FIXED_AXES = (EURO,)


class ZoneTimeIndex(ShapedIndex):
    FIXED_AXES = (ZONE, TIME_AXIS)


class MapIndex(ShapedIndex):
    FIXED_AXES = (Y, X)


class ZoneMapIndex(ShapedIndex):
    FIXED_AXES = (ZONE, Y, X)


class FieldIndex(ShapedIndex):
    FIXED_AXES = (TIME_AXIS, Y, X)


# The uncertain variant of a shape: DistributionIndex first, so it wins construction
# (the same recipe as ConstTimeseriesIndex = ConstIndex + TimeseriesIndex).
class DistributionZoneIndex(DistributionIndex, ZoneIndex):
    def __init__(self, name: str, distribution: Callable[..., Any], params: dict[str, Any], *, shape: tuple[int, ...]):
        super().__init__(name, distribution, params, axes=self.FIXED_AXES, shape=shape)


print(f"  ZoneTimeIndex.FIXED_AXES = {tuple(a.name for a in ZoneTimeIndex.FIXED_AXES)}")
print(f"  FieldIndex.FIXED_AXES    = {tuple(a.name for a in FieldIndex.FIXED_AXES)}")

# ── Synthetic city data ─────────────────────────────────────────────────────
hours = np.arange(N_HOURS)
profile = 0.1 + np.exp(-(((hours - 8) / 1.5) ** 2)) + 0.8 * np.exp(-(((hours - 18) / 2.0) ** 2))
INFLOW = np.outer([3000.0, 1500.0, 1000.0], profile)  # (zone, time)  veh/h

FLEET_SPLIT = np.array([0.04, 0.05, 0.08, 0.13, 0.20, 0.22, 0.28])  # (euro_class,)
EMISSION_FACTOR = np.array([1.00, 0.80, 0.60, 0.45, 0.30, 0.15, 0.06])  # g NOx / km
ENTRY_FEE = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0])  # € per entry

ZONE_MAP = np.zeros((len(ZONES), N_CELLS, N_CELLS))  # (zone, y, x)
ZONE_MAP[0, 5:11, 5:11] = 1.0  # centre: the inner square
ZONE_MAP[1, 0:5, 2:14] = 1.0  # north: a band at the top
ZONE_MAP[2, 11:16, 2:14] = 1.0  # south: a band at the bottom
ZONE_MAP /= ZONE_MAP.sum(axis=(1, 2), keepdims=True)  # each zone spreads 100 % of its emissions

yy, xx = np.mgrid[0:N_CELLS, 0:N_CELLS]
INDUSTRY = 40.0 * np.exp(-(((xx - 14) ** 2 + (yy - 3) ** 2) / 4.0))  # (y, x)  plant in the north-east
INDUSTRY_ACTIVITY = np.where((hours >= 6) & (hours < 22), 1.0, 0.4)  # (time,)  two shifts, reduced at night

# fig_city.png — the zones on the grid, and the plant
ZONE_COLOURS = ("#1f77b4", "#ff7f0e", "#2ca02c")
zone_of_cell = np.full((N_CELLS, N_CELLS), np.nan)  # (y, x)
for z in range(len(ZONES)):
    zone_of_cell[ZONE_MAP[z] > 0] = z
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(zone_of_cell, cmap=ListedColormap(ZONE_COLOURS), alpha=0.6, origin="upper")
ax.contour(INDUSTRY, levels=[5.0, 20.0], colors="#555", linewidths=1)
ax.plot(14, 3, marker="^", markersize=12, color="#333")
ax.set_xticks(np.arange(-0.5, N_CELLS, 1), minor=True)
ax.set_yticks(np.arange(-0.5, N_CELLS, 1), minor=True)
ax.grid(which="minor", color="#d0d0d0", linewidth=0.5)
ax.set_xticks([-0.5, N_CELLS - 0.5], ["0", f"{N_CELLS * 0.5:g} km"])
ax.set_yticks([-0.5, N_CELLS - 0.5], ["0", f"{N_CELLS * 0.5:g} km"])
ax.set_xlabel("x")
ax.set_ylabel("y")
for name, (yc, xc) in zip(ZONES, ((7.5, 7.5), (2, 7.5), (13, 7.5)), strict=True):
    ax.text(xc, yc, name, ha="center", va="center", fontsize=13, fontweight="bold", color="#222")
ax.text(14, 1.2, "plant", ha="center", fontsize=10, color="#333")
ax.set_title("The city: 3 zones on a 16 × 16 grid")
fig.tight_layout()
fig.savefig(_SEMINAR_DIR / "fig_city.png", dpi=150)
plt.close(fig)

# ── axes= on an injected value: zipped against the array's shape, by position
inflow_idx = ZoneTimeIndex("inflow", INFLOW)
print(f"\n  ZoneTimeIndex('inflow', array{INFLOW.shape}) -> sizes {inflow_idx.sizes}")

# Same rank, wrong order: accepted silently, the sizes are simply swapped.
swapped = ZoneTimeIndex("inflow (transposed)", INFLOW.T)
print(f"  ZoneTimeIndex('inflow', INFLOW.T)       -> sizes {swapped.sizes}  (silent!)")

# ── A formula's axes are INFERRED from its operands ─────────────────────────
for label, formula in (
    ("inflow * 2", inflow_idx * 2),
    ("inflow.sum(axis=ZONE)", inflow_idx.sum(axis=ZONE)),
    ("inflow.sum(axis=ZONE).sum(axis=TIME_AXIS)", inflow_idx.sum(axis=ZONE).sum(axis=TIME_AXIS)),
):
    print(f"  {label:<42} -> inferred axes {tuple(a.name for a in formula.output_axes)}")

# ── axes= on a formula: VERIFIED against the inferred axes, never relabelled
total_inflow = TimeseriesIndex("total inflow", inflow_idx.sum(axis=ZONE))
print(
    f"  TimeseriesIndex(..., inflow.sum(axis=ZONE)) -> ok, axes {tuple(a.name for a in total_inflow.node.output_axes)}"
)
try:
    ZoneTimeIndex("total inflow", inflow_idx.sum(axis=ZONE))
except ValueError as exc:
    print(f"  ZoneTimeIndex(..., inflow.sum(axis=ZONE)) -> {_error(exc)}")

# ── Shaped DistributionIndex: one independent draw per zone ─────────────────
# Price-elasticity threshold: drivers in the centre are less sensitive to price.
cost_threshold = DistributionZoneIndex(
    "price threshold",
    stats.uniform,
    {"loc": np.array([6.0, 3.0, 3.0]), "scale": np.array([6.0, 5.0, 5.0])},
    shape=(3,),  # Note: len(ZONES) = 3
)
print(f"\n  DistributionZoneIndex: axes={tuple(a.name for a in cost_threshold.node.output_axes)}")


# ===========================================================================
# §3  COMPUTING ALONG AXES
# ===========================================================================
_section("§3  Computing along axes")

LN2 = float(np.log(2.0))


@define("Traffic")
class TrafficModel(Model):
    """Entry-fee effect on zonal inflow, and the resulting NOx emissions."""

    @inputs
    class Inputs:
        inflow: ZoneTimeIndex  # veh/h, by zone and hour
        fleet_split: EuroIndex  # share of each Euro class
        emission_factor: EuroIndex  # g NOx / km
        entry_fee: EuroIndex  # € per entry
        cost_threshold: ZoneIndex  # € at which 50 % give up — uncertain, per zone
        trip_length: Index  # km driven inside the zone

    @outputs
    class Outputs:
        modified_inflow: ZoneTimeIndex
        emissions: ZoneTimeIndex  # g/h
        hourly_emissions: TimeseriesIndex
        cumulative_emissions: TimeseriesIndex
        total_emissions: Index

    @expose
    class Expose:
        rigid_fraction: Index  # (zone, euro_class)
        modified_split: Index  # (zone, euro_class)

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        i = inputs
        # (euro_class,) / (zone,) -> (zone, euro_class): an intended outer product.
        rigid = Index(
            "rigid fraction",
            graph.exp(-(i.entry_fee / i.cost_threshold) * LN2),
            axes=(ZONE, EURO),
        )
        kept = ZoneIndex("kept fraction", (i.fleet_split * rigid).sum(axis=EURO))
        modified_split = Index("modified split", i.fleet_split * rigid / kept, axes=(ZONE, EURO))
        modified_inflow = ZoneTimeIndex("modified inflow", i.inflow * kept)
        avg_factor = ZoneIndex("average emission factor", (i.emission_factor * modified_split).sum(axis=EURO))
        emissions = ZoneTimeIndex("emissions", modified_inflow * avg_factor * i.trip_length)
        hourly = TimeseriesIndex("hourly emissions", emissions.sum(axis=ZONE))
        cumulative = TimeseriesIndex("cumulative emissions", hourly.cumulative(axis=TIME_AXIS))
        total = Index("total emissions", hourly.sum())  # one DOMAIN axis left: axis= may be omitted
        return (
            TrafficModel.Outputs(
                modified_inflow=modified_inflow,
                emissions=emissions,
                hourly_emissions=hourly,
                cumulative_emissions=cumulative,
                total_emissions=total,
            ),
            TrafficModel.Expose(rigid_fraction=rigid, modified_split=modified_split),
        )


@define("Dispersion")
class DispersionModel(Model):
    """Hourly emission source field -> NOx concentration field, by explicit diffusion."""

    @inputs
    class Inputs:
        source: FieldIndex  # g/h per cell, (time, y, x)
        diffusivity: Index  # per diffusion step
        persistence: Index  # share of the previous hour still in the air

    @config
    class Config:
        diffusion_steps: int = 6
        boundary: BoundaryCondition = Constant(0.0)

    @outputs
    class Outputs:
        concentration: FieldIndex
        exposure: MapIndex  # daily sum, (y, x)

    @classmethod
    def default_config(cls) -> Config:
        return cls.Config(diffusion_steps=6, boundary=Constant(0.0))

    def compute(self, inputs: Inputs, *, config: Config) -> Outputs:
        # Same axes (identity is (name, role)), with the boundary chosen by the config.
        x = DomainAxis("x", type=SpaceType(spacing=0.5, boundary=config.boundary))
        y = DomainAxis("y", type=SpaceType(spacing=0.5, boundary=config.boundary))

        c = FieldIndex("lingering source", inputs.source + inputs.persistence * inputs.source.shift(1, axis=TIME_AXIS))
        for step in range(config.diffusion_steps):
            c = FieldIndex(f"diffusion step {step + 1}", c + inputs.diffusivity * c.laplacian(axes=(x, y)))
        concentration = FieldIndex("concentration", c)
        exposure = MapIndex("daily exposure", concentration.sum(axis=TIME_AXIS))
        return DispersionModel.Outputs(concentration=concentration, exposure=exposure)


@define("CityAir")
class CityAirModel(Model):
    """Root: traffic emissions by zone -> grid -> dispersion, plus an industrial background."""

    @inputs
    class Inputs:
        inflow: ZoneTimeIndex
        fleet_split: EuroIndex
        emission_factor: EuroIndex
        entry_fee: EuroIndex
        cost_threshold: ZoneIndex
        trip_length: Index
        zone_map: ZoneMapIndex  # (zone, y, x), each zone sums to 1
        industry: MapIndex  # (y, x), plant emissions at full activity
        industry_activity: TimeseriesIndex  # (time,), share of full activity
        diffusivity: Index
        persistence: Index

    @config
    class Config:
        dispersion: DispersionModel.Config

    @outputs
    class Outputs:
        modified_inflow: ZoneTimeIndex
        emissions: ZoneTimeIndex
        cumulative_emissions: TimeseriesIndex
        total_emissions: Index
        traffic_concentration: FieldIndex
        industry_concentration: FieldIndex
        concentration: FieldIndex
        exposure: MapIndex

    @expose
    class Expose:
        modified_split: Index

    @classmethod
    def default_config(cls) -> Config:
        return cls.Config(dispersion=DispersionModel.default_config())

    def compute(self, inputs: Inputs, *, config: Config) -> tuple[Outputs, Expose]:
        i = inputs
        traffic = TrafficModel(
            inputs=TrafficModel.Inputs(
                inflow=i.inflow,
                fleet_split=i.fleet_split,
                emission_factor=i.emission_factor,
                entry_fee=i.entry_fee,
                cost_threshold=i.cost_threshold,
                trip_length=i.trip_length,
            )
        )
        # (zone, time) * (zone, y, x), summed over zone -> (time, y, x)
        traffic_source = FieldIndex("traffic source", (traffic.outputs.emissions * i.zone_map).sum(axis=ZONE))
        traffic_air = DispersionModel(
            inputs=DispersionModel.Inputs(source=traffic_source, diffusivity=i.diffusivity, persistence=i.persistence),
            config=config.dispersion,
        )
        # (y, x) * (time,) -> (time, y, x): an intended outer product, declared by FieldIndex.
        industry_source = FieldIndex("industry source", i.industry * i.industry_activity)
        industry_air = DispersionModel(
            inputs=DispersionModel.Inputs(source=industry_source, diffusivity=i.diffusivity, persistence=i.persistence),
            config=config.dispersion,
        )
        concentration = FieldIndex(
            "concentration",
            traffic_air.outputs.concentration + industry_air.outputs.concentration,
        )
        exposure = MapIndex("exposure", traffic_air.outputs.exposure + industry_air.outputs.exposure)
        return (
            CityAirModel.Outputs(
                modified_inflow=traffic.outputs.modified_inflow,
                emissions=traffic.outputs.emissions,
                cumulative_emissions=traffic.outputs.cumulative_emissions,
                total_emissions=traffic.outputs.total_emissions,
                traffic_concentration=traffic_air.outputs.concentration,
                industry_concentration=industry_air.outputs.concentration,
                concentration=concentration,
                exposure=exposure,
            ),
            CityAirModel.Expose(modified_split=traffic.expose.modified_split),
        )


def default_inputs(trip_length: Index | None = None) -> CityAirModel.Inputs:
    return CityAirModel.Inputs(
        inflow=ZoneTimeIndex("inflow", INFLOW),
        fleet_split=EuroIndex("fleet split", FLEET_SPLIT),
        emission_factor=EuroIndex("emission factor", EMISSION_FACTOR),
        entry_fee=EuroIndex("entry fee", ENTRY_FEE),
        cost_threshold=cost_threshold,
        trip_length=trip_length if trip_length is not None else Index("trip length", 3.0),
        zone_map=ZoneMapIndex("zone map", ZONE_MAP),
        industry=MapIndex("industry", INDUSTRY),
        industry_activity=TimeseriesIndex("industry activity", INDUSTRY_ACTIVITY),
        diffusivity=Index("diffusivity", 0.05),
        persistence=Index("persistence", 0.3),
    )


model = CityAirModel(inputs=default_inputs(), config=CityAirModel.default_config())
print(f"  CityAirModel built: {len(model.indexes)} indexes, abstract: {[i.name for i in model.abstract_indexes()]}")

# ── 3a. Reductions name their axis ──────────────────────────────────────────
print("\n  3a. Reductions name their axis")
try:
    model.outputs.emissions.sum()
except ValueError as exc:
    print(f"    emissions.sum()           -> {_error(exc)}")
print(
    f"    emissions.sum(axis=ZONE)  -> axes {tuple(a.name for a in model.outputs.emissions.sum(axis=ZONE).output_axes)}"
)

# ── 3b. Disjoint axes: an outer product, flagged unless declared ─────────────
print("\n  3b. Disjoint operands: an outer product emerges")
fee, thr = EuroIndex("fee", ENTRY_FEE), ZoneIndex("thr", np.array([8.0, 5.0, 5.0]))
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    undeclared = Index("fee ratio (undeclared)", fee / thr)
print(f"    Index(..., fee / thr) -> axes {tuple(a.name for a in undeclared.node.output_axes)}")
for w in caught:
    if issubclass(w.category, AxesInferenceWarning):
        print(f"    AxesInferenceWarning: {str(w.message).splitlines()[0][:110]}…")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Index("fee ratio (declared)", fee / thr, axes=(ZONE, EURO))
print(f"    ... with axes=(ZONE, EURO): {len(caught)} warning(s)")

# ── 3c. .broadcast(): an axis that nothing in the formula provides ──────────
print("\n  3c. .broadcast()")
industry = MapIndex("industry", INDUSTRY)
try:
    DispersionModel(
        inputs=DispersionModel.Inputs(
            source=industry,  # pyright: ignore[reportArgumentType]  — the violation is the point
            diffusivity=Index("d", 0.05),
            persistence=Index("p", 0.3),
        ),
        config=DispersionModel.default_config(),
    )
except ValueError as exc:
    print(f"    source=industry (y, x)              -> {_error(exc)}")
try:
    FieldIndex("industry source", industry)
except ValueError as exc:
    print(f"    FieldIndex(..., industry)           -> {_error(exc)}")
ok = FieldIndex("industry source", industry.broadcast(TIME_AXIS))
print(f"    FieldIndex(..., industry.broadcast(TIME_AXIS)) -> axes {tuple(a.name for a in ok.node.output_axes)}")


# broadcast() promises the axis at size 1; it does not repeat the data along it.
@define("Broadcast")
class BroadcastModel(Model):
    @inputs
    class Inputs:
        industry: MapIndex
        activity: TimeseriesIndex

    @outputs
    class Outputs:
        daily_broadcast: MapIndex
        daily_with_profile: MapIndex

    def compute(self, inputs: Inputs) -> Outputs:
        broadcast = FieldIndex("broadcast", inputs.industry.broadcast(TIME_AXIS))
        profiled = FieldIndex("with profile", inputs.industry * inputs.activity)
        return BroadcastModel.Outputs(
            daily_broadcast=MapIndex("daily (broadcast)", broadcast.sum(axis=TIME_AXIS)),
            daily_with_profile=MapIndex("daily (with profile)", profiled.sum(axis=TIME_AXIS)),
        )


bc_model = BroadcastModel(
    inputs=BroadcastModel.Inputs(industry=industry, activity=TimeseriesIndex("always on", np.ones(N_HOURS)))
)
bc_res = Evaluation(Scenario(bc_model)).evaluate()
peak_cell = float(INDUSTRY.max())
print(f"    peak cell {peak_cell:.0f} g/h, summed over the day:")
daily_broadcast = np.max(bc_res.expected_value(bc_model.outputs.daily_broadcast))
daily_profile = np.max(bc_res.expected_value(bc_model.outputs.daily_with_profile))
print(f"      industry.broadcast(TIME).sum(axis=TIME)  -> {daily_broadcast:.0f}  (size 1: counted once)")
print(f"      (industry * ones(24)).sum(axis=TIME)     -> {daily_profile:.0f}")

# ── 3d. Registered functions see positions, not names ───────────────────────
print("\n  3d. Registered functions: a numpy function sees positions, not names")


def peak_by_last_axis(values: np.ndarray) -> np.ndarray:
    """Peak hour — assuming time is the LAST dimension (it is not, here)."""
    return np.max(values, axis=-1, keepdims=True)


@define("Peak")
class PeakModel(Model):
    @inputs
    class Inputs:
        inflow: ZoneTimeIndex

    @outputs
    class Outputs:
        peak_unsigned: Index
        peak_named: ZoneIndex

    def compute(self, inputs: Inputs) -> Outputs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AxesInferenceWarning)
            unsigned = Index("peak (registered fn)", graph.function_call("peak", inputs.inflow))
        return PeakModel.Outputs(
            peak_unsigned=unsigned,
            peak_named=ZoneIndex("peak (named)", inputs.inflow.max(axis=TIME_AXIS)),
        )


peak_model = PeakModel(inputs=PeakModel.Inputs(inflow=ZoneTimeIndex("inflow", INFLOW)))
peak_res = Evaluation(Scenario(peak_model)).evaluate(functions={"peak": NumpyBackend.adapt(peak_by_last_axis)})
unsigned = np.asarray(peak_res.expected_value(peak_model.outputs.peak_unsigned))
named = peak_res.labeled(peak_model.outputs.peak_named)
print(f"    DOMAIN layout {[a.name for a, _ in peak_res.layout.entries]}: time is NOT the last axis")
print(f"    np.max(axis=-1)       -> shape {unsigned.shape}, h0–2 {unsigned[:3].ravel().round(0)} (max over ZONES!)")
print(f"    inflow.max(axis=TIME) -> dims {named.dims}, values {named.values.round(0)}")

signed_peak = NumpyBackend.adapt(peak_by_last_axis, output_axes=(ZONE,), input_axes=((ZONE, TIME_AXIS),))
fc = graph.function_call("peak", ZoneTimeIndex("inflow", INFLOW), functor=signed_peak)
print(f"    declared signature    -> output_axes {tuple(a.name for a in fc.output_axes)} (checked at build time)")
try:
    graph.function_call("peak", EuroIndex("fee", ENTRY_FEE), functor=signed_peak)
except ValueError as exc:
    print(f"    ... wrong argument    -> {_error(exc)}")


# ===========================================================================
# §4  READING RESULTS
# ===========================================================================
_section("§4  Reading results")

ENSEMBLE_SIZE = 100

policy = Scenario(model)
baseline = Scenario(model, overrides={model.inputs.entry_fee: np.zeros(N_EURO)})
res = Evaluation(policy).evaluate(
    ensemble=DistributionEnsemble(policy, size=ENSEMBLE_SIZE, rng=np.random.default_rng(42))
)
res_base = Evaluation(baseline).evaluate(
    ensemble=DistributionEnsemble(baseline, size=ENSEMBLE_SIZE, rng=np.random.default_rng(42))
)

# ── 4a. result.layout (union) vs layout_of(index) (per output) ──────────────
print("\n  4a. result.layout — the union over the whole evaluation:")
for ax, size in res.layout.entries:
    print(f"    {ax.name:<12} {ax.role:<9} size={size}")

print("\n  4b. layout_of(index) / labeled(index) — the truth per output:")
for name in ("modified_inflow", "emissions", "cumulative_emissions", "total_emissions", "concentration", "exposure"):
    idx = getattr(model.outputs, name)
    declared = tuple(a.name for a in idx.node.output_axes)
    lab = res.labeled(idx)
    print(f"    {name:<22} declared {str(declared):<28} -> dims {str(lab.dims):<24} shape {lab.values.shape}")

# ── 4c. Name-based access ───────────────────────────────────────────────────
inflow_lab = res.labeled(model.outputs.modified_inflow)
centre = inflow_lab.sel(zone=ZONES.index("centre"))
print(
    f"\n  4c. labeled(modified_inflow).sel(zone=0)          -> dims {centre.dims}, peak {centre.values.max():.0f} veh/h"
)
print(f"      labeled(modified_inflow).sel(time=slice(7, 10)) -> dims {inflow_lab.sel(time=slice(7, 10)).dims}")

# The alphabetical trap: declared (zone, time), returned (time, zone).
raw = np.asarray(res.expected_value(model.outputs.modified_inflow))
print(f"\n  4d. The trap: expected_value(modified_inflow).shape = {raw.shape}  (declared (zone, time)!)")
print(f"      raw[0] is hour 0 for every zone: {raw[0].round(0)} — not the centre's day")
by_name = np.transpose(inflow_lab.values, [inflow_lab.dims.index(n) for n in ("zone", "time")])
print(f"      reordered by name -> shape {by_name.shape}")

try:
    xr = res.labeled(model.outputs.concentration).to_xarray()
    print(f"\n  4e. to_xarray(): {xr.dims}")
except ImportError:
    print("\n  4e. to_xarray(): xarray not installed (optional) — skipped")

# ── 4f. All three roles in one result: sweep the trip length ────────────────
trip_length = Index("trip length")  # no value: supplied by the sweep
swept = CityAirModel(inputs=default_inputs(trip_length=trip_length), config=CityAirModel.default_config())
sweep = Scenario(swept, parameter_axes=[trip_length])
res_sweep = Evaluation(sweep).evaluate(
    ensemble=DistributionEnsemble(sweep, size=ENSEMBLE_SIZE, rng=np.random.default_rng(42)),
    parameters={trip_length: np.array([2.0, 3.0, 4.0])},
)
print("\n  4f. Sweeping the trip length over [2, 3, 4] km — res_sweep.layout:")
for ax, size in res_sweep.layout.entries:
    print(f"    {ax.name:<12} {ax.role:<9} size={size}")
print(f"    res_sweep.labeled(emissions).dims = {res_sweep.labeled(swept.outputs.emissions).dims}")

# ── KPIs ─────────────────────────────────────────────────────────────────────
tot = np.asarray(res[model.outputs.total_emissions]).ravel()
tot_base = float(res_base.expected_value(model.outputs.total_emissions))
exp_pol = res.labeled(model.outputs.exposure)
exp_base = res_base.labeled(model.outputs.exposure)
print("\n  KPIs (policy vs no fee):")
print(f"    total NOx, no fee   : {tot_base / 1000:8.1f} kg/day")
print(
    f"    total NOx, policy   : {tot.mean() / 1000:8.1f} kg/day"
    f"   (5th–95th pct {np.percentile(tot, 5) / 1000:.1f} – {np.percentile(tot, 95) / 1000:.1f})"
)
print(f"    peak exposure       : {exp_base.values.max():8.0f} -> {exp_pol.values.max():.0f}")


# ── Figures ──────────────────────────────────────────────────────────────────
def _zone_time(result, idx) -> np.ndarray:
    lab = result.labeled(idx)
    return np.transpose(lab.values, [lab.dims.index(n) for n in ("zone", "time")])


def _yx(lab) -> np.ndarray:
    return np.transpose(lab.values, [lab.dims.index(n) for n in ("y", "x")])


# fig_zone_time.png — modified inflow by zone, base vs policy
fig, ax = plt.subplots(figsize=(8, 4))
pol, base = _zone_time(res, model.outputs.modified_inflow), _zone_time(res_base, model.outputs.modified_inflow)
for z, name in enumerate(ZONES):
    ax.plot(hours, base[z], "--", color=ZONE_COLOURS[z], alpha=0.6)
    ax.plot(hours, pol[z], "-", color=ZONE_COLOURS[z], label=name)
ax.set_xlabel("hour of day")
ax.set_ylabel("vehicle inflow [veh/h]")
ax.set_title("Inflow by zone — dashed: no fee, solid: entry fee (expected value)")
ax.legend()
fig.tight_layout()
fig.savefig(_SEMINAR_DIR / "fig_zone_time.png", dpi=150)
plt.close(fig)

# fig_euro_class.png — fleet mix by Euro class, base vs policy, per zone
split = res.labeled(model.expose.modified_split)
fig, ax = plt.subplots(figsize=(8, 4))
width = 0.2
ax.bar(np.arange(N_EURO) - 1.5 * width, FLEET_SPLIT, width, color="#999", label="no fee")
for z, name in enumerate(ZONES):
    ax.bar(
        np.arange(N_EURO) + (z - 0.5) * width,
        split.sel(zone=z).values,
        width,
        color=ZONE_COLOURS[z],
        label=f"fee — {name}",
    )
ax.set_xticks(np.arange(N_EURO), [f"Euro {e}" for e in range(N_EURO)])
ax.set_ylabel("share of entering vehicles")
ax.set_title("Fleet mix by Euro class — (zone, euro_class) from one formula")
ax.legend()
fig.tight_layout()
fig.savefig(_SEMINAR_DIR / "fig_euro_class.png", dpi=150)
plt.close(fig)

# fig_exposure.png — daily exposure, base vs policy, and the reduction
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
vmax = float(_yx(exp_base).max())
for a, data, title in (
    (axs[0], _yx(exp_base), "no fee"),
    (axs[1], _yx(exp_pol), "entry fee"),
):
    im = a.imshow(data, origin="upper", cmap="magma_r", vmin=0, vmax=vmax)
    a.set_title(f"daily NOx exposure — {title}")
    fig.colorbar(im, ax=a, shrink=0.8)
im = axs[2].imshow(_yx(exp_base) - _yx(exp_pol), origin="upper", cmap="Greens")
axs[2].set_title("reduction")
fig.colorbar(im, ax=axs[2], shrink=0.8)
for a in axs:
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig(_SEMINAR_DIR / "fig_exposure.png", dpi=150)
plt.close(fig)

# fig_attribution.png — concentration at 08:00: traffic vs industry
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for a, name, title in (
    (axs[0], "traffic_concentration", "traffic"),
    (axs[1], "industry_concentration", "industry"),
    (axs[2], "concentration", "total"),
):
    lab = res.labeled(getattr(model.outputs, name)).sel(time=8)
    im = a.imshow(_yx(lab), origin="upper", cmap="magma_r")
    a.set_title(f"08:00 — {title}")
    a.set_xticks([])
    a.set_yticks([])
    fig.colorbar(im, ax=a, shrink=0.8)
fig.tight_layout()
fig.savefig(_SEMINAR_DIR / "fig_attribution.png", dpi=150)
plt.close(fig)

print("\n  Figures: fig_zone_time.png, fig_euro_class.png, fig_exposure.png, fig_attribution.png")


# ===========================================================================
# §5  @config
# ===========================================================================
_section("§5  @config")

print(f"  CityAirModel.default_config() = {CityAirModel.default_config()}")

# ── Same inputs, different construction-time choices ────────────────────────
configs = {
    "no diffusion": CityAirModel.Config(dispersion=DispersionModel.Config(diffusion_steps=0)),
    "Constant(0) — clean air outside": CityAirModel.Config(
        dispersion=DispersionModel.Config(diffusion_steps=12, boundary=Constant(0.0))
    ),
    "Reflect — closed valley": CityAirModel.Config(
        dispersion=DispersionModel.Config(diffusion_steps=12, boundary=Reflect)
    ),
}
fig, axs = plt.subplots(1, len(configs), figsize=(12, 4))
for a, (title, cfg) in zip(axs, configs.items(), strict=True):
    m = CityAirModel(inputs=default_inputs(), config=cfg)
    sc = Scenario(m)
    r = Evaluation(sc).evaluate(ensemble=DistributionEnsemble(sc, size=20, rng=np.random.default_rng(0)))
    field = _yx(r.labeled(m.outputs.exposure))
    print(f"  {title:<33} total exposure={field.sum():9.0f}   peak={field.max():5.0f}")
    im = a.imshow(field, origin="upper", cmap="magma_r")
    a.set_title(title)
    a.set_xticks([])
    a.set_yticks([])
    fig.colorbar(im, ax=a, shrink=0.8)
fig.tight_layout()
fig.savefig(_SEMINAR_DIR / "fig_config.png", dpi=150)
plt.close(fig)
print("  Figure: fig_config.png")

# ── Config is type-checked at the model boundary ────────────────────────────
try:
    CityAirModel(inputs=default_inputs(), config=DispersionModel.default_config())  # type: ignore[arg-type]
except ConfigTypeMismatchError as exc:
    print(f"\n  wrong Config class  -> {_error(exc)}")

# ── ... and never holds an Index ────────────────────────────────────────────
try:
    DispersionModel.Config(diffusion_steps=Index("steps", 12))  # type: ignore[arg-type]
except TypeError as exc:
    print(f"  Index inside Config -> {_error(exc)}")
