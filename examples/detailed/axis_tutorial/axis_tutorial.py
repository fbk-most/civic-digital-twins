# SPDX-License-Identifier: Apache-2.0

"""A timeseries and a matrix combined, and how axes are tracked across outputs.

Narrative and explanations: see Axis_Tutorial.md in this directory.
Each "PART N" banner below matches a section of the same name there.

  PART 1  Defining axes: ready-made vs custom, typed properties, reading back.
  PART 2  Axis roles together, result.layout vs layout_of(), name-based access.
  PART 3  Multiplying two arrays: which cases make an axis EMERGE (and warn).
  PART 4  The same products evaluated: shared axis, outer product, dot product.
  PART 5  .broadcast(): adding an axis nothing else in the formula provides.
"""

import warnings
from typing import ClassVar

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    AxesInferenceWarning,
    DistributionEnsemble,
    DistributionIndex,
    Evaluation,
    Index,
    Model,
    Scenario,
    TimeseriesIndex,
    define,
    graph,
    inputs,
    outputs,
)
from civic_digital_twins.dt_model.axes import (
    DOMAIN,
    ENSEMBLE,
    PARAMETER,
    TIME_AXIS,
    Axis,
    DomainAxis,
    SequenceType,
    SetType,
    SpaceType,
    TimeType,
    Wrap,
)

# =============================================================================
# PART 1 -- defining axes: ready-made vs custom, properties, reading them back
# =============================================================================
print("=" * 77)
print(" PART 1 -- defining axes: ready-made vs custom, properties, reading back")
print("=" * 77)

# --- what comes ready-made --------------------------------------------------
print("\n=== ready-made: the three roles, and TIME_AXIS ===")
print(f"  roles     : {DOMAIN!r}, {PARAMETER!r}, {ENSEMBLE!r}")
print(f"  TIME_AXIS : {TIME_AXIS!r}")
ts = TimeseriesIndex("ts", np.array([1.0, 2.0, 3.0]))
print(f"  TimeseriesIndex('ts', ...).node.output_axes = {ts.node.output_axes}")

# --- a custom axis with properties ------------------------------------------
# A physical 1-D grid: neighbours 0.5 apart, and periodic (the right end wraps
# round to the left). Both facts are stored ON the axis, so every operator
# that touches x reads the same spacing and boundary.
x = DomainAxis("x", type=SpaceType(spacing=0.5, boundary=Wrap()))

# row/col index a matrix: ordered positions, not calendar time and not a
# physical grid, so SequenceType is the honest description (used from PART 2).
row = DomainAxis("row", type=SequenceType())
col = DomainAxis("col", type=SequenceType())

print("\n=== custom axes ===")
print(f"  x   = {x!r}")
print(f"  row = {row!r}")
print(f"  x.role = {x.role!r}   (a DomainAxis is always DOMAIN)")

# --- the type decides which operators an axis supports ----------------------
probe_axes = {
    "Axis(name, DOMAIN)":       Axis("p1", DOMAIN),
    "DomainAxis (untyped)":     DomainAxis("p2"),
    "DomainAxis SetType":       DomainAxis("p3", type=SetType()),
    "DomainAxis SequenceType":  DomainAxis("p4", type=SequenceType()),
    "DomainAxis TimeType":      DomainAxis("p5", type=TimeType()),
    "DomainAxis SpaceType":     DomainAxis("p6", type=SpaceType(spacing=2.0)),
}
OPS = ("sum", "diff", "cumulative", "shift", "gradient", "laplacian")

print(f"\n  {'axis declaration':26s}" + "".join(f"{o:11s}" for o in OPS))
for label, ax in probe_axes.items():
    probe = Index("probe", np.array([1.0, 2.0, 4.0, 8.0]), axes=(ax,))
    cells = ""
    for op in OPS:
        try:
            # laplacian takes axes=(...), the others take axis=...
            if op == "laplacian":
                getattr(probe, op)(axes=(ax,))
            else:
                getattr(probe, op)(axis=ax)
            cells += f"{'ok':11s}"
        except ValueError:
            cells += f"{'--':11s}"
    print(f"  {label:26s}{cells}")

# --- typing never breaks matching -------------------------------------------
plain = Axis("time", DOMAIN)
print("\n=== identity is (name, role): the type is not part of it ===")
print(f"  Axis('time', DOMAIN) == TIME_AXIS : {plain == TIME_AXIS}")
print(f"  same hash                         : {hash(plain) == hash(TIME_AXIS)}")
x_coarse = DomainAxis("x", type=SpaceType(spacing=10.0))
print(f"  x == DomainAxis('x', spacing=10)  : {x == x_coarse}  <- conflicting copy, still 'equal'")

# --- attaching axes to an index ---------------------------------------------
U_VAL = np.array([0.0, 1.0, 4.0, 9.0])
u_on_x = Index("u_on_x", U_VAL, axes=(x,))
u_bare = Index("u_bare", U_VAL)
print("\n=== an index carries exactly the axes you attach ===")
print(f"  Index('u_on_x', array, axes=(x,)) -> {u_on_x.node.output_axes}")
print(f"  Index('u_bare', array)            -> {u_bare.node.output_axes}  <- none at all")


# A reusable named shape, built the same way TimeseriesIndex is: subclass,
# set FIXED_AXES, pass it to axes=. Every LineIndex lives on x -- and, as a
# formula, is VERIFIED to live on x.
class LineIndex(Index):
    FIXED_AXES: ClassVar[tuple[Axis, ...]] = (x,)

    def __init__(self, name: str, value: np.ndarray | graph.Node | None = None) -> None:
        super().__init__(name, value, axes=self.FIXED_AXES)


print(f"  LineIndex('u', array)             -> {LineIndex('u', U_VAL).node.output_axes}")
try:
    LineIndex("total", u_on_x.sum(axis=x))
except ValueError:
    print("  LineIndex('total', u.sum(axis=x)) -> ValueError: the formula no longer carries x")

# --- the type travels with the axis through formulas ------------------------
print("\n=== the axis (and its type) travels through formulas ===")
for label, formula in (
    ("u * 2 + 1", u_on_x * 2 + 1),
    ("u.diff(axis=x)", u_on_x.diff(axis=x)),
    ("u.gradient(axis=x)", u_on_x.gradient(axis=x)),
    ("u.sum(axis=x)", u_on_x.sum(axis=x)),
):
    print(f"  {label:20s} -> {Index('out', formula).node.output_axes}")

# --- trap: a same-name UNTYPED axis can shadow the type ---------------------
# x_untyped == x, so the two operands share ONE axis -- but the result keeps
# whichever copy it met first, and only one of them carries the SpaceType.
x_untyped = Axis("x", DOMAIN)
ones = Index("ones", np.ones(4), axes=(x_untyped,))
print("\n=== trap: a same-name untyped axis can shadow the type ===")
for label, formula in (("u * ones", u_on_x * ones), ("ones * u", ones * u_on_x)):
    prod = Index("prod", formula)
    try:
        prod.gradient()  # default: the sole DOMAIN axis the result carries
        verdict = "gradient() ok"
    except ValueError:
        verdict = "gradient() -> ValueError: no SpaceType"
    print(f"  {label:9s} carries {prod.node.output_axes[0]!r:60s} {verdict}")


# --- carrying the axes into an evaluated model ------------------------------
@define("rod")
class Rod(Model):
    @inputs
    class Inputs:
        u: LineIndex   # DOMAIN (x,) -- checked against LineIndex.FIXED_AXES
        k: Index       # PARAMETER: a conductivity, swept externally

    @outputs
    class Outputs:
        flux: LineIndex        # (x,)  -k du/dx
        diffusion: LineIndex   # (x,)   k d2u/dx2
        net: Index             # ()     flux summed over the whole rod

    def compute(self, inputs: Inputs) -> Outputs:
        flux = LineIndex("flux", -inputs.k * inputs.u.gradient(axis=x))
        diffusion = LineIndex("diffusion", inputs.k * inputs.u.laplacian(axes=(x,)))
        net = Index("net", flux.sum(axis=x), axes=())
        return Rod.Outputs(flux=flux, diffusion=diffusion, net=net)


k = Index("k")
rod = Rod(inputs=Rod.Inputs(u=LineIndex("u", U_VAL), k=k))
rod_scenario = Scenario(rod, parameter_axes=[k])
rod_result = Evaluation(rod_scenario).evaluate(
    ensemble=DistributionEnsemble(rod_scenario, size=1),
    parameters={k: np.array([1.0, 2.0])},
)

print("\n=== reading the axes back off the answers ===")
print("  rod_result.layout -- k and _ensemble were added by the library:")
for rod_ax, rod_size in rod_result.layout.entries:
    print(f"    {rod_ax!r:62s} size={rod_size}")

flux_layout = rod_result.layout_of(rod.outputs.flux)
print(f"  layout_of(flux).axes : {flux_layout.axes}")
print(f"  by role              : DOMAIN={[a.name for a, _ in flux_layout.axes_by_role(DOMAIN)]} "
      f"PARAMETER={[a.name for a, _ in flux_layout.axes_by_role(PARAMETER)]}")

found = flux_layout.find_axis("x")
# find_axis returns a plain Axis | None; narrow it to read the type's fields.
assert isinstance(found, DomainAxis) and isinstance(found.type, SpaceType)
print(f"  find_axis('x') is x  : {found is x}")
print(f"    size={flux_layout.size_of(found)}  spacing={found.type.spacing}  "
      f"boundary={found.type.boundary!r}")

print("\n=== and the numbers were computed with those properties ===")
for out_name in ("flux", "diffusion", "net"):
    out_lab = rod_result.labeled(getattr(rod.outputs, out_name))
    for i, k_val in enumerate((1.0, 2.0)):
        print(f"  {out_name:9s} k={k_val:g}  dims={str(out_lab.dims):12s} {out_lab.sel(k=i).values}")


# =============================================================================
# PART 2 -- axis roles together, and how axes are tracked across outputs
# =============================================================================
print("\n" + "=" * 77)
print(" PART 2 -- axis roles together, and how axes are tracked across outputs")
print("=" * 77)


@define("extended")
class ExtendedModel(Model):
    @inputs
    class Inputs:
        x: DistributionIndex   # ENSEMBLE: stochastic noise
        param: Index           # PARAMETER: swept externally
        M: Index               # DOMAIN (row, col)
        T: TimeseriesIndex     # DOMAIN (time,)

    @outputs
    class Outputs:
        matrix: Index      # (row, col)
        series: Index      # (time,)
        mixed: Index       # (row, col, time) -- outer product of both
        row_total: Index   # (col,) after summing over row
        scalar: Index      # () -- fully reduced

    def compute(self, inputs: Inputs) -> Outputs:
        matrix = Index("matrix", inputs.x * inputs.M * inputs.param, axes=(row, col))
        series = Index("series", inputs.x * inputs.T)
        # axes= is required here: an outer product of disjoint operands is
        # flagged as probably-accidental unless stated explicitly (PART 3).
        mixed = Index("mixed", matrix * series, axes=(row, col, TIME_AXIS))
        row_total = Index("row_total", matrix.sum(axis=row), axes=(col,))
        scalar = Index("scalar", row_total.sum(axis=col), axes=())
        return ExtendedModel.Outputs(
            matrix=matrix, series=series, mixed=mixed,
            row_total=row_total, scalar=scalar,
        )


param = Index("param")
model = ExtendedModel(inputs=ExtendedModel.Inputs(
    x=DistributionIndex("x", stats.norm, {"loc": 1.0, "scale": 0.5}),
    param=param,
    M=Index("M", np.array([[1.0, 2.0], [3.0, 4.0]]), axes=(row, col)),
    T=TimeseriesIndex("T", np.array([1.0, 10.0, 100.0])),
))

scenario = Scenario(model, parameter_axes=[param])
ensemble = DistributionEnsemble(scenario, size=5000, rng=np.random.default_rng(0))
result = Evaluation(scenario).evaluate(
    ensemble=ensemble,
    parameters={param: np.array([1.0, 2.0])},
)

print("\n=== result.layout — UNION over the whole evaluation ===")
for ax, size in result.layout.entries:
    print(f"  {ax.name:10s} {ax.role:10s} size={size}")
print("  full_shape:", result.layout.full_shape, "<- no single output has this shape")
print(f"  n_params={result.layout.n_params} "
      f"n_ensemble={result.layout.n_ensemble} n_domain={result.layout.n_domain}")

print("\n=== layout_of(each output) — the PER-OUTPUT structure ===")
for name in ("matrix", "series", "mixed", "row_total", "scalar"):
    idx = getattr(model.outputs, name)
    lo = result.layout_of(idx)
    arr = result.expected_value(idx)
    roles = [f"{ax.name}:{ax.role[0]}" for ax in lo.axes]
    print(f"  {name:10s} dims={str([a.name for a in lo.axes]):28s} "
          f"shape={str(arr.shape):12s} roles={roles}")

print("\n=== name-based access, immune to axis order ===")
lab = result.labeled(model.outputs.matrix)
print("  matrix dims:", lab.dims, " (declared order was row, col)")
print("  param=0, row=0:", lab.sel(param=0, row=0).values)
print("  param=1, row=1, col=0:", lab.sel(param=1, row=1, col=0).values)

lab_mixed = result.labeled(model.outputs.mixed)
print("\n  mixed dims:", lab_mixed.dims)
print("  mixed at param=0, time=2:\n", lab_mixed.sel(param=0, time=2).values)

print("\n=== why this order: DOMAIN axes are sorted by NAME ===")
mixed_layout = result.layout_of(model.outputs.mixed)
mixed_domain = [ax.name for ax, _ in mixed_layout.axes_by_role(DOMAIN)]
print(f"  mixed declared (row, col, time), DOMAIN axes returned as {mixed_domain}")
print(f"  alphabetical? {mixed_domain == sorted(mixed_domain)}")

# The trap: reading "the second dimension is time" off a habit, not the layout.
mixed_ev = result.expected_value(model.outputs.mixed)
print(f"  mixed_ev.shape = {mixed_ev.shape}; mixed_ev[0, 1] is col=1, "
      f"shape {mixed_ev[0, 1].shape} -- NOT time=1")

# When an API needs positions, look the position up instead of assuming it.
time_pos = mixed_layout.position_of(TIME_AXIS)
by_time = np.moveaxis(mixed_ev, time_pos, 1)       # (param, time, col, row)
print(f"  time is at position {time_pos}: np.moveaxis(mixed_ev, {time_pos}, 1).shape = {by_time.shape}")
print(f"  by_time[0, 1] == lab_mixed.sel(param=0, time=1)? "
      f"{np.allclose(by_time[0, 1], lab_mixed.sel(param=0, time=1).values)}")

print("\n=== reorder to your own convention, derived not assumed ===")
want = ("param", "row", "col")
perm = [lab.dims.index(n) for n in want]
print(f"  {lab.dims} -> {want} via transpose{tuple(perm)}")
print(np.transpose(lab.values, perm))


# =============================================================================
# PART 3 -- multiplying two arrays: when does an axis EMERGE?
# =============================================================================
print("\n" + "=" * 77)
print(" PART 3 -- multiplying two arrays: when does an axis EMERGE?")
print("=" * 77)

time_axis = TIME_AXIS

# Values chosen so every operation below is checkable by eye.
A_VAL = np.array([[1.0, 2.0], [3.0, 4.0]])
VCOL_VAL = np.array([10.0, 20.0])
VTIME_VAL = np.array([1.0, 10.0, 100.0])

A = Index("A", A_VAL, axes=(row, col))                 # [[1, 2], [3, 4]]
Vcol = Index("Vcol", VCOL_VAL, axes=(col,))            # [10, 20]
Vtime = Index("Vtime", VTIME_VAL, axes=(time_axis,))   # [1, 10, 100]

print("\n  the operands:")
print(f"    A    (row,col) =\n{A_VAL}")
print(f"    Vcol (col,)    = {VCOL_VAL}      <- one factor per COLUMN")
print(f"    Vtime(time,)   = {VTIME_VAL}  <- a 3-point series, no axis in common with A")


# The warning is raised when the Index is BUILT (that is when its axes are
# inferred), so catch it around the Index(...) call.
print("\n=== when multiplying two arrays, which cases warn? ===")
for label, formula in (
    ("A(row,col) * A(row,col)", A * A),
    ("A(row,col) * Vcol(col)", A * Vcol),
    ("A(row,col) * 2.0", A * 2.0),
    ("A(row,col) * Vtime(time)", A * Vtime),
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        product = Index("product", formula)
    product_axes = tuple(a.name for a in product.node.output_axes)
    warned = any(issubclass(c.category, AxesInferenceWarning) for c in caught)
    print(f"  {label:32s} -> {str(product_axes):26s} {'WARNS' if warned else 'quiet'}")

# Declaring axes= says "I meant this": quiet, and verified against the formula.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    declared = Index("declared", A * Vtime, axes=(row, col, time_axis))
print(f"  {'  ...same, axes= declared':32s} -> "
      f"{str(tuple(a.name for a in declared.node.output_axes)):26s} "
      f"{'WARNS' if caught else 'quiet'}")

# A WRONG declaration is not an override: it is rejected.
print("    ...declared WRONGLY:")
try:
    Index("wrong", A * Vtime, axes=(row, col))
except ValueError as exc:
    print(f"    ValueError: {exc}")


# =============================================================================
# PART 4 -- the same products, EVALUATED: shared, outer, dot
# =============================================================================
print("\n" + "=" * 77)
print(" PART 4 -- the same products, EVALUATED: shared, outer, dot")
print("=" * 77)


@define("products")
class Products(Model):
    @inputs
    class Inputs:
        A: Index
        Vcol: Index
        Vtime: Index

    @outputs
    class Outputs:
        shared: Index   # (row,col)      -- broadcast along the SHARED col axis
        outer: Index    # (row,col,time) -- DISJOINT axes, so time emerges
        dotted: Index   # (row,)         -- broadcast, then contract col away

    def compute(self, inputs: Inputs) -> Outputs:
        shared = Index("shared", inputs.A * inputs.Vcol)
        outer = Index("outer", inputs.A * inputs.Vtime,
                      axes=(row, col, time_axis))
        # A dot product is the shared-axis product followed by summing it away.
        dotted = Index("dotted", shared.sum(axis=col), axes=(row,))
        return Products.Outputs(shared=shared, outer=outer, dotted=dotted)


pmodel = Products(inputs=Products.Inputs(A=A, Vcol=Vcol, Vtime=Vtime))
pscenario = Scenario(pmodel)
presult = Evaluation(pscenario).evaluate(
    ensemble=DistributionEnsemble(pscenario, size=1),
)


# DOMAIN axes come back sorted by name (col before row), so derive
# the transpose from the names, exactly as in PART 2.
shared_lab = presult.labeled(pmodel.outputs.shared)
outer_lab = presult.labeled(pmodel.outputs.outer)
dotted_lab = presult.labeled(pmodel.outputs.dotted)

shared_vals = np.transpose(shared_lab.values,
                           [shared_lab.dims.index(n) for n in ("row", "col")])
outer_vals = np.transpose(outer_lab.values,
                          [outer_lab.dims.index(n) for n in ("row", "col", "time")])

print("\n=== SHARED axis: A * Vcol -> (row, col) ===")
print(shared_vals)

print("\n=== DISJOINT axes: A * Vtime -> (row, col, time) ===")
for k, factor in enumerate(VTIME_VAL):
    print(f"    time={k} (x{factor:g}):\n{outer_vals[:, :, k]}")

print("\n=== DOT product: (A * Vcol) then sum over col -> (row,) ===")
print(f"  {dotted_lab.values}")
print(f"  numpy cross-check: A @ Vcol = {A_VAL @ VCOL_VAL}")


# =============================================================================
# PART 5 -- BROADCAST: adding an axis nothing else in the formula provides
# =============================================================================
print("\n" + "=" * 77)
print(" PART 5 -- BROADCAST: adding an axis nothing else in the formula provides")
print("=" * 77)

bcast_row = DomainAxis("brow", type=SequenceType())
bcast_col = DomainAxis("bcol", type=SequenceType())
VEC = np.array([1.0, 2.0, 3.0])          # (brow,)
MAT = np.zeros((3, 4))                   # (brow, bcol)

vec_index = Index("vec_index", VEC, axes=(bcast_row,))
print(f"\n  vec_index carries {tuple(a.name for a in vec_index.node.output_axes)}")

# Declaring the extra axis without broadcasting is rejected outright.
try:
    Index("rejected", vec_index, axes=(bcast_row, bcast_col))
except ValueError as exc:
    print("  axes=(brow, bcol) alone ->")
    print(f"    ValueError: {exc}")

broadcast_ok = Index("broadcast_ok", vec_index.broadcast(bcast_col),
                     axes=(bcast_row, bcast_col))
print("  vec_index.broadcast(bcol) -> axes "
      f"{tuple(a.name for a in broadcast_ok.node.output_axes)}")


@define("broadcasting")
class Broadcasting(Model):
    @inputs
    class Inputs:
        vec: Index          # (brow,)
        mat: Index          # (brow, bcol)

    @outputs
    class Outputs:
        spread: Index       # (brow, bcol) -- but size 1 along bcol
        combined: Index     # (brow, bcol) -- bcol gets its real extent here

    def compute(self, inputs: Inputs) -> Outputs:
        # broadcast alone declares the axis with an IMPLICIT size-1 extent.
        spread = Index("spread", inputs.vec.broadcast(bcast_col),
                       axes=(bcast_row, bcast_col))
        # An operand with a real extent along bcol expands the size-1 dimension.
        combined = Index("combined", inputs.vec.broadcast(bcast_col) + inputs.mat,
                         axes=(bcast_row, bcast_col))
        return Broadcasting.Outputs(spread=spread, combined=combined)


bcast_model = Broadcasting(inputs=Broadcasting.Inputs(
    vec=Index("vec", VEC, axes=(bcast_row,)),
    mat=Index("mat", MAT, axes=(bcast_row, bcast_col)),
))
bcast_scenario = Scenario(bcast_model)
bcast_res = Evaluation(bcast_scenario).evaluate(
    ensemble=DistributionEnsemble(bcast_scenario, size=1),
)

spread_raw = np.squeeze(np.asarray(bcast_res[bcast_model.outputs.spread]))
combined_raw = np.squeeze(np.asarray(bcast_res[bcast_model.outputs.combined]))
print(f"\n  vec = {VEC}, mat = zeros(3, 4)")
print(f"  spread   raw shape "
      f"{np.asarray(bcast_res[bcast_model.outputs.spread]).shape}"
      f"  -> values {spread_raw}")
print(f"  combined raw shape "
      f"{np.asarray(bcast_res[bcast_model.outputs.combined]).shape}")
print(combined_raw)
