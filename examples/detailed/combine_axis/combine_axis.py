# SPDX-License-Identifier: Apache-2.0

"""A timeseries and a matrix combined, and how axes are tracked across outputs.

Narrative and explanations: see Combine_Axis_Tutorial.md in this directory.
Each "PART N" banner below matches a section of the same name there.

  PART 1  Axis roles, result.layout vs layout_of(), name-based access.
  PART 2  Multiplying two arrays: which cases make an axis EMERGE (and warn).
  PART 3  The same products evaluated: shared axis, outer product, dot product.
  PART 4  DomainAxis: typing an axis decides which operators it supports.
  PART 5  .broadcast(): adding an axis nothing else in the formula provides.
"""

import warnings

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    define, inputs, outputs, Index, Model, DistributionIndex,
    Scenario, DistributionEnsemble, Evaluation, TimeseriesIndex,
    AxesInferenceWarning,
)
from civic_digital_twins.dt_model.axes import (
    Axis, DOMAIN, DomainAxis, TIME_AXIS,
    SetType, SequenceType, TimeType, SpaceType,
)

# row/col index a matrix: ordered positions, not calendar time and not a
# physical grid, so SequenceType is the honest description (see PART 4).
row = DomainAxis("row", type=SequenceType())
col = DomainAxis("col", type=SequenceType())


# =============================================================================
# PART 1 -- axis roles, and how axes are tracked across outputs
# =============================================================================
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
        # flagged as probably-accidental unless stated explicitly (PART 2).
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

print("=== result.layout — UNION over the whole evaluation ===")
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

print("\n=== reorder to your own convention, derived not assumed ===")
want = ("param", "row", "col")
perm = [lab.dims.index(n) for n in want]
print(f"  {lab.dims} -> {want} via transpose{tuple(perm)}")
print(np.transpose(lab.values, perm))


# =============================================================================
# PART 2 -- multiplying two arrays: when does an axis EMERGE?
# =============================================================================
print("\n" + "=" * 77)
print(" PART 2 -- multiplying two arrays: when does an axis EMERGE?")
print("=" * 77)

time_axis = TIME_AXIS

# Values chosen so every operation below is checkable by eye.
A_VAL = np.array([[1.0, 2.0], [3.0, 4.0]])
VCOL_VAL = np.array([10.0, 20.0])
VTIME_VAL = np.array([1.0, 10.0, 100.0])

A = Index("A", A_VAL, axes=(row, col))              # (row,col)
Vcol = Index("Vcol", VCOL_VAL, axes=(col,))         # (col,)
Vtime = Index("Vtime", VTIME_VAL, axes=(time_axis,))  # (time,)

print("\n  the operands:")
print(f"    A    (row,col) =\n{A_VAL}")
print(f"    Vcol (col,)    = {VCOL_VAL}      <- one factor per COLUMN")
print(f"    Vtime(time,)   = {VTIME_VAL}  <- a 3-point series, no axis in common with A")


def show(label, build):
    """Build an Index, reporting its inferred axes and any warning raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = build()
        except ValueError as exc:
            print(f"  {label:32s} REJECTED: {str(exc).splitlines()[0][:60]}...")
            return
        axes = tuple(a.name for a in out.node.output_axes)
        warned = any(issubclass(c.category, AxesInferenceWarning) for c in caught)
        print(f"  {label:32s} -> {str(axes):26s} {'WARNS' if warned else 'quiet'}")


print("\n=== when multiplying two arrays, which cases warn? ===")
show("A(row,col) * A(row,col)", lambda: Index("m1", A * A))
show("A(row,col) * Vcol(col)", lambda: Index("m2", A * Vcol))
show("A(row,col) * 2.0", lambda: Index("m3", A * 2.0))
show("A(row,col) * Vtime(time)", lambda: Index("m4", A * Vtime))
show("  ...same, axes= declared", lambda: Index("m5", A * Vtime, axes=(row, col, time_axis)))
show("  ...declared WRONGLY", lambda: Index("m6", A * Vtime, axes=(row, col)))


# =============================================================================
# PART 3 -- the same products, EVALUATED: shared, outer, dot
# =============================================================================
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


def values_of(name, want):
    """Evaluated values for an output, transposed by NAME to the order *want*."""
    lab = presult.labeled(getattr(pmodel.outputs, name))
    vals = np.squeeze(lab.values)
    dims = tuple(d for d in lab.dims if d != "_ensemble")
    perm = [dims.index(n) for n in want]
    return np.transpose(vals, perm)


print("\n=== SHARED axis: A * Vcol -> (row, col) ===")
print(values_of("shared", ("row", "col")))

print("\n=== DISJOINT axes: A * Vtime -> (row, col, time) ===")
vals = values_of("outer", ("row", "col", "time"))
for k, factor in enumerate(VTIME_VAL):
    print(f"    time={k} (x{factor:g}):\n{vals[:, :, k]}")

print("\n=== DOT product: (A * Vcol) then sum over col -> (row,) ===")
print(f"  {values_of('dotted', ('row',))}")
print(f"  numpy cross-check: A @ Vcol = {A_VAL @ VCOL_VAL}")


# =============================================================================
# PART 4 -- DomainAxis: typing an axis decides which operators it supports
# =============================================================================
print("\n" + "=" * 77)
print(" PART 4 -- DomainAxis: typing an axis decides which operators it supports")
print("=" * 77)

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
print("\n=== adding a type is backward compatible ===")
print(f"  Axis('time', DOMAIN) == TIME_AXIS : {plain == TIME_AXIS}")
print(f"  same hash                         : {hash(plain) == hash(TIME_AXIS)}")
print(f"  TIME_AXIS is really               : {TIME_AXIS!r}")


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
    print(f"  axes=(brow, bcol) alone -> REJECTED: {str(exc).splitlines()[0][:52]}...")

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
