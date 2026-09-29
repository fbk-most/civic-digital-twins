"""A timeseries and a matrix combined, and how axes are tracked across outputs.

Here we define four inputs that between them cover the three axis ROLES the
library distinguishes:

  * x     ENSEMBLE  -- stochastic noise, drawn 5000 times;
  * param PARAMETER -- not random but swept externally, here over [1.0, 2.0];
  * M     DOMAIN    -- a fixed (row, col) matrix;
  * T     DOMAIN    -- a fixed (time,) timeseries of 3 points.

We then compute five outputs that combine them in different ways: a matrix
(row, col), a series (time,), their outer product (row, col, time), a partial
sum over row leaving (col,), and a full reduction to a scalar. The run is
5000 ensemble samples for each of the 2 parameter values.

The point is that no output carries every axis. result.layout reports the UNION
over the whole evaluation -- full_shape (2, 5000, 2, 2, 3), a shape nothing
actually has -- while layout_of(output) reports what each output really has.
Note in particular that `series` never touches param, so its param axis has
size 1 rather than 2: the library keeps the axis but does not broadcast the
work.

We then show name-based access via labeled(), which is immune to axis order
-- the engine returns matrix as (param, col, row) even though we declared
(row, col) -- and derive a transpose back to our own convention rather than
assuming one.

The final section covers DomainAxis, the preferred way to declare a domain
axis. A bare Axis(name, DOMAIN) says only "this is a dimension"; a DomainAxis
also records what KIND -- unordered set, ordered sequence, calendar time, or a
physical grid with a spacing -- and operators dispatch on that type. We show
which operators each type unlocks and that typing is backward compatible,
because axis identity ignores the type. (What the SpaceType operators actually
compute -- gradient, laplacian and boundary conditions -- has its own file:
see differential_operators.py.)

The file closes on .broadcast(), which gives an index an axis its own formula
never references. That is the one thing axes= cannot do, since axes= verifies
inferred axes rather than declaring them.

The middle sections tackle multiplication directly: which combinations of
operand axes make the library WARN, and how outer products and dot products
are both expressed with the same `*`. The short version is that `*` is always
elementwise broadcasting over the union of the operands' axes -- the engine
never chooses a product type for you -- and the warning fires only when that
union is broader than BOTH operands, i.e. when an axis emerged from nowhere.
"""

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    define, inputs, outputs, Index, Model, DistributionIndex,
    Scenario, DistributionEnsemble, Evaluation, TimeseriesIndex,
)
from civic_digital_twins.dt_model.axes import (
    Axis, DOMAIN, DomainAxis, TIME_AXIS,
    SetType, SequenceType, TimeType, SpaceType,
)

# Prefer DomainAxis over a bare Axis(name, DOMAIN). Both identify a dimension,
# but DomainAxis also records WHAT KIND of dimension it is, which decides what
# operators the axis supports (see the last section). Axis(name, DOMAIN) is the
# untyped form and behaves as SequenceType by default.
#
# row/col index a matrix: they are ordered positions, not calendar time and
# not a physical grid, so SequenceType is the honest description.
row = DomainAxis("row", type=SequenceType())
col = DomainAxis("col", type=SequenceType())


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
        # Declaring axes= here is required: an outer product from disjoint
        # operands is flagged as probably-accidental unless stated explicitly.
        # TIME_AXIS is the library's own DomainAxis("time", type=TimeType()).
        # Identity is (name, role) only, so this matches the "time" axis that
        # TimeseriesIndex introduced -- the type never affects matching.
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

# Notice how series is NOT multiplied by the parameter axis and although it lists param as an axis, it has size 1.

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


print("\n=============================================================================\n",
      "MULTIPLYING TWO ARRAYS: when does an axis EMERGE?\n",
      "=============================================================================")
# First, the thing to unlearn. The engine never picks between "outer product"
# and "dot product". `*` is ALWAYS elementwise multiplication with broadcasting
# over the union of the operands' axes -- exactly like NumPy, except that axes
# are matched by IDENTITY (the Axis object) rather than by position.
#
# So the result axes are always union(left, right). The only question is
# whether that union is bigger than either operand started with:
#
#   A:(row,col) * B:(row,col)  -> (row,col)       union == both operands
#   A:(row,col) * B:(col,)     -> (row,col)       union == left operand
#   A:(row,col) * 2.0          -> (row,col)       union == left operand
#   A:(row,col) * B:(time,)    -> (row,col,time)  union > BOTH operands  <-- !
#
# Only the last case invents an axis that neither operand had. That is an
# "emergent outer product", and it is the one the library warns about, because
# it is far more often a modelling slip (multiplying two things that were never
# meant to meet) than a deliberate outer product. The rule, from the library
# source, is exactly: warn if the result axes equal NO operand's own axes.
#
# The warning is ADVISORY, not a refusal: the value is computed either way.
# Passing axes= silences it by saying "yes, I meant this" -- and is then
# VERIFIED against the formula, so it cannot be used to fake a shape.

import warnings  # noqa: E402  (kept here so this section reads standalone)

from civic_digital_twins.dt_model import AxesInferenceWarning  # noqa: E402

time_axis = TIME_AXIS

# Values chosen so every operation below is checkable by eye: multiplying by
# Vcol scales a column by 10 or 20, and by Vtime scales a slice by 1/10/100.
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
# Shared axes: the result is no broader than the left operand. Nothing emerged.
show("A(row,col) * A(row,col)", lambda: Index("m1", A * A))
# Partial overlap: col is shared, so (row,col) still equals A's own axes.
show("A(row,col) * Vcol(col)", lambda: Index("m2", A * Vcol))
# Scalar broadcast: trivially no new axis.
show("A(row,col) * 2.0", lambda: Index("m3", A * 2.0))
# DISJOINT: time appears from nowhere. Neither operand had (row,col,time).
show("A(row,col) * Vtime(time)", lambda: Index("m4", A * Vtime))
# Same expression, but declared -- the warning is a request for confirmation.
show("  ...same, axes= declared", lambda: Index("m5", A * Vtime, axes=(row, col, time_axis)))
# Declaring cannot RELABEL: axes= is checked against the formula and rejected.
show("  ...declared WRONGLY", lambda: Index("m6", A * Vtime, axes=(row, col)))

print("""
  Reading the table: only the disjoint case warns, because only there is the
  result broader than BOTH operands. Note the last row -- declaring axes= is
  not an override. It is verified, so a wrong declaration is a hard ValueError.
  You cannot use it to force a dot-product shape out of an outer product.""")


# =============================================================================
# The same three cases, but EVALUATED so you can see the numbers.
# =============================================================================
# Building an Index only records a formula; to get values we evaluate a model.
# This tiny model computes all three products at once.
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
        # A dot product is not a separate operator: it is the shared-axis
        # product above, followed by summing that axis away. The axis being
        # contracted is named explicitly rather than implied by position.
        dotted = Index("dotted", shared.sum(axis=col), axes=(row,))
        return Products.Outputs(shared=shared, outer=outer, dotted=dotted)


pmodel = Products(inputs=Products.Inputs(A=A, Vcol=Vcol, Vtime=Vtime))
pscenario = Scenario(pmodel)
presult = Evaluation(pscenario).evaluate(
    ensemble=DistributionEnsemble(pscenario, size=1),
)


def values_of(name, want):
    """Evaluated values for an output, reordered to the axis order *want*.

    The engine is free to return axes in any order (here it gives col before
    row), so we transpose by NAME rather than trusting the order -- otherwise
    the printed matrices would silently come out transposed.
    """
    lab = presult.labeled(getattr(pmodel.outputs, name))
    vals = np.squeeze(lab.values)
    dims = tuple(d for d in lab.dims if d != "_ensemble")
    perm = [dims.index(n) for n in want]
    return np.transpose(vals, perm)


print("\n=== SHARED axis: A * Vcol -> (row, col) ===")
vals = values_of("shared", ("row", "col"))
print("  col is shared, so nothing new appeared. Shown as (row, col):")
print(vals)
print("  each COLUMN of A scaled by its own factor: col0 x10, col1 x20.")

print("\n=== DISJOINT axes: A * Vtime -> (row, col, time) ===")
vals = values_of("outer", ("row", "col", "time"))
print("  time EMERGED; this is the case that warns.")
print("  the whole matrix A, reproduced once per time point (x1, x10, x100):")
for k, factor in enumerate(VTIME_VAL):
    print(f"    time={k} (x{factor:g}):\n{vals[:, :, k]}")
print("  nothing was summed -- every combination of (row,col) and time is kept,")
print("  which is why the array grew from 2x2 to 2x2x3.")

print("\n=== DOT product: (A * Vcol) then sum over col -> (row,) ===")
vals = values_of("dotted", ("row",))
print("  col was contracted away, leaving one number per row:")
print(f"  {vals}")
print(f"  numpy cross-check: A @ Vcol = {A_VAL @ VCOL_VAL}")
print("  by hand: row0 = 1*10 + 2*20 = 50 ; row1 = 3*10 + 4*20 = 110")

print("""
  The contrast, now visible in the numbers:
    outer product -- operands share NO axis, so axes accumulate and the
                     result grows (2x2 and 3 -> 2x2x3). Nothing is added up.
    dot product   -- operands SHARE an axis, you multiply elementwise along
                     it and then sum it away, so the result shrinks (2x2 and
                     2 -> 2). The shared axis disappears into the total.
  Both start from the same `*`. What differs is whether the axes overlap,
  and whether you follow the product with a reduction.""")


# =============================================================================
# WHY DomainAxis INSTEAD OF Axis(name, DOMAIN)?
# =============================================================================
# A bare Axis(name, DOMAIN) says "this is a dimension". A DomainAxis also says
# WHAT KIND of dimension, by carrying a DomainType. That type is not decoration:
# operators dispatch on it, so it decides which methods the axis supports.
#
# The lattice, each level adding vocabulary to the one before:
#
#   SetType       unordered labels (regions, categories).
#                 Reductions and selection only -- "next" is meaningless.
#   SequenceType  ordered 1-D positions. Adds shift / roll / diff / cumulative.
#   TimeType      SequenceType specialised for calendar time.
#   SpaceType     SequenceType plus a METRIC: spacing and a boundary rule.
#                 Adds gradient / laplacian, which need a physical distance.
#   MeshType      irregular cells with explicit adjacency (graph laplacian).
#
# Untyped (a plain Axis, or DomainAxis with no type) is treated as SequenceType.

print("\n=============================================================================")
print(" DomainAxis: typing an axis decides which operators it supports")
print("=============================================================================")

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

print("""
  Only gradient/laplacian are actually gated today: they need a METRIC (how
  far apart two points are), which only SpaceType carries. The ordered-domain
  operators are not yet enforced against SetType -- so treat the type as
  documentation the engine will increasingly rely on, not as a full guarantee.""")


# --- typing never breaks matching -------------------------------------------
# The type is deliberately excluded from equality and hashing, so a typed axis
# and an untyped one with the same name are THE SAME AXIS to the engine. That
# is what lets you add types to an existing model without rewiring anything.
plain = Axis("time", DOMAIN)
print("\n=== adding a type is backward compatible ===")
print(f"  Axis('time', DOMAIN) == TIME_AXIS : {plain == TIME_AXIS}")
print(f"  same hash                         : {hash(plain) == hash(TIME_AXIS)}")
print(f"  TIME_AXIS is really               : {TIME_AXIS!r}")
print("""
  Identity is (name, role) only -- the type is excluded on purpose. So typing
  an axis is additive: existing lookups, unions and saved results keep matching.
  Note the library's own TIME_AXIS is itself a typed DomainAxis, which is the
  pattern to copy: name the axis once, type it, and import it everywhere.""")


# =============================================================================
# BROADCAST: giving an index an axis it does not structurally carry
# =============================================================================
# Earlier we saw that `axes=` only VERIFIES a formula's inferred axes -- it
# cannot declare or relabel them. That leaves a gap: how do you get an index
# that carries (x,) to also carry (y,), when nothing in the formula supplies y?
#
# You cannot, by declaration:
#     Index("standalone", ind_x, axes=(x, y))     -> ValueError
# because the node genuinely does not reference y. `.broadcast(y)` fixes that
# by making the node itself reference the axis, with an implicit size-1 extent:
#     Index("standalone", ind_x.broadcast(y), axes=(x, y))   -> fine
#
# Note what this is NOT for. Combining with an operand that ALREADY carries y
# needs no broadcast at all, because a formula's axes are the union of its
# operands' -- `ind_x + ind_xy` infers (x, y) on its own. Broadcasting matters
# only when nothing else in the formula supplies the axis.
print("\n=============================================================================")
print(" BROADCAST -- adding an axis nothing else in the formula provides")
print("=============================================================================")

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
        # Combined with an operand that carries a real extent along bcol, the
        # size-1 dimension expands the usual numpy way.
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
print("    bcol is DECLARED but still size 1: broadcast promises the axis,")
print("    it does not materialise data along it.")
print(f"  combined raw shape "
      f"{np.asarray(bcast_res[bcast_model.outputs.combined]).shape}")
print(combined_raw)
print("    adding mat supplies the real extent, so the size-1 bcol dimension")
print("    expands and each row of vec repeats across all 4 columns.")

print("""
  When you need it:
    * an index must satisfy a declared output shape that includes an axis its
      own formula never touches (a contract, or an outer-product result where
      one operand is genuinely constant along the new axis);
    * you want a (time,) series to line up against a (region, time) field
      without writing a dummy multiplication by ones.

  When you do NOT need it:
    * any formula where another operand already carries the axis -- the union
      rule handles it, and an extra broadcast is noise.

  The mental model: broadcast is a DECLARATION about axes, not a reshape. It
  makes a node reference an axis at size 1; real extent still has to come from
  somewhere, exactly as with numpy broadcasting.""")
