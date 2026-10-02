<!-- SPDX-License-Identifier: Apache-2.0 -->

# Combining axes: a timeseries and a matrix, and how axes are tracked

> Script: [`combine_axis.py`](combine_axis.py) — run it with
> `uv run python examples/detailed/combine_axis/combine_axis.py`.
> Each section below matches a `PART N` banner in the script.

This example covers how the engine tracks axes when indexes with different
shapes are combined:

| Part | Topic                                                                            |
|------|----------------------------------------------------------------------------------|
| 1    | The three axis roles, `result.layout` vs `layout_of()`, and name-based access    |
| 2    | Multiplying two arrays: which cases make an axis *emerge*, and why that warns    |
| 3    | The same products evaluated: shared axis, outer product, dot product             |
| 4    | `DomainAxis`: typing an axis decides which operators it supports                 |
| 5    | `.broadcast()`: giving an index an axis its own formula never references         |

---

## PART 1 — axis roles, and how axes are tracked across outputs

We define four inputs that between them cover the three axis **roles** the
library distinguishes:

| input   | role        | what it is                                     |
|---------|-------------|------------------------------------------------|
| `x`     | `ENSEMBLE`  | stochastic noise, drawn 5000 times             |
| `param` | `PARAMETER` | not random but swept externally, over `[1, 2]` |
| `M`     | `DOMAIN`    | a fixed `(row, col)` matrix                    |
| `T`     | `DOMAIN`    | a fixed `(time,)` timeseries of 3 points       |

The domain axes for the matrix are declared as `DomainAxis` with a
`SequenceType` — see [PART 4](#part-4--domainaxis-typing-an-axis-decides-which-operators-it-supports)
for why that is preferred over a bare `Axis(name, DOMAIN)`:

```python
row = DomainAxis("row", type=SequenceType())
col = DomainAxis("col", type=SequenceType())
```

The model computes five outputs that combine the inputs in different ways:

```python
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
```

The four inputs, one per row of the table above, and the evaluation: `param` is
swept over `[1, 2]`, and `x` is sampled 5000 times.

```python
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
```

Two details in `mixed`:

- Declaring `axes=` is **required** there. An outer product of disjoint operands
  is flagged as probably-accidental unless stated explicitly (PART 2 explains
  the rule).
- `TIME_AXIS` is the library's own `DomainAxis("time", type=TimeType())`.
  Identity is `(name, role)` only, so it matches the `"time"` axis that
  `TimeseriesIndex` introduced — the type never affects matching.

### `result.layout` is the union; `layout_of()` is the truth per output

No output carries every axis. `result.layout` reports the **union** over the
whole evaluation:

```text
  param      PARAMETER  size=2
  _ensemble  ENSEMBLE   size=5000
  col        DOMAIN     size=2
  row        DOMAIN     size=2
  time       DOMAIN     size=3
  full_shape: (2, 5000, 2, 2, 3) <- no single output has this shape
```

`layout_of(output)` reports what each output really has:

```python
lo = result.layout_of(idx)
arr = result.expected_value(idx)
```

```text
>>> result.layout_of(model.outputs.series)
AxisLayout(Axis('param', role='PARAMETER'): 2, DomainAxis('time', type=TimeType()): 3)
```

For every output (`dims` and `shape` are of `expected_value()`, so the ensemble
axis is gone; `roles` are `P`arameter / `D`omain):

```text
  matrix     dims=['param', 'col', 'row']      shape=(2, 2, 2)    roles=['param:P', 'col:D', 'row:D']
  series     dims=['param', 'time']            shape=(1, 3)       roles=['param:P', 'time:D']
  mixed      dims=['param', 'col', 'row', 'time'] shape=(2, 2, 2, 3) roles=['param:P', 'col:D', 'row:D', 'time:D']
  row_total  dims=['param', 'col']             shape=(2, 2)       roles=['param:P', 'col:D']
  scalar     dims=['param']                    shape=(2,)         roles=['param:P']
```

Note `series`: it never touches `param`, so its `param` axis has **size 1**
rather than 2. The library keeps the axis but does not broadcast the work.

### Name-based access, immune to axis order

The engine returned `matrix` as `(param, col, row)` even though we declared
`(row, col)`. `labeled()` lets you select by name, so the order doesn't matter:

```python
lab = result.labeled(model.outputs.matrix)
```

```text
>>> lab
LabeledArray(dims=('param', 'col', 'row'), shape=(2, 2, 2))
>>> lab.sel(param=0, row=0).values
array([0.99773388, 1.99546775])
>>> lab.sel(param=1, row=1, col=0).values
np.float64(5.986403257135033)
```

The script prints these, plus the same for `mixed`:

```text
  matrix dims: ('param', 'col', 'row')  (declared order was row, col)
  param=0, row=0: [0.99773388 1.99546775]
  param=1, row=1, col=0: 5.986403257135033

  mixed dims: ('param', 'col', 'row', 'time')
  mixed at param=0, time=2:
 [[124.31754449 372.95263348]
 [248.63508899 497.27017797]]
```

These are expected values, so the noise `x` (mean 1.0) has been averaged out
and the numbers sit close to the deterministic ones. `matrix = x · M · param`:

- `param=0` (value 1), `row=0` → the first row of `M`, `[1, 2]` → `≈ [1.0, 2.0]`.
- `param=1` (value 2), `row=1, col=0` → `2 · M[1,0] = 2 · 3` → `≈ 6.0`.

Note that `sel()` takes **positions** along each axis (`param=0` is the first
parameter value), but by **name**, so it works whatever order the engine
chose. In the `mixed` slice the rows printed are `col` and the columns are
`row` — exactly the trap that selecting by name avoids. (The values there are
around `x² · M · 100`, not `M · 100`, because `x` appears in both `matrix` and
`series`, and `E[x²] = 1 + 0.5² = 1.25`.)

When you need your own convention, **derive** the transpose from the names
rather than assuming an order:

```python
want = ("param", "row", "col")
perm = [lab.dims.index(n) for n in want]
print(np.transpose(lab.values, perm))
```

```text
  ('param', 'col', 'row') -> ('param', 'row', 'col') via transpose(0, 2, 1)
[[[0.99773388 1.99546775]
  [2.99320163 3.9909355 ]]

 [[1.99546775 3.9909355 ]
  [5.98640326 7.98187101]]]
```

Now each `param` slice reads like `M` as declared (`[[1, 2], [3, 4]]`), scaled
by 1 and by 2 respectively.

---

## PART 2 — multiplying two arrays: when does an axis emerge?

First, the thing to unlearn: the engine **never** picks between "outer product"
and "dot product". `*` is *always* elementwise multiplication with broadcasting
over the union of the operands' axes — exactly like NumPy, except that axes are
matched by **identity** (the `Axis` object) rather than by position.

So the result axes are always `union(left, right)`. The only question is whether
that union is bigger than either operand started with:

| expression                      | result           | union vs operands          |
|---------------------------------|------------------|----------------------------|
| `A:(row,col) * B:(row,col)`     | `(row,col)`      | equals both operands       |
| `A:(row,col) * B:(col,)`        | `(row,col)`      | equals left operand        |
| `A:(row,col) * 2.0`             | `(row,col)`      | equals left operand        |
| `A:(row,col) * B:(time,)`       | `(row,col,time)` | **larger than BOTH** ⚠     |

Only the last case invents an axis that neither operand had. That is an
**emergent outer product**, and it is the one the library warns about, because
it is far more often a modelling slip (multiplying two things that were never
meant to meet) than a deliberate outer product. The rule is exactly: *warn if
the result axes equal no operand's own axes.*

The warning is **advisory**, not a refusal: the value is computed either way.
Passing `axes=` silences it by saying "yes, I meant this" — and is then
**verified** against the formula, so it cannot be used to fake a shape.

The operands are chosen so every result is checkable by eye:

```python
time_axis = TIME_AXIS

A_VAL = np.array([[1.0, 2.0], [3.0, 4.0]])
VCOL_VAL = np.array([10.0, 20.0])
VTIME_VAL = np.array([1.0, 10.0, 100.0])

A = Index("A", A_VAL, axes=(row, col))                 # [[1, 2], [3, 4]]
Vcol = Index("Vcol", VCOL_VAL, axes=(col,))            # [10, 20]
Vtime = Index("Vtime", VTIME_VAL, axes=(time_axis,))   # [1, 10, 100]
```

The warning is raised when the `Index` is **built**, which is when its axes
are inferred, so it is caught around the `Index(...)` call. The inferred axes
are on `.node.output_axes`:

```python
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
```

Then the same disjoint product with `axes=` declared, correctly and wrongly:

```python
    declared = Index("declared", A * Vtime, axes=(row, col, time_axis))
```

```python
try:
    Index("wrong", A * Vtime, axes=(row, col))
except ValueError as exc:
    print(f"    ValueError: {exc}")
```

```text
  A(row,col) * A(row,col)          -> ('row', 'col')             quiet
  A(row,col) * Vcol(col)           -> ('row', 'col')             quiet
  A(row,col) * 2.0                 -> ('row', 'col')             quiet
  A(row,col) * Vtime(time)         -> ('row', 'col', 'time')     WARNS
    ...same, axes= declared        -> ('row', 'col', 'time')     quiet
    ...declared WRONGLY:
    ValueError: Index 'wrong': declared axes (DomainAxis('row', type=SequenceType()), DomainAxis('col', type=SequenceType())) do not match the actual output_axes (DomainAxis('row', type=SequenceType()), DomainAxis('col', type=SequenceType()), DomainAxis('time', type=TimeType())) (compared as sets, order is not significant). Declaring axes verifies the shape, it cannot relabel it — fix whichever of the two is wrong.
```

Only the disjoint case warns, because only there is the result broader than
both operands. Note the last row: declaring `axes=` is **not an override**. It
is verified, so a wrong declaration is a hard `ValueError`. You cannot use it to
force a dot-product shape out of an outer product.

---

## PART 3 — the same products, evaluated

Building an `Index` only records a formula; to get values we evaluate a model.
The `Products` model computes all three at once:

```python
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
```

The engine is free to return axes in any order, and here it gives `col` before
`row`:

```python
shared_lab = presult.labeled(pmodel.outputs.shared)
outer_lab = presult.labeled(pmodel.outputs.outer)
dotted_lab = presult.labeled(pmodel.outputs.dotted)
```

```text
>>> shared_lab
LabeledArray(dims=('col', 'row'), shape=(2, 2))
>>> shared_lab.values                    # rows are col here, not row
array([[10., 30.],
       [40., 80.]])
>>> outer_lab
LabeledArray(dims=('col', 'row', 'time'), shape=(2, 2, 3))
```

So, as in PART 1, the transpose is derived from the names, otherwise the
printed matrices would silently come out transposed:

```python
shared_vals = np.transpose(shared_lab.values,
                           [shared_lab.dims.index(n) for n in ("row", "col")])
outer_vals = np.transpose(outer_lab.values,
                          [outer_lab.dims.index(n) for n in ("row", "col", "time")])
```

**Shared axis** — `A * Vcol -> (row, col)`. Nothing new appears; each *column*
of `A` is scaled by its own factor (col0 ×10, col1 ×20):

```text
[[10. 40.]
 [30. 80.]]
```

**Disjoint axes** — `A * Vtime -> (row, col, time)`. `time` emerged (the case
that warns): the whole matrix `A` is reproduced once per time point (×1, ×10,
×100):

```text
    time=0 (x1):
[[1. 2.]
 [3. 4.]]
    time=1 (x10):
[[10. 20.]
 [30. 40.]]
    time=2 (x100):
[[100. 200.]
 [300. 400.]]
```

Nothing was summed — every combination of `(row, col)` and `time` is
kept, which is why the array grew from 2×2 to 2×2×3.

**Dot product** — `(A * Vcol)` then `.sum(axis=col) -> (row,)`. A dot product
is not a separate operator: it is the shared-axis product followed by summing
that axis away, with the contracted axis named explicitly:

```text
  [ 50. 110.]
  numpy cross-check: A @ Vcol = [ 50. 110.]
```

By hand: row0 = 1·10 + 2·20 = 50; row1 = 3·10 + 4·20 = 110.

| product       | operands share an axis? | what happens                                                          |
|---------------|-------------------------|-----------------------------------------------------------------------|
| outer product | no                      | axes accumulate, result grows (2×2 and 3 → 2×2×3). Nothing is added.  |
| dot product   | yes                     | multiply along it, then sum it away; result shrinks (2×2 and 2 → 2).  |

Both start from the same `*`. What differs is whether the axes overlap, and
whether you follow the product with a reduction.

---

## PART 4 — `DomainAxis`: typing an axis decides which operators it supports

A bare `Axis(name, DOMAIN)` says "this is a dimension". A `DomainAxis` also says
**what kind** of dimension, by carrying a `DomainType`. That type is not
decoration: operators dispatch on it, so it decides which methods the axis
supports. Each level of the lattice adds vocabulary to the one before:

| type           | meaning                                              | adds                                  |
|----------------|------------------------------------------------------|---------------------------------------|
| `SetType`      | unordered labels (regions, categories)               | reductions and selection only         |
| `SequenceType` | ordered 1-D positions                                | `shift` / `roll` / `diff` / `cumulative` |
| `TimeType`     | `SequenceType` specialised for calendar time         | —                                     |
| `SpaceType`    | `SequenceType` plus a **metric**: spacing + boundary | `gradient` / `laplacian`              |
| `MeshType`     | irregular cells with explicit adjacency              | graph laplacian                       |

Untyped (a plain `Axis`, or a `DomainAxis` with no type) is treated as
`SequenceType`.

The script probes each declaration against six operators:

```python
probe_axes = {
    "Axis(name, DOMAIN)":       Axis("p1", DOMAIN),
    "DomainAxis (untyped)":     DomainAxis("p2"),
    "DomainAxis SetType":       DomainAxis("p3", type=SetType()),
    "DomainAxis SequenceType":  DomainAxis("p4", type=SequenceType()),
    "DomainAxis TimeType":      DomainAxis("p5", type=TimeType()),
    "DomainAxis SpaceType":     DomainAxis("p6", type=SpaceType(spacing=2.0)),
}
OPS = ("sum", "diff", "cumulative", "shift", "gradient", "laplacian")

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
```

```text
  axis declaration          sum        diff       cumulative shift      gradient   laplacian
  Axis(name, DOMAIN)        ok         ok         ok         ok         --         --
  DomainAxis (untyped)      ok         ok         ok         ok         --         --
  DomainAxis SetType        ok         ok         ok         ok         --         --
  DomainAxis SequenceType   ok         ok         ok         ok         --         --
  DomainAxis TimeType       ok         ok         ok         ok         --         --
  DomainAxis SpaceType      ok         ok         ok         ok         ok         ok
```

Only `gradient`/`laplacian` are actually gated today: they need a **metric**
(how far apart two points are), which only `SpaceType` carries. The
ordered-domain operators are not yet enforced against `SetType` — so treat the
type as documentation the engine will increasingly rely on, not as a full
guarantee. What the `SpaceType` operators actually compute has its own example:
[differential_operators](../differential_operators/Differential_Operators_Tutorial.md).

### Typing never breaks matching

```python
plain = Axis("time", DOMAIN)
```

```text
>>> plain == TIME_AXIS
True
>>> hash(plain) == hash(TIME_AXIS)
True
>>> TIME_AXIS
DomainAxis('time', type=TimeType())
```

Identity is `(name, role)` only — the type is excluded on purpose. So typing an
axis is **additive**: existing lookups, unions and saved results keep matching,
and you can add types to an existing model without rewiring anything. The
library's own `TIME_AXIS` is itself a typed `DomainAxis`, which is the pattern
to copy: name the axis once, type it, and import it everywhere.

---

## PART 5 — `.broadcast()`: adding an axis nothing else provides

`axes=` only **verifies** a formula's inferred axes — it cannot declare or
relabel them. That leaves a gap: how do you get an index that carries `(brow,)`
to also carry `(bcol,)`, when nothing in the formula supplies `bcol`?

Not by declaration — the node genuinely does not reference `bcol`:

```python
bcast_row = DomainAxis("brow", type=SequenceType())
bcast_col = DomainAxis("bcol", type=SequenceType())
VEC = np.array([1.0, 2.0, 3.0])          # (brow,)
MAT = np.zeros((3, 4))                   # (brow, bcol)

vec_index = Index("vec_index", VEC, axes=(bcast_row,))
```

```python
try:
    Index("rejected", vec_index, axes=(bcast_row, bcast_col))
```

```text
  axes=(brow, bcol) alone ->
    ValueError: Index 'rejected': declared axes (DomainAxis('brow', type=SequenceType()), DomainAxis('bcol', type=SequenceType())) do not match the actual output_axes (DomainAxis('brow', type=SequenceType()),) (compared as sets, order is not significant). Declaring axes verifies the shape, it cannot relabel it — fix whichever of the two is wrong.
```

`.broadcast(bcol)` fixes that by making the node itself reference the axis,
with an implicit size-1 extent:

```python
broadcast_ok = Index("broadcast_ok", vec_index.broadcast(bcast_col),
                     axes=(bcast_row, bcast_col))
```

```text
  vec_index.broadcast(bcol) -> axes ('brow', 'bcol')
```

What this is **not** for: combining with an operand that *already* carries the
axis needs no broadcast at all, because a formula's axes are the union of its
operands' — `ind_x + ind_xy` infers `(x, y)` on its own.

The `Broadcasting` model shows both situations:

```python
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
```

The raw arrays carry a leading size-1 ensemble axis (nothing is uncertain):

```text
>>> np.asarray(bcast_res[bcast_model.outputs.spread]).shape       # (ensemble, bcol, brow)
(1, 1, 3)
>>> np.asarray(bcast_res[bcast_model.outputs.combined]).shape
(1, 4, 3)
>>> bcast_res.labeled(bcast_model.outputs.combined)
LabeledArray(dims=('bcol', 'brow'), shape=(4, 3))
```

```text
  spread   raw shape (1, 1, 3)  -> values [1. 2. 3.]
  combined raw shape (1, 4, 3)
[[1. 2. 3.]
 [1. 2. 3.]
 [1. 2. 3.]
 [1. 2. 3.]]
```

- `spread`: `bcol` is **declared** but still size 1. Broadcast promises the
  axis; it does not materialise data along it.
- `combined`: adding `mat` supplies the real extent, so the size-1 `bcol`
  dimension expands and `vec` repeats across all 4 columns. (The raw array comes
  back as `(bcol, brow)` — another reason to select by name.)

**When you need it:**

- an index must satisfy a declared output shape that includes an axis its own
  formula never touches (a contract, or an outer-product result where one
  operand is genuinely constant along the new axis);
- you want a `(time,)` series to line up against a `(region, time)` field
  without writing a dummy multiplication by ones.

**When you do not:** any formula where another operand already carries the axis
— the union rule handles it, and an extra broadcast is noise.

**The mental model:** broadcast is a *declaration about axes*, not a reshape. It
makes a node reference an axis at size 1; real extent still has to come from
somewhere, exactly as with NumPy broadcasting.
