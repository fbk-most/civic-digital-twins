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
def compute(self, inputs: Inputs) -> Outputs:
    matrix = Index("matrix", inputs.x * inputs.M * inputs.param, axes=(row, col))
    series = Index("series", inputs.x * inputs.T)
    mixed = Index("mixed", matrix * series, axes=(row, col, TIME_AXIS))
    row_total = Index("row_total", matrix.sum(axis=row), axes=(col,))
    scalar = Index("scalar", row_total.sum(axis=col), axes=())
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
  matrix     dims=['param', 'col', 'row']      shape=(2, 2, 2)
  series     dims=['param', 'time']            shape=(1, 3)
  mixed      dims=['param', 'col', 'row', 'time'] shape=(2, 2, 2, 3)
  row_total  dims=['param', 'col']             shape=(2, 2)
  scalar     dims=['param']                    shape=(2,)
```

Note `series`: it never touches `param`, so its `param` axis has **size 1**
rather than 2. The library keeps the axis but does not broadcast the work.

### Name-based access, immune to axis order

The engine returned `matrix` as `(param, col, row)` even though we declared
`(row, col)`. `labeled()` lets you select by name, so the order doesn't matter:

```python
lab = result.labeled(model.outputs.matrix)
lab.sel(param=0, row=0).values
lab.sel(param=1, row=1, col=0).values
```

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
np.transpose(lab.values, perm)
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
A = Index("A", A_VAL, axes=(row, col))                 # [[1, 2], [3, 4]]
Vcol = Index("Vcol", VCOL_VAL, axes=(col,))            # [10, 20]
Vtime = Index("Vtime", VTIME_VAL, axes=(time_axis,))   # [1, 10, 100]
```

The `show()` helper builds an `Index`, catches any `AxesInferenceWarning`, and
reports the inferred axes:

```python
show("A(row,col) * Vtime(time)", lambda: Index("m4", A * Vtime))
show("  ...same, axes= declared", lambda: Index("m5", A * Vtime, axes=(row, col, time_axis)))
show("  ...declared WRONGLY", lambda: Index("m6", A * Vtime, axes=(row, col)))
```

```text
  A(row,col) * A(row,col)          -> ('row', 'col')             quiet
  A(row,col) * Vcol(col)           -> ('row', 'col')             quiet
  A(row,col) * 2.0                 -> ('row', 'col')             quiet
  A(row,col) * Vtime(time)         -> ('row', 'col', 'time')     WARNS
    ...same, axes= declared        -> ('row', 'col', 'time')     quiet
    ...declared WRONGLY            REJECTED: Index 'm6': declared axes ...
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
def compute(self, inputs: Inputs) -> Outputs:
    shared = Index("shared", inputs.A * inputs.Vcol)
    outer = Index("outer", inputs.A * inputs.Vtime,
                  axes=(row, col, time_axis))
    dotted = Index("dotted", shared.sum(axis=col), axes=(row,))
```

The engine is free to return axes in any order (here it gives `col` before
`row`), so `values_of()` transposes **by name** — otherwise the printed matrices
would silently come out transposed.

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
probe = Index("probe", np.array([1.0, 2.0, 4.0, 8.0]), axes=(ax,))
getattr(probe, op)(axis=ax)          # laplacian takes axes=(ax,) instead
```

```text
  axis declaration          sum   diff  cumulative shift gradient laplacian
  Axis(name, DOMAIN)        ok    ok    ok         ok    --       --
  DomainAxis (untyped)      ok    ok    ok         ok    --       --
  DomainAxis SetType        ok    ok    ok         ok    --       --
  DomainAxis SequenceType   ok    ok    ok         ok    --       --
  DomainAxis TimeType       ok    ok    ok         ok    --       --
  DomainAxis SpaceType      ok    ok    ok         ok    ok       ok
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
plain == TIME_AXIS                   # True
hash(plain) == hash(TIME_AXIS)       # True
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
Index("rejected", vec_index, axes=(bcast_row, bcast_col))          # ValueError
```

`.broadcast(bcol)` fixes that by making the node itself reference the axis,
with an implicit size-1 extent:

```python
Index("broadcast_ok", vec_index.broadcast(bcast_col),
      axes=(bcast_row, bcast_col))                                 # fine
```

What this is **not** for: combining with an operand that *already* carries the
axis needs no broadcast at all, because a formula's axes are the union of its
operands' — `ind_x + ind_xy` infers `(x, y)` on its own.

The `Broadcasting` model shows both situations:

```python
spread = Index("spread", inputs.vec.broadcast(bcast_col),
               axes=(bcast_row, bcast_col))
combined = Index("combined", inputs.vec.broadcast(bcast_col) + inputs.mat,
                 axes=(bcast_row, bcast_col))
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
