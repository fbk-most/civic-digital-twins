<!-- SPDX-License-Identifier: Apache-2.0 -->

# Differential operators on a `SpaceType` axis: gradient, laplacian, boundaries

> Script: [`differential_operators.py`](differential_operators.py) — run it with
> `uv run python examples/detailed/differential_operators/differential_operators.py`.
> Each section below matches a `PART N` banner in the script.

An axis declared with `SpaceType` carries two pieces of physical metadata that
nothing else in the library provides:

| field      | meaning                                              |
|------------|------------------------------------------------------|
| `spacing`  | how far apart two neighbouring samples are           |
| `boundary` | what the field does just *outside* the domain        |

Those two unlock `.gradient()` and `.laplacian()`, and they decide the numbers
those operators return. (For where `SpaceType` sits among the other domain
types, see PART 4 of
[combine_axis](../combine_axis/Combine_Axis_Tutorial.md).)

| Part | Topic                                                                                 |
|------|---------------------------------------------------------------------------------------|
| 1    | What `.gradient()` computes — a *directional* derivative along one axis; vs `.diff()` |
| 2    | The 2-D case: `.gradient` is per-axis, `.laplacian` sums several axes                 |
| 3    | Boundary conditions: every policy against both operators                              |
| 4    | A trap: the *order* of `axes=` on an injected array is load-bearing and unchecked     |

Naming is deliberately abstract: `f` is the field, `x` / `y` / `depth` are axes.

---

## PART 1 — what `.gradient(axis=...)` actually computes

"Gradient" in vector calculus is the vector of partial derivatives over *all*
dimensions. That is **not** what this operator is. `.gradient(axis=ax)` takes a
**directional derivative along one named axis** — a single partial derivative,
`d(value)/d(ax)` — and is perfectly well defined in 1-D. It is exactly
`np.gradient(values, spacing, axis=<that axis>)`. For a full multi-dimensional
gradient you call it once per axis (PART 2).

How each point is computed: **every** point uses the central difference
`(next - prev) / (2 * spacing)` — including the two edges, which get their
missing neighbour from a ghost cell filled in by the axis's **boundary
condition** (PART 3). The output therefore has the same length as the input;
nothing is dropped, and nothing uses a different formula.

The data is a temperature profile down a borehole, deliberately non-linear (a
straight line would make every output identical and hide the mechanism):

```python
DEPTH_VALS = np.array([0.0, 10.0, 20.0, 50.0])   # note the jump at the end

grid = DomainAxis("depth", type=SpaceType(spacing=spacing))
...
slope=Index("slope", inputs.T.gradient(axis=grid), axes=(grid,)),
step=Index("step", inputs.T.diff(axis=grid), axes=(grid,)),
```

```text
  spacing=1 m
    .gradient -> [ 0. 10. 20.  0.]   (degrees per METRE)
    .diff     -> [ 0. 10. 10. 30.]   (degrees per SAMPLE, spacing ignored)
  spacing=2 m
    .gradient -> [ 0.  5. 10.  0.]   (degrees per METRE)
    .diff     -> [ 0. 10. 10. 30.]   (degrees per SAMPLE, spacing ignored)
```

Checking the `spacing=1` row by hand. The boundary defaults to `Neumann(0)` —
"the slope just outside is zero":

| i | position | computation                     | value |
|---|----------|---------------------------------|-------|
| 0 | edge     | slope pinned to 0 by Neumann(0) | 0     |
| 1 | interior | (20 − 0) / (2 · 1)              | 10    |
| 2 | interior | (50 − 10) / (2 · 1)             | 20    |
| 3 | edge     | slope pinned to 0 by Neumann(0) | 0     |

So the default deliberately reports a **flat edge** rather than guessing a
one-sided slope. If a real slope at the boundary matters to you, say so:
`Linear()` extends the local trend, `Nearest()` repeats the edge cell, and
`Constant(v)` / `Neumann(v)` state an actual physical condition (PART 3).

**Doubling the spacing halves every value**: the same temperature change spread
over twice the distance is half the rate. That is the whole point of the metric.
`.diff` is unchanged by spacing because it never divides by it — `.diff`
answers "per sample" while `.gradient` answers "per metre".

---

## PART 2 — the 2-D case: gradient is per-axis, laplacian is multi-axis

This is where the difference between the two operators becomes obvious:

| call                      | axes        | result                                                         |
|---------------------------|-------------|----------------------------------------------------------------|
| `.gradient(axis=ax)`      | **one**     | one partial derivative; call it twice for a 2-D gradient       |
| `.laplacian(axes=(...))`  | **several** | second derivatives summed into **one** scalar field            |

Note the plural keyword on `laplacian`.

The field is `f(x, y) = x²`, constant in `y`, because its derivatives are known
exactly and every printed number can be checked:
`df/dx = 2x`, `df/dy = 0`, `∇²f = 2 + 0 = 2`.

```python
y_axis = DomainAxis("y", type=SpaceType(spacing=1.0))
x_axis = DomainAxis("x", type=SpaceType(spacing=1.0))

XS = np.arange(4.0)                                   # x = 0,1,2,3
FIELD = (XS**2)[None, :] * np.ones((3, 1))            # shape (3, 4)

df_dx=Index("df_dx", inputs.F.gradient(axis=x_axis), axes=(y_axis, x_axis)),
df_dy=Index("df_dy", inputs.F.gradient(axis=y_axis), axes=(y_axis, x_axis)),
lap=Index("lap", inputs.F.laplacian(axes=(y_axis, x_axis)), axes=(y_axis, x_axis)),
```

Why `y` **before** `x`: when you attach `axes=` to an injected array, the axes
are zipped **positionally** against that array's shape. `FIELD` has shape
`(3, 4)` — 3 rows, 4 columns — so `axes=(y_axis, x_axis)` declares "first
dimension is `y` with 3 points, second is `x` with 4". The order follows the
array, not taste. Getting it wrong is silent — see PART 4.

The input field (rows are `y`, columns are `x = 0, 1, 2, 3`):

```text
[[0. 1. 4. 9.]
 [0. 1. 4. 9.]
 [0. 1. 4. 9.]]
```

**`.gradient(axis=x)`** — exact answer `2x = [0, 2, 4, 6]`:

```text
[[0. 2. 4. 0.]
 [0. 2. 4. 0.]
 [0. 2. 4. 0.]]
```

Interior points are exact (2 and 4). The **edges** are set by the default
`Neumann(0)` boundary, which pins the slope there to 0 — right by luck at
`x=0`, wrong at `x=3` where the true value is 6.

**`.gradient(axis=y)`** — exact answer 0 everywhere:

```text
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
```

`f` does not vary along `y`, so this component is identically zero. Two calls, two components: *that*
is the full 2-D gradient.

**`.laplacian(axes=(y, x))`** — exact answer 2:

```text
[[  2.   2.   2. -10.]
 [  2.   2.   2. -10.]
 [  2.   2.   2. -10.]]
```

One scalar field, not two: the axes were summed over, not kept. The first three
columns are exactly 2 (2 from `x`, 0 from `y`). The last column is again the
boundary at work: `Neumann(0)` mirrors `f[2] = 4` into the ghost cell past
`x=3`, giving `(4 − 2·9 + 4) / 1 = −10`.

---

## PART 3 — boundary conditions

Both operators keep the output the same length as the input, so both need a
value for the cell just past each edge. They get it the same way: the axis's
`SpaceType` carries a **boundary condition**, the kernel pads one ghost cell on
each side according to it, and then the *same* central-difference stencil runs
everywhere, edges included:

```text
gradient   (f[i+1] - f[i-1]) / (2h)
laplacian  (f[i-1] - 2 f[i] + f[i+1]) / h^2
```

So a boundary condition is really just "the rule for filling in one value past
each edge". The rules split into families:

| family                       | rule          | meaning                                                         |
|------------------------------|---------------|-----------------------------------------------------------------|
| value-fixing (Dirichlet)     | `Constant(v)` | the field just outside is `v` (`Dirichlet(v)` is an alias)      |
| derivative-fixing (Neumann)  | `Neumann(v)`  | the slope just outside is `v` (`Reflect(v)` is an alias). `Neumann(0.0)` is the **default** — a "no flux through the wall" edge |
| structural                   | `Nearest()`   | repeat the outermost cell                                       |
| structural                   | `Wrap()`      | periodic — opposite borders connect                             |
| structural                   | `Linear()`    | extend the local linear trend                                   |

The value-carrying rules let you state an actual physical edge condition (a wall
held at 20 degrees, a known flux) rather than only picking an extrapolation
shape. The script attaches each rule to the axis and runs both operators:

```python
edge_axis = DomainAxis("edge", type=SpaceType(spacing=1.0, boundary=bc))
...
grad=Index("grad", inputs.T.gradient(axis=edge_axis), axes=(edge_axis,)),
lap=Index("lap", inputs.T.laplacian(axes=(edge_axis,)), axes=(edge_axis,)),
```

With `f = [0, 10, 20, 50]` and `h = 1`:

```text
  boundary             gradient                   laplacian
  Neumann(0)  default  [ 0. 10. 20.  0.]          [ 20.   0.  20. -60.]
  Neumann(5)           [ 5. 10. 20.  5.]          [ 10.   0.  20. -50.]
  Constant(0)          [  5.  10.  20. -10.]      [ 10.   0.  20. -80.]
  Constant(100)        [-45.  10.  20.  40.]      [110.   0.  20.  20.]
  Nearest()            [ 5. 10. 20. 15.]          [ 10.   0.  20. -30.]
  Wrap()               [-20.  10.  20. -10.]      [ 60.   0.  20. -80.]
  Linear()             [10. 10. 20. 30.]          [ 0.  0. 20.  0.]
```

Reading the table:

- The **interior** values never move (gradient 10 and 20, laplacian 0 and 20). A
  boundary condition only ever changes the two end positions.
- `Neumann(0)` forces the slope to 0 at both ends, so the gradient's first and
  last entries are exactly 0. `Neumann(5)` forces them to 5. That is a
  *derivative* being pinned — read straight off the output.
- `Constant(100)` puts a large value just outside the left edge, so the gradient
  there goes sharply negative: the field appears to fall steeply as you enter
  the domain.
- `Linear()` extends the local trend, which makes the laplacian 0 at both ends —
  a straight line has no curvature.

**Pick the rule that matches the physics:** `Wrap` for a periodic domain,
`Constant(v)` for a boundary held at a known value, `Neumann(v)` for a known
flux, `Nearest` / `Linear` for a neutral continuation.

---

## PART 4 — axis order on input is load-bearing (and unchecked)

Declaring the axes in the wrong order is accepted silently, so this failure is
worth seeing once: the same array, two orders.

```python
for label, order in (("axes=(y,x)  correct", (y_axis, x_axis)),
                     ("axes=(x,y)  swapped", (x_axis, y_axis))):
    swapped = Index("F", FIELD, axes=order)
    print(f"  {label}: recorded sizes {swapped.sizes}")
```

```text
  axes=(y,x)  correct: recorded sizes {'y': 3, 'x': 4}
  axes=(x,y)  swapped: recorded sizes {'x': 3, 'y': 4}
```

Both are accepted — the second just **relabels** the dimensions. Asking for
`df/dx` then differentiates down the *rows* instead of across the columns, and
since `f` is constant along `y` that returns all zeros rather than `2x`. A wrong
answer with no warning.

Axis order is load-bearing **on input**: `axes=` must follow the array's shape.
Only afterwards is it safe to ignore order and select by name (as `labeled()`
does — see the `field()` helper in the script).
