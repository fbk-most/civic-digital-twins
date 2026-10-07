<!-- SPDX-License-Identifier: Apache-2.0 -->

# CDT cheatsheet

Details in the tutorials: [axes](axis_tutorial/Axis_Tutorial.md) ·
[differential operators](differential_operators_tutorial/Differential_Operators_Tutorial.md) ·
[ensembles](ensembles_tutorial/Ensembles_Tutorial.md) ·
[parameters vs scenarios](parameters_vs_scenarios_tutorial/Parameters_vs_Scenarios_Tutorial.md)

## Mental model

| quantity    | declare as                                                         | becomes                                     |
|-------------|--------------------------------------------------------------------|---------------------------------------------|
| uncertainty | `DistributionIndex`, `CategoricalIndex`, `ConditionalDistributionIndex` | ENSEMBLE axis, averaged by `expected_value()` |
| parameter   | `Index("p")` + `Scenario(model, parameter_axes=[p])`               | PARAMETER axis, never averaged              |
| scenario    | `Scenario(model, overrides={idx: ...})`                            | a separate run with its own ensemble        |

Result dims are always `(*PARAMETER, *ENSEMBLE, *DOMAIN sorted by name)`.

## Axes

```python
x = DomainAxis("x", type=SpaceType(spacing=0.5, boundary=Wrap()))   # define ONCE, import everywhere
```

- Types: `SetType` (labels) · `SequenceType` (ordered: `diff`, `shift`, `cumulative`) · `TimeType` · `SpaceType` (+ `gradient`, `laplacian`).
- Ready-made: `TIME_AXIS` (via `TimeseriesIndex`). PARAMETER and ENSEMBLE axes are created for you.
- `==` ignores the type: an untyped `Axis("x", DOMAIN)` placed first (`ones * u`) strips the `SpaceType` from the result.
- Reusable shape: subclass `Index`, set `FIXED_AXES = (x, y)`, pass it to `axes=` (like `TimeseriesIndex`).

## Building indexes and `compute()`

| you write                         | result                                                            |
|-----------------------------------|-------------------------------------------------------------------|
| `Index("u", arr, axes=(y, x))`    | axes matched to `arr.shape` **by position**; a wrong order is silent |
| `Index("u", arr)`                 | **no axes**; fails at evaluation (`unexpected ndim`)              |
| `Index("f", formula, axes=(...))` | axes checked against the formula (`ValueError` if they differ)    |
| `Index("p")`                      | a placeholder you must supply, **not** a zero                     |
| `a * b`, `a + b`, `a.sum(axis=x)` | a graph node, not an `Index`: wrap the final expression once      |
| `idx.broadcast(ax)`               | adds `ax` at size 1, when no operand supplies it                  |

- Disjoint axes in `*` make an outer product and emit `AxesInferenceWarning`; pass `axes=` if intended.
- Sums: never start from `None` (fails only at evaluation). Use a generator:
  `Index("em", sum(i.em_int[t] * i.em_prof[t] * i.lu_grid[t] for t in i.em_int), axes=(Y, X, TIME_AXIS))`.

## Differential operators (`SpaceType` axes)

- `f.diff(axis=x)` per sample · `f.gradient(axis=x)` per unit distance, one axis · `f.laplacian(axes=(x, y))` summed.
- Boundaries: `Reflect` = `Neumann(0)` default (flat edge) · `Neumann(v)` flux · `Constant(v)` value · `Wrap()` periodic · `Nearest()` / `Linear()`.
- Boundaries only change the two end values.

## Ensembles

| ensemble               | use for                                                  |
|------------------------|----------------------------------------------------------|
| none: `evaluate()`     | nothing uncertain                                        |
| `DistributionEnsemble` | output distribution, quantiles (equal weights)           |
| `PartitionedEnsemble`  | which independent input drives the output (grid)         |
| `CrossProductEnsemble` | per-branch answers, rare branches, conditional indexes   |

```python
(w,) = ens.ensemble_weights                    # one array per ENSEMBLE axis
ey_b = np.average(y[m], weights=w[m])          # E[y | branch]; never y[m].mean()
```

- Fresh `np.random.default_rng(seed)` per ensemble; same seed across compared scenarios.
- `max_categorical_size` (default 20): larger categoricals are silently **sampled**.
- `n_samples_per_combo` is per branch. `sample_across(..., total=)` needs `total` ≫ number of scenarios.

## Parameters and scenarios

- Supply every declared parameter (a 1-element array is fine). Dim order = keys of `parameters=`.
- Diagonal sweep: `parameter_axes={"level": v}`, `parameters={p1: lambda level: level}`.
- Overrides: `{"a": .2, "b": .8}` reweight · `"b"` pin · `["a"]` restrict · a distribution replaces.
- Compare scenarios after reducing (expected values), never draw by draw.

## Reading results

| you want                       | do                                                                   |
|--------------------------------|----------------------------------------------------------------------|
| E[·] with named dims           | `lab = res.labeled(idx)`; check `lab.dims`                           |
| E[·] at one cell               | `lab.sel(p=0, time=2, x=3, y=1).values`                              |
| E[·] grid at one time step     | `lab.sel(p=0, time=2).values` (axes you don't name are kept)         |
| keep an axis, cut it           | `lab.sel(time=slice(0, 2))`                                          |
| at a parameter *value*         | `lab.sel(p=list(res.parameter_values_for(p)).index(2.0))`            |
| your own dim order             | `np.transpose(lab.values, [lab.dims.index(n) for n in ("p", "time", "y", "x")])` |
| position for numpy             | `res.layout_of(idx).position_of(TIME_AXIS)`                          |
| every draw, named              | `labeled_draws(res, idx).sel(p=0, x=3, y=1).values` (helper below)   |
| E[·] as a plain array          | `res.expected_value(idx)` (dims: `res.layout_of(idx)`)               |
| draws as a plain array         | `np.asarray(res[idx])` (dims: `res.layout`)                          |
| compare scenarios              | `np.stack([r.labeled(idx).sel(p=0).values for r in results])`        |

- `.sel()` takes **positions** (Python `int`, or a slice), not values; an int drops the axis.
- Never index by position or `.ravel()` a result with parameter axes. DOMAIN axes are alphabetical, so `(y, x)` comes back as `(x, y)`. Renaming axes to force an order just moves the problem.
- With one ensemble member, `labeled()` equals the single draw. It is still the expected value.

```python
def labeled_draws(res, idx):
    """Like res.labeled(idx), but keeps the ENSEMBLE axis: one value per draw."""
    layout, carried = res.layout, set(idx.output_axes)
    raw = np.asarray(res[idx])
    raw = raw.reshape((1,) * (len(layout.axes) - raw.ndim) + raw.shape)
    keep = [i for i, ax in enumerate(layout.axes) if ax.role != DOMAIN or ax in carried]
    stray = tuple(i for i in range(len(layout.axes)) if i not in keep)
    entries = [layout.entries[i] for i in keep]
    values = np.broadcast_to(np.squeeze(raw, axis=stray), tuple(n for _, n in entries))
    return LabeledArray(values, AxisLayout(entries))
```

## Errors you will meet

| error                                                        | cause                                                    |
|--------------------------------------------------------------|----------------------------------------------------------|
| `Outputs.y: expected GenericIndex ..., got multiply`         | bare formula as an output: wrap it in `Index(...)`       |
| `unsupported operand ... 'NoneType' and 'float'`             | a sum started from `None`                                |
| `AssertionError: ... unexpected ndim`                        | an array without `axes=`                                 |
| `zip() argument 2 is shorter than argument 1`                | numpy int in `.sel()` (use `int()`), or `axes=` on a scalar |
| `gradient/laplacian require a SpaceType axis`                | untyped axis copy, or axis not declared with `SpaceType` |
| `labeled(idx)`: `values.shape ... does not match layout`     | `idx` ignores a swept parameter: use `expected_value(idx)` |
| pyright rejects `overrides={x: stats.norm(...)}`             | [known issue](parameters_vs_scenarios_tutorial/ISSUE_distribution_protocol.md); runtime is fine |
