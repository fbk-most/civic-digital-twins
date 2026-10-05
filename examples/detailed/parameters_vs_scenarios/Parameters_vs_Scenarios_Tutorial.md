<!-- SPDX-License-Identifier: Apache-2.0 -->

# Parameter axes and Scenarios: the two things you control from outside

> Script: [`parameters_vs_scenarios.py`](parameters_vs_scenarios.py). Run it with
> `uv run python examples/detailed/parameters_vs_scenarios/parameters_vs_scenarios.py`.
> Each section below matches a `PART N` banner in the script.

Uncertainty *inside* a model is declared as indexes and integrated away by an ensemble. That is covered in [ensembles](../ensembles/Ensemble_Tutorial.md). This example covers the other half: the things you, the analyst, set on purpose and compare.

There are two such mechanisms, and they are not interchangeable:

| mechanism          | what it is                     | how                                                                                                                                                  |
| ------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **parameter axis** | a value you **sweep**          | declared with `Scenario(..., parameter_axes=[...])` and supplied as an array at `evaluate()` time. The result gains one dimension per parameter, so one run answers every combination. |
| **Scenario**       | an assumption you **override** | replaces an index the model already declared, which changes the ensemble itself. Each scenario is a separate run, and the runs are compared side by side. |

The dividing line is mechanical:

> a parameter axis changes the **values** fed into one fixed ensemble;
> a scenario changes the **ensemble**, so the weights themselves move.

That is why a sweep is one evaluation and a scenario comparison is several. It also decides **what the result arrays look like**, and therefore how you read them. Most of this tutorial is about that.

| Part | Topic                                                                       | What you learn to read                                     |
| ---- | --------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1    | One parameter sweep                                                         | the layout, raw shapes, weights, reading by name, per-branch reads |
| 2    | Two parameters: a grid                                                      | 2-D results by name, why order matters, and per branch     |
| 3    | A correlated sweep: several parameters along one axis                       | named axes and callable parameters                         |
| 4    | Scenarios: the four override forms                                          | how each override changes rows, weights and label arrays   |
| 5    | They compose: sweep inside each scenario                                    | which arrays can be stacked across scenarios, and which cannot |
| 6    | Traps and boundaries                                                        | errors you will meet, and averaging over a parameter       |

## The model used throughout

Every part uses one model. `p1` and `p2` are **parameters**: abstract indexes with no value, which you will supply. `cat` and `x` are **uncertainties**, which an ensemble integrates away.

```python
p1 = Index("p1")          # note: no value. An abstract Index IS the parameter.
p2 = Index("p2")
cat = CategoricalIndex("cat", {"a": 0.7, "b": 0.3})
x = DistributionIndex("x", stats.norm, {"loc": 1.0, "scale": 0.1})


@define("sweep")
class SweepModel(Model):
    @inputs
    class Inputs:
        p1: Index
        p2: Index
        cat: CategoricalIndex
        x: DistributionIndex

    @outputs
    class Outputs:
        y: Index

    def compute(self, inputs: Inputs) -> Outputs:
        # outcome "b" triples the output, so the categorical's weighting is
        # visible in every number below.
        factor = 1.0 + 2.0 * (inputs.cat == "b")
        return SweepModel.Outputs(
            y=Index("y", inputs.p1 * inputs.p2 * inputs.x * factor)
        )


model = SweepModel(inputs=SweepModel.Inputs(p1=p1, p2=p2, cat=cat, x=x))
y_idx = model.outputs.y
```

Since `E[x] = 1`, at `p1 = p2 = 1` the expected output is
`0.7·1 + 0.3·3 = 1.6`. Use it to check the numbers below.

---

## PART 1 — one parameter sweep: what the result looks like

Listing an index in `parameter_axes=` tells the `Scenario`: "do not sample this, I will supply it". The ensemble then **skips** it, and `evaluate(parameters={...})` provides an array of values for it.

```python
base_scenario = Scenario(model, parameter_axes=[p1, p2])     # "I will supply p1, p2"
base_ens = CrossProductEnsemble(base_scenario, n_samples_per_combo=400,
                                rng=np.random.default_rng(0))

sweep_res = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: np.array([1.0, 2.0, 3.0]), p2: np.array([1.0])},
)
```

Only `cat` and `x` are left for the ensemble: 2 `cat` branches × 400 samples = 800 rows.

```text
>>> [ix.name for ix in base_scenario.abstract_indexes()]
['cat', 'x']
>>> len(base_ens)
800
```

Every parameter you declare must be supplied, even with a single value: here `p2` gets `np.array([1.0])`. Leaving it out raises an error (see 6c).

### 1a. The layout: which dimension is which

With an ensemble alone, a result has one axis (the rows). With parameters, it has **one dimension per parameter, then the ensemble**. `layout` tells you the order:

```text
>>> sweep_res.layout
AxisLayout(Axis('p1', role='PARAMETER'): 3, Axis('p2', role='PARAMETER'): 1, Axis('_cross_product', role='ENSEMBLE'): 800)
>>> sweep_res.full_shape
(3, 1, 800)
```

The parameters come first, and the ensemble is last.

The parameters appear in the order of the keys in the `parameters=` dict you
pass to `evaluate()`. That order does **not** come from the `parameter_axes=`
list, and it is not alphabetical. Here both happen to say `p1, p2`, but section
2b shows how easily that changes. That is why this tutorial reads results **by
name** (1c) rather than by position.

### 1b. Raw arrays: every index has the full rank

Reading values works as in the ensembles tutorial, with `res[index]`. The
difference is the shape:

```python
sweep_y = np.asarray(sweep_res[y_idx])          # (p1, p2, ensemble)
sweep_p1 = np.asarray(sweep_res[p1])            # (p1, 1, 1)
sweep_cat = np.asarray(sweep_res[cat])          # (1, 1, ensemble)
sweep_x = np.asarray(sweep_res[x])              # (1, 1, ensemble)
(sweep_w,) = base_ens.ensemble_weights      # (ensemble,), shared by every p1
```

```text
  1b. raw shapes:
    y        (3, 1, 800)
    p1       (3, 1, 1)
    cat      (1, 1, 800)
    x        (1, 1, 800)
    weights  (800,)
```

Every array has the same number of dimensions as the layout, with **size 1 wherever it does not vary**. `p1` does not change from row to row, so its ensemble dimension is 1. `cat` and `x` do not depend on the parameters, so their parameter dimensions are 1. They broadcast against each other, and that is how the model computed `y`.

What they look like:

```text
>>> sweep_y
array([[[1.01257302, 0.98678951, 1.06404227, ..., 2.68858075,
         2.61269144, 3.03055886]],

       [[2.02514604, 1.97357903, 2.12808453, ..., 5.3771615 ,
         5.22538287, 6.06111772]],

       [[3.03771907, 2.96036854, 3.1921268 , ..., 8.06574225,
         7.83807431, 9.09167658]]], shape=(3, 1, 800))

>>> sweep_p1
array([[[1.]],

       [[2.]],

       [[3.]]])

>>> sweep_cat
array([[['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
         ...
         'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']]], dtype=object)
```

Each of the three blocks of `sweep_y` is the **same 800 draws**, scaled by
`p1 = 1, 2, 3`: the row for `p1=2` is exactly twice the row for `p1=1`. That is the point of a parameter axis: the uncertainty is drawn once and reused for every value you sweep.

The weights are **one vector of 800**, not one per parameter value. They belong to the ensemble, and the ensemble is the same for every `p1`:

```text
>>> sweep_w[[0, 400]]            # first row of each branch
array([0.00175 0.00075])
>>> np.allclose(sweep_res.weights, sweep_w)    # the result carries the same weights
True
```

`0.7 / 400 = 0.00175` for `cat=a` and `0.3 / 400 = 0.00075` for `cat=b`, as in the ensembles tutorial.

### 1c. Reading by name: `LabeledArray` and `.sel()`

Indexing `sweep_y[1, 0, :]` works, but only as long as you remember that `p1` is
the first dimension. A `LabeledArray` attaches the axis names to an array, and
its `.sel()` method selects **by name**, so the code no longer depends on the
order.

`res[idx]` returns a plain array. To name its dimensions, wrap it with the
result's `layout`. Inputs such as `cat` have size-1 dimensions, so they are
first broadcast to the full shape. A three-line helper does both, for any
index:

```python
from civic_digital_twins.dt_model import LabeledArray

def labeled_raw(res, idx):
    """Return the raw values of idx with every dimension named."""
    return LabeledArray(np.broadcast_to(np.asarray(res[idx]), res.full_shape),
                        res.layout)

y_raw = labeled_raw(sweep_res, y_idx)         # dims ('p1', 'p2', '_cross_product')
cat_raw = labeled_raw(sweep_res, cat)         # same dims, labels broadcast
```

```text
>>> y_raw
LabeledArray(dims=('p1', 'p2', '_cross_product'), shape=(3, 1, 800))
>>> cat_raw
LabeledArray(dims=('p1', 'p2', '_cross_product'), shape=(3, 1, 800))
```

`.sel()` takes **positions, not values**: `p1=1` means "the second `p1`". Get
the position of a value from `parameter_values_for`:

```python
p1_values = sweep_res.parameter_values_for(p1)
i_p1 = list(p1_values).index(2.0)         # value -> position along p1

y_at_p1_2 = y_raw.sel(p1=i_p1, p2=0).values       # (ensemble,)
cat_labels = cat_raw.sel(p1=0, p2=0).values       # (ensemble,) -- any p1 will do
```

```text
>>> p1_values
array([1., 2., 3.])
>>> i_p1
1
>>> y_raw.sel(p1=i_p1, p2=0)             # both parameters fixed: only the ensemble is left
LabeledArray(dims=('_cross_product',), shape=(800,))
>>> y_at_p1_2
array([2.02514604, 1.97357903, 2.12808453, 2.02098002, 1.89286613,
       ...
       5.35094883, 5.69846004, 5.3771615 , 5.22538287, 6.06111772])
>>> np.average(y_at_p1_2, weights=sweep_w)
np.float64(3.1920317717125086)
```

An integer in `.sel()` drops that axis, and a slice keeps it. Once **every**
parameter is fixed, what is left is a plain 800-row array aligned with `sweep_w`.
From there, everything in the ensembles tutorial applies unchanged.

`cat` does not depend on the parameters, so every `p1` gives the same labels.
`p1=0, p2=0` is just a convenient choice.

### 1d. `expected_value`: the ensemble averaged out, the parameters kept

`expected_value()` takes the weighted mean over the ensemble and **keeps every
parameter dimension**. Its named version is `labeled()`:

```python
ev = sweep_res.labeled(y_idx)                         # dims ('p1', 'p2')
```

```text
>>> ev
LabeledArray(dims=('p1', 'p2'), shape=(3, 1))
>>> ev.sel(p2=0).values                  # every p1, at the only p2
array([1.59601589, 3.19203177, 4.78804766])
>>> ev.sel(p1=1, p2=0).values            # p1=2
np.float64(3.1920317717125086)
```

`sweep_res.expected_value(y_idx)` returns the same numbers as a plain `(3, 1)`
array, without the names.

It is the weighted mean of the raw draws over the ensemble axis. In this model
the ensemble is the last axis, so the check is:

```python
manual = np.average(sweep_y, axis=-1, weights=sweep_w)    # ensemble is the last axis
```

```text
  1d. sweep_res.labeled(y_idx) dims: ('p1', 'p2')
      sweeping p1 over [1, 2, 3] with p2 = 1:
    p1= 1.0 -> ev.sel(p1=0, p2=0) =  1.5960   np.average(axis=-1) =  1.5960
    p1= 2.0 -> ev.sel(p1=1, p2=0) =  3.1920   np.average(axis=-1) =  3.1920
    p1= 3.0 -> ev.sel(p1=2, p2=0) =  4.7880   np.average(axis=-1) =  4.7880
```

We did **not** re-run the model three times, and the ensemble was not rebuilt.
One ensemble, one `evaluate()`, three answers. The categorical was integrated
away in the same way for all of them, so the three numbers differ *only* by
`p1`.

### 1e. Per-branch reads: mask, then name again

A branch is selected with a boolean mask over the ensemble rows, as in the
ensembles tutorial. `.sel()` takes only integers and slices, so the mask is
applied with plain numpy on `.values`. Then the reduced array is wrapped again
with `layout_of(y_idx)`, the layout of `expected_value`, so you are back to
reading by name:

```python
is_b = cat_labels == "b"                  # cat_labels from 1c, shape (800,)
e_b = LabeledArray(np.average(y_raw.values[..., is_b], axis=-1,
                              weights=sweep_w[is_b]),
                   sweep_res.layout_of(y_idx))        # dims ('p1', 'p2') again
```

```text
>>> cat_labels[395:405]                  # around the switch from a to b
array(['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'], dtype=object)
>>> is_b[395:405]
array([False, False, False, False, False,  True,  True,  True,  True,
        True])
>>> sweep_w[is_b].sum()                    # the branch's exact probability
np.float64(0.3)
>>> e_b
LabeledArray(dims=('p1', 'p2'), shape=(3, 1))
>>> e_b.sel(p2=0).values                 # E[y | cat=b] for every p1
array([2.99526428, 5.99052856, 8.98579284])
```

One line gives `E[y | cat=b]` for **every** `p1` at once (≈ 3 · `p1`, as
expected).

This is the one place where a position is unavoidable: numpy needs `[..., mask]`
and `axis=-1`. It is safe here because, in a model without DOMAIN axes, the
ensemble is always the last dimension, whatever the parameter order. When you
are not sure, look the position up by name or role instead of assuming it:

```python
from civic_digital_twins.dt_model.axes import ENSEMBLE

ens_axis = sweep_res.layout.axes_by_role(ENSEMBLE)[0][0]
sweep_res.layout.position_of(ens_axis)                 # -> 2
sweep_res.layout.position_of(sweep_res.layout.find_axis("p1"))   # -> 0
```

For a single parameter value, select it first and mask the resulting 1-D
array. No numpy indexing on parameter dimensions is needed:

```python
np.average(y_raw.sel(p1=i_p1, p2=0).values[is_b], weights=sweep_w[is_b])
```

```text
      one cell, by name: 5.9905  (p1=2)
```

### 1f. The trap: `.ravel()` mixes the parameter values

The ensembles tutorial reads results with `np.asarray(res[y]).ravel()`. That
is safe only when the ensemble is the *only* dimension. On a swept result it
puts all the parameter values into one long array:

```text
  1f. sweep_y.ravel().shape = (2400,)  (= 3 p1 values x 800 rows, mixed)
      plain mean of ALL of it = 3.9916  <- not any E[y]
```

2400 values no longer line up with the 800 weights, and their mean is not the
expected value at any `p1`. Fix every parameter with `.sel()` first (1c), and
only then work with the 800 rows. Never flatten.

---

## PART 2 — two parameters: a grid

Same `base_scenario` and `base_ens` as Part 1. Each parameter has its own
dimension, so supplying two arrays gives every combination in one run:

```python
grid_res = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: np.array([1.0, 2.0, 3.0]), p2: np.array([10.0, 20.0])},
)

labelled = grid_res.labeled(y_idx)                 # dims ('p1', 'p2')
p1_vals = grid_res.parameter_values_for(p1)
p2_vals = grid_res.parameter_values_for(p2)
```

```text
>>> labeled_raw(grid_res, y_idx)         # the raw draws
LabeledArray(dims=('p1', 'p2', '_cross_product'), shape=(3, 2, 800))
>>> labelled                             # the expected values
LabeledArray(dims=('p1', 'p2'), shape=(3, 2))
>>> p1_vals, p2_vals
(array([1., 2., 3.]), array([10., 20.]))
```

The script prints the grid as a table. Every cell is read with
`labelled.sel(p1=i, p2=j)`, so the table is correct whatever the order of the
dimensions:

```python
for i, v1 in enumerate(p1_vals):
    row = "".join(f"{float(labelled.sel(p1=i, p2=j).values):12.3f}"
                  for j in range(len(p2_vals)))
```

```text
                 p2=10       p2=20
    p1=1        15.960      31.920
    p1=2        31.920      63.841
    p1=3        47.880      95.761
```

Six answers, still **one** evaluation and **one** ensemble. Every cell is the
Part 1 base value (≈ 1.596) times `p1 · p2`, so the grid is easy to check.

### 2a. Cells, rows and columns with `.sel()`

An integer drops the axis, and an axis you don't mention is kept whole:

```text
>>> labelled.sel(p1=0, p2=1).values      # one cell: p1=p1_vals[0]=1, p2=p2_vals[1]=20
np.float64(31.920317717125087)
>>> labelled.sel(p2=1)                   # a whole column: every p1 at p2=20
LabeledArray(dims=('p1',), shape=(3,))
>>> labelled.sel(p2=1).values
array([31.92031772, 63.84063543, 95.76095315])
```

Adding a `p3` later does not change any `.sel(p1=..., p2=...)` you already
wrote. A positional index such as `labelled.values[0, 1]` would now point
somewhere else.

### 2b. Why by name: the dimension order comes from the `parameters=` dict

The scenario says `parameter_axes=[p1, p2]`, but that list only declares
*which* indexes are parameters. The **order of the dimensions** follows the
keys of the `parameters=` dict. Here is the same scenario and the same values,
with the dict written `p2` first:

```python
swapped_res = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p2: np.array([10.0, 20.0]), p1: np.array([1.0, 2.0, 3.0])},
)
swapped_lab = swapped_res.labeled(y_idx)
```

```text
>>> swapped_res.layout
AxisLayout(Axis('p2', role='PARAMETER'): 2, Axis('p1', role='PARAMETER'): 3, Axis('_cross_product', role='ENSEMBLE'): 800)
>>> swapped_lab.values                   # rows: p2, columns: p1 -- transposed
array([[15.96015886, 31.92031772, 47.88047658],
       [31.92031772, 63.84063543, 95.76095315]])
```

```text
  2b. same scenario, parameters= written as {p2: ..., p1: ...}:
    dims                   : ('p2', 'p1')   (was ('p1', 'p2'))
    .values[0]             : [15.96015886 31.92031772 47.88047658]   (p2=10, every p1)
    .sel(p1=0)             : [15.96015886 31.92031772]   (p1=1, every p2 -- as before)
    .sel(p1=0, p2=1)       : 31.920   (same cell as before)
```

The numbers are the same, but the array is transposed. Positional code
(`.values[0]`, `ev[i, j]`, `raw[i, j, :]`, `axis=0`) now reads a different
quantity. Here the shapes differ (`(3, 2)` vs `(2, 3)`), so a wrong index will
often raise an error. If two parameters have the **same number of values**,
nothing fails and you silently read the wrong cells. `.sel()` returns the same
answer either way.

Named axes (Part 3) follow the same rule: they appear in the order of the
`parameter_axes={...}` dict, and always **before** any parameters given
directly as arrays.

### 2c. Finding a cell by its value

`np.argmax` returns positions in the array's own order. Zip them with `dims`
to name them, then translate each position into a value:

```python
flat = np.argmax(labelled.values)
best = {dim: int(pos)                              # int(): .sel needs Python ints
        for dim, pos in zip(labelled.dims,
                            np.unravel_index(flat, labelled.values.shape))}
labelled.sel(**best)
```

```text
  2c. argmax positions by name: {'p1': 2, 'p2': 1}
      largest E[y] at p1=3, p2=20: 95.761
```

The `int(...)` matters. `np.unravel_index` returns numpy integers, and
`.sel()` currently only recognises Python `int`. Given a numpy integer, it
fails with an unhelpful `ValueError: zip() argument 2 is shorter than argument
1`.

### 2d. Per-branch over the whole grid

The same pattern as 1e: mask the ensemble rows, average, then wrap the result
again so it can be read by name:

```python
grid_raw = labeled_raw(grid_res, y_idx)
grid_cat = labeled_raw(grid_res, cat).sel(p1=0, p2=0).values
(grid_w,) = base_ens.ensemble_weights
grid_b = grid_cat == "b"
grid_e_b = LabeledArray(np.average(grid_raw.values[..., grid_b], axis=-1,
                                   weights=grid_w[grid_b]),
                        grid_res.layout_of(y_idx))
```

```text
  2d. E[y | cat=b] over the grid: LabeledArray(dims=('p1', 'p2'), shape=(3, 2))
[[ 29.95264282  59.90528563]
 [ 59.90528563 119.81057126]
 [ 89.85792845 179.71585689]]
      grid_e_b.sel(p1=2, p2=1) = 179.716   (p1=3, p2=20)
```

A `(p1, p2)` table of conditional means from one evaluation, read by name.

---

## PART 3 — a correlated sweep: several parameters along one axis

A grid assumes every combination makes sense. Sometimes parameters move
**together**: a "level" setting that raises `p1` and `p2` at the same time. You
want the diagonal, not the grid.

For that, `evaluate()` accepts **named axes** (`parameter_axes={name: values}`)
and, in `parameters=`, **functions** of those axes instead of arrays. Each
function receives the axis values through an argument with the same name:

```python
free_scenario = Scenario(model)                    # p1, p2 stay abstract
free_ens = CrossProductEnsemble(free_scenario, n_samples_per_combo=400,
                                rng=np.random.default_rng(0))

diag_res = Evaluation(free_scenario).evaluate(
    ensemble=free_ens,
    parameter_axes={"level": np.array([1.0, 2.0, 3.0])},   # ONE named axis
    parameters={p1: lambda level: level,                    # p1 = level
                p2: lambda level: 10.0 * level},            # p2 = 10 * level
)
```

Two parameters, **one** dimension, and `.sel()` uses the name you chose:

```python
diag_lab = diag_res.labeled(y_idx)
```

```text
>>> diag_res.layout
AxisLayout(Axis('level', role='PARAMETER'): 3, Axis('_cross_product', role='ENSEMBLE'): 800)
>>> diag_res.named_axis_values
{'level': array([1., 2., 3.])}
>>> diag_lab
LabeledArray(dims=('level',), shape=(3,))
>>> diag_lab.sel(level=1).values         # level=2 -> p1=2, p2=20
np.float64(63.840635434250174)
>>> labeled_raw(diag_res, p1).sel(_cross_product=0).values   # p1 along the axis (any row)
array([1., 2., 3.])
```

```text
  layout:
    level           PARAMETER  size=3
    _cross_product  ENSEMBLE   size=800
  named_axis_values       : {'level': array([1., 2., 3.])}
  p1 along the axis       : [1. 2. 3.]
  p2 along the axis       : [10. 20. 30.]
  diag_res.labeled(y_idx) : LabeledArray(dims=('level',), shape=(3,))
  .values                 : [ 15.96015886  63.84063543 143.64142973]
  .sel(level=1)           : 63.841   (level=2: p1=2, p2=20)
```

How to read it:

- The axis is named after the key you chose (`"level"`), not after any index.
  Select on it with `.sel(level=...)`. Its values come from
  `diag_res.named_axis_values["level"]`.
- `parameter_values_for(p1)` works only for parameters given as **arrays**. For
  a function-backed `p1` it raises `KeyError`. Read the values it actually took
  from `labeled_raw(diag_res, p1)` instead. `p1` does not vary from row to row, so
  any row of the ensemble axis (`_cross_product=0`) gives them.
- The three values are the diagonal of a grid: (1, 10), (2, 20), (3, 30). They
  are 1.596 × 10, × 40 and × 90.

Here we used `Scenario(model)` **without** `parameter_axes=`. The `Scenario`
version declares which indexes are parameters and makes sure you supply them.
The named-axis version supplies them all at `evaluate()` time.

---

## PART 4 — Scenarios: overrides that change the ensemble

A `Scenario` override replaces an index the **model** declared. Unlike a
parameter, it is not a value fed in at `evaluate()` time. It changes what the
ensemble *is*, so each scenario needs its own ensemble and its own run.

There are four override forms:

| override value     | effect       | on a `CategoricalIndex` / `DistributionIndex` |
| ------------------ | ------------ | --------------------------------------------- |
| `dict[str, float]` | **reweight** | same outcomes, new probabilities              |
| `str`              | **pin**      | collapse to one outcome (branches shrink)     |
| `list[str]`        | **restrict** | keep a subset, renormalise its weights        |
| a distribution     | **replace**  | for a `DistributionIndex`, swap the law       |

Each scenario is built from the same `model` and evaluated at `p1 = p2 = 1`.
Keep the results (`scenario_res`) **and** the ensembles (`scenario_ens`): you
need the ensembles for the weights.

```python
scenarios = {
    "baseline":            Scenario(model, parameter_axes=[p1, p2]),
    "reweight a=.2 b=.8":  Scenario(model, overrides={cat: {"a": 0.2, "b": 0.8}},
                                    parameter_axes=[p1, p2]),
    "pin cat='b'":         Scenario(model, overrides={cat: "b"},
                                    parameter_axes=[p1, p2]),
    "restrict to ['a']":   Scenario(model, overrides={cat: ["a"]},
                                    parameter_axes=[p1, p2]),
    "replace x law":       Scenario(model,
                                    overrides={x: stats.norm(loc=2.0, scale=0.1)},
                                    parameter_axes=[p1, p2]),
}

scenario_res = {}
scenario_ens = {}
for label, sc in scenarios.items():
    ens = CrossProductEnsemble(sc, n_samples_per_combo=400,
                               rng=np.random.default_rng(0))
    scenario_res[label] = Evaluation(sc).evaluate(
        ensemble=ens,
        parameters={p1: np.array([1.0]), p2: np.array([1.0])},
    )
    scenario_ens[label] = ens
```

For each scenario, the script reads the number of rows, the raw shape of the
`cat` labels, the distinct weights, the weighted mean of `x`, and `E[y]`. The
last two are read by name, as in Part 1:

```python
for label, res in scenario_res.items():
    (w,) = scenario_ens[label].ensemble_weights
    x_draws = labeled_raw(res, x).sel(p1=0, p2=0).values
    mean_x = float(np.average(x_draws, weights=w))
    e_y = float(res.labeled(y_idx).sel(p1=0, p2=0).values)
```

```text
  scenario              rows    cat shape           weights (unique)  mean x    E[y]
  baseline               800  (1, 1, 800)          [0.00075 0.00175]   0.997  1.5960
  reweight a=.2 b=.8     800  (1, 1, 800)            [0.0005 0.002 ]   0.998  2.5955
  pin cat='b'            400    (1, 1, 1)                   [0.0025]   0.996  2.9890
  restrict to ['a']      400  (1, 1, 400)                   [0.0025]   0.996  0.9963
  replace x law          800  (1, 1, 800)          [0.00075 0.00175]   1.997  3.1960
```

Row by row, this is what each override did to the result:

- **reweight**: same 800 rows, same labels, **different weights**:
  `0.2 / 400 = 0.0005` for `a`, `0.8 / 400 = 0.002` for `b`. The draws did
  not change, only how much each one counts.

  ```text
  >>> scenario_ens["reweight a=.2 b=.8"].ensemble_weights[0][[0, 799]]   # first a row, last b row
  array([0.0005, 0.002 ])
  ```

- **pin**: half the rows, and `cat` **is no longer part of the ensemble**. Its
  array has shape `(1, 1, 1)`, a single constant, and only `x` is left to
  sample:

  ```text
  >>> np.asarray(scenario_res["pin cat='b'"][cat])
  array([[['b']]], dtype='<U1')
  >>> [ix.name for ix in scenarios["pin cat='b'"].abstract_indexes()]
  ['x']
  ```

  There is only one branch, so you don't need a mask. If shared code builds
  one anyway, `labeled_raw` broadcasts the constant to every row, so the 1c
  code keeps working:

  ```text
  >>> labeled_raw(scenario_res["pin cat='b'"], cat).sel(p1=0, p2=0).values.shape
  (400,)                                 # every label is 'b'
  ```

- **restrict**: also half the rows, but `cat` **stays** in the ensemble, with
  one outcome. Its labels have shape `(1, 1, 400)`, all `'a'`, and the
  remaining outcome gets all the probability (`1 / 400 = 0.0025`). The same
  mask code still works, which is the practical difference from pinning.

- **replace**: the branch structure and the weights are identical to the
  baseline. What changed is the *values* drawn for `x` (mean ≈ 2 instead of 1).

The `rows` column shows that a scenario is not a parameter: a parameter can
never change the number of branches, because it is not part of the ensemble.

### Same seed, same draws

Every scenario was built with a fresh `np.random.default_rng(0)`. When two
scenarios have the same branch structure, they get **the same draws**:

```text
  baseline vs reweight: identical x draws? True
```

So the difference between baseline (1.5960) and reweight (2.5955) comes
entirely from the weights, not from sampling noise. These are *common random
numbers*: use the same seed for every scenario you want to compare. With
different seeds, part of each difference would be noise.

---

## PART 5 — they compose: sweep inside each scenario

This is the usual shape of a real study. Each scenario is its own run, and the
parameter sweep happens *inside* it:

```python
sweep_values = np.array([1.0, 2.0, 3.0])
study = {
    "baseline":       Scenario(model, parameter_axes=[p1, p2]),
    "pessimistic":    Scenario(model, overrides={cat: {"a": 0.2, "b": 0.8}},
                               parameter_axes=[p1, p2]),
    "cat pinned 'a'": Scenario(model, overrides={cat: "a"},
                               parameter_axes=[p1, p2]),
}

study_res = {}
for label, sc in study.items():
    ens = CrossProductEnsemble(sc, n_samples_per_combo=400,
                               rng=np.random.default_rng(0))
    study_res[label] = Evaluation(sc).evaluate(
        ensemble=ens,
        parameters={p1: sweep_values, p2: np.array([1.0])},
    )
```

### 5a. Raw arrays do not line up across scenarios

```text
  raw y per scenario:
    baseline         LabeledArray(dims=('p1', 'p2', '_cross_product'), shape=(3, 1, 800))
    pessimistic      LabeledArray(dims=('p1', 'p2', '_cross_product'), shape=(3, 1, 800))
    cat pinned 'a'   LabeledArray(dims=('p1', 'p2', '_cross_product'), shape=(3, 1, 400))
```

The parameter dimensions match, but the **ensemble dimension does not**: the
pinned scenario has half the rows. Even when the sizes happen to match
(baseline vs pessimistic), row `i` carries a different weight in each. Raw draws
from different scenarios must not be stacked, subtracted or compared row by
row. Reduce each scenario on its own first.

### 5b. Expected values do line up

After `expected_value()`, the ensemble dimension is gone and only the parameter
dimensions remain, which are the same for every scenario. Select the same named
cells in each scenario, and stacking is safe:

```python
table = np.stack([res.labeled(y_idx).sel(p2=0).values      # (p1,) by name
                  for res in study_res.values()])          # (scenario, p1)
```

`.sel(p2=0)` gives the `p1` sweep in every scenario, even if one of them was
evaluated with a different `parameters=` order (2b).

```text
>>> table
array([[1.59601589, 3.19203177, 4.78804766],
       [2.59547903, 5.19095805, 7.78643708],
       [0.996338  , 1.992676  , 2.98901401]])
```

```text
  scenario              p1=1      p1=2      p1=3
  baseline             1.596     3.192     4.788
  pessimistic          2.595     5.191     7.786
  cat pinned 'a'       0.996     1.993     2.989
```

You can read the table two ways:

```text
  across a row (p1 sweep, baseline)  : [1.59601589 3.19203177 4.78804766]
  down a column (scenarios, p1=2)    : [3.19203177 5.19095805 1.992676  ]
  scenario effect vs baseline        : [0.99946314 1.99892628 2.99838942]
```

- **Across a row** (`table[0]`): the parameter sweep, i.e. what changes as you
  turn the dial. One run produced the whole row.
- **Down a column** (`table[:, 1]`): the scenario comparison, i.e. what changes
  if your *assumptions* were different. Each row needed its own run and its own
  ensemble.
- **Row minus row** (`table[1] - table[0]`): the effect of the scenario at each
  parameter value, without sampling noise thanks to the shared seed (Part 4).

The stacked `table` is a plain array whose order **you** chose (scenarios in
dict order, then `p1`), so indexing it by position is safe.

The same applies to anything else you reduce per scenario: a per-branch mean
(1e), a per-branch percentile, and so on. Compute it **inside** each scenario
with that scenario's own weights, select the same named cells, then stack the
results.

The uncertainty never appears in the table: `cat` and `x` were integrated away
in every cell. Uncertainty, assumptions and parameters get three different
treatments, and only the last two are under your control.

---

## PART 6 — traps and boundaries

### 6a. A Scenario acts *on* a declared index; it cannot add one

There is no override that **adds** a new categorical. The model author decides
*what* is uncertain; the analyst decides *which assumptions* to test about it.
Adding a new uncertainty means editing the model.

### 6b. An index cannot be both a parameter and an override

```python
clash = Scenario(model, overrides={p1: 5.0}, parameter_axes=[p1, p2])
Evaluation(clash).evaluate(
    ensemble=CrossProductEnsemble(clash, n_samples_per_combo=10),
    parameters={p1: np.array([1.0]), p2: np.array([1.0])},
)
```

```text
  6b. an index cannot be both a parameter and an override
      raises: The following indexes appear in both parameters= and Scenario.overrides: 'p1'
```

One says "sweep this", the other says "fix this".

### 6c. Every declared parameter must be supplied

```python
Evaluation(base_scenario).evaluate(ensemble=base_ens,
                                   parameters={p1: np.array([1.0])})   # p2 missing
```

```text
  6c. forgetting to supply a declared parameter
      raises: Scenario declares 'p2' as parameter_axes but it was not supplied in parameters=. Pass their valu
```

A parameter you do not want to sweep still needs a value: pass a one-element
array, as Part 1 does for `p2`.

### 6d. Parameters carry no probability and are never averaged over

```text
  6d. parameters carry no probability, and are never averaged over
      sweep_res.labeled(y_idx) -> LabeledArray(dims=('p1', 'p2'), shape=(3, 1)), one value PER p1
```

The **ensemble** dimension was integrated away; the **parameter** dimension was
kept. The library will not average over `p1`, because a value you sweep has no
probability distribution. If you want a single number across `p1`, you choose
the weights yourself and apply them to the expected values:

```python
ev_p1 = sweep_res.labeled(y_idx).sel(p2=0).values       # (3,): one value per p1
my_weights = np.array([0.5, 0.3, 0.2])              # YOUR belief about p1
np.average(ev_p1, weights=my_weights)
```

```text
      your own average over p1 (weights [0.5 0.3 0.2]) = 2.7132
```

If you find yourself doing this often, `p1` may really be an uncertainty. In
that case, declare it as a `DistributionIndex` or `CategoricalIndex` in the
model and let the ensemble integrate it.

---

## Reading results, in one place

`labeled_raw(res, idx)` is the three-line helper from 1c. In `.sel(...)`, the
values are **positions** along each axis. Translate values to positions with
`list(res.parameter_values_for(p)).index(value)`.

| you want                                 | do                                                                                       | result                          |
| ---------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------- |
| which dimension is which                 | `res.layout` (raw), `res.layout_of(idx)` (after `expected_value`)                        |                                 |
| raw values of any index, named           | `labeled_raw(res, idx)`                                                                  | dims `(*params, ensemble)`      |
| the ensemble weights                     | `(w,) = ens.ensemble_weights`, or `res.weights`                                          | `(ensemble,)`, shared by all parameter values |
| the values a parameter took              | `res.parameter_values_for(p)` (arrays) / `labeled_raw(res, p)` (functions)               | `(n,)`                          |
| draws at one parameter value             | `labeled_raw(res, idx).sel(p1=i, p2=j).values`                                           | `(ensemble,)`, aligned with `w` |
| branch labels of each row                | `labeled_raw(res, cat).sel(p1=0, p2=0).values`                                           | `(ensemble,)`                   |
| E[y] for every parameter value           | `res.labeled(idx)`                                                                       | dims `(*params)`                |
| E[y] at one cell / row / column          | `res.labeled(idx).sel(p1=i, p2=j)` / `.sel(p1=i)` / `.sel(p2=j)`                         |                                 |
| E[y \| branch] for every parameter value | `LabeledArray(np.average(raw.values[..., m], axis=-1, weights=w[m]), res.layout_of(idx))` | dims `(*params)`                |
| where the argmax is, by name             | `dict(zip(lab.dims, map(int, np.unravel_index(np.argmax(lab.values), lab.values.shape))))` |                               |
| a position you can't avoid (`axis=`)     | `res.layout.position_of(res.layout.find_axis("p1"))`                                     | `int`                           |
| compare scenarios                        | `np.stack([res.labeled(idx).sel(...).values for res in ...])`                            | `(scenario, ...)`               |

Never `.ravel()` a result that has parameter dimensions, and never assume the
position of a parameter dimension: it follows the order of the `parameters=`
dict (2b).

## In one line each

| quantity           | treatment                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| **parameter axis** | a value you sweep. One ensemble, one run, one new result dimension per parameter (or one per named axis). No probability.     |
| **Scenario**       | an assumption you override. A different ensemble, a separate run, compared side by side after reducing.                       |
| **uncertainty**    | neither: declared in the model, integrated away by the ensemble (see [ensembles](../ensembles/Ensemble_Tutorial.md)).          |

Ask which of the three a quantity is, and the API follows.
