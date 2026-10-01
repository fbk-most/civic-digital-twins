<!-- SPDX-License-Identifier: Apache-2.0 -->

# Parameter axes and Scenarios: the two things you control from outside

> Script: [`parameters_vs_scenarios.py`](parameters_vs_scenarios.py) — run it with
> `uv run python examples/detailed/parameters_vs_scenarios/parameters_vs_scenarios.py`.
> Each section below matches a `PART N` banner in the script.

Uncertainty *inside* a model is declared as indexes and integrated away by an
ensemble — that is the subject of
[ensembles](../ensembles/Ensemble_Tutorial.md). This example is about the other
half: the things you, the analyst, set deliberately and compare.

There are exactly two such mechanisms, and they are not interchangeable:

| mechanism          | what it is              | how                                                                                                   |
|--------------------|-------------------------|-------------------------------------------------------------------------------------------------------|
| **parameter axis** | a value you **sweep**   | declared via `Scenario(..., parameter_axes=[...])`, supplied at `evaluate()` time as an array. The result gains a real dimension per parameter, so one run answers every combination at once. |
| **Scenario**       | an assumption you **override** | shadows an index the model already declared, changing what the ensemble itself is. Each scenario is a separate run, compared side by side. |

The dividing line is mechanical, not stylistic:

> a parameter axis changes the **values** fed into one fixed ensemble;
> a scenario changes the **ensemble**, so the weights themselves move.

That is why a sweep is one evaluation and a scenario comparison is several —
and why you cannot express one with the other.

| Part | Topic                                                                 |
|------|-----------------------------------------------------------------------|
| 1    | A parameter sweep: one run, one dimension added per parameter         |
| 2    | Two parameters at once — a genuine grid, not nested loops             |
| 3    | Scenarios: the four override forms and what each does to the ensemble |
| 4    | They compose: sweep *inside* each scenario, read the table two ways   |
| 5    | The boundary — what a Scenario can and cannot do                      |

## The model used throughout

```python
p1 = Index("p1")          # note: no value. An abstract Index IS the parameter.
p2 = Index("p2")
cat = CategoricalIndex("cat", {"a": 0.7, "b": 0.3})
x = DistributionIndex("x", stats.norm, {"loc": 1.0, "scale": 0.1})

def compute(self, inputs: Inputs) -> Outputs:
    factor = 1.0 + 2.0 * (inputs.cat == "b")      # outcome "b" triples y
    return SweepModel.Outputs(
        y=Index("y", inputs.p1 * inputs.p2 * inputs.x * factor)
    )
```

`p1`, `p2` are parameters (abstract indexes with no value); `cat` and `x` are
uncertainties the ensemble integrates away. Since `E[x] = 1`, at `p1 = p2 = 1`
the expected output is `0.7·1 + 0.3·3 = 1.6` — useful for checking the numbers
below.

---

## PART 1 — a parameter sweep: one run, one new dimension

Listing an index in `parameter_axes=` tells the `Scenario` "do not sample this,
I will supply it". The ensemble then **skips** it, and
`evaluate(parameters={...})` provides an array of values. The result gains a
`PARAMETER` axis of that length.

```python
base_scenario = Scenario(model, parameter_axes=[p1, p2])
base_ens = CrossProductEnsemble(base_scenario, n_samples_per_combo=400,
                                rng=np.random.default_rng(0))

one_d = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: np.array([1.0, 2.0, 3.0]), p2: np.array([1.0])},
)
```

```text
  the result layout now carries the parameters as real dimensions:
    p1         PARAMETER  size=3
    p2         PARAMETER  size=1
    _cross_product ENSEMBLE   size=800

  sweeping p1 over [1, 2, 3] with p2 = 1:
    p1= 1.0 -> E[y] =   1.596
    p1= 2.0 -> E[y] =   3.192
    p1= 3.0 -> E[y] =   4.788
```

Note what did **not** happen: we did not re-run the model three times, and the
ensemble was not rebuilt. One ensemble, one `evaluate()`, three answers. The
categorical was integrated away identically in all of them, so the three
numbers are directly comparable — they differ *only* by `p1`.

---

## PART 2 — two parameters: a grid, not nested loops

Every parameter axis is independent, so supplying two arrays gives their full
cross product in one shot. The result is genuinely 2-D and labelled by name:

```python
grid_res = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: np.array([1.0, 2.0, 3.0]), p2: np.array([10.0, 20.0])},
)
labelled = grid_res.labeled(model.outputs.y)      # dims ('p1', 'p2')
```

```text
  expected_value shape : (3, 2)   dims ('p1', 'p2')
  p1 values : [1. 2. 3.]
  p2 values : [10. 20.]

                 p2=10       p2=20
    p1=1        15.960      31.920
    p1=2        31.920      63.841
    p1=3        47.880      95.761
```

Six answers, still **one** evaluation and **one** ensemble. Because the axes are
named, you select by meaning rather than position:

```python
labelled.sel(p1=0, p2=1)          # indices into each parameter axis -> p1=1, p2=20
grid_res.parameter_values_for(p1) # the actual values along that axis
```

```text
  labelled.sel(p1=0, p2=1) = 31.920   (p1=1, p2=20)
```

Every cell is just the PART 1 base value (≈ 1.596) times `p1 · p2`, e.g.
`1.596 · 1 · 20 = 31.92`. The model is linear in both parameters, so the grid is
easy to check by eye.

That is what makes a sweep safe to extend: adding `p3` does not renumber
anything you already wrote.

---

## PART 3 — Scenarios: overrides that change the ensemble itself

A `Scenario` override shadows an index the **model** declared. Unlike a
parameter, it is not a value fed in at `evaluate()` time — it changes what the
ensemble *is*, which is why each scenario needs its own ensemble and its own
run.

There are four override forms:

| override value       | effect       | on a `CategoricalIndex` / `DistributionIndex`  |
|----------------------|--------------|------------------------------------------------|
| `dict[str, float]`   | **reweight** | same outcomes, new probabilities               |
| `str`                | **pin**      | collapse to one outcome (branches shrink)      |
| `list[str]`          | **restrict** | keep a subset, renormalise its weights         |
| a distribution       | **replace**  | for a `DistributionIndex`, swap the law        |

```python
scenarios = {
    "baseline":            Scenario(model, parameter_axes=[p1, p2]),
    "reweight a=.2 b=.8":  Scenario(model, overrides={cat: {"a": 0.2, "b": 0.8}}, ...),
    "pin cat='b'":         Scenario(model, overrides={cat: "b"}, ...),
    "restrict to ['a']":   Scenario(model, overrides={cat: ["a"]}, ...),
    "replace x law":       Scenario(model, overrides={x: stats.norm(loc=2.0, scale=0.1)}, ...),
}
```

```text
  scenario               scenarios    E[y] at p1=p2=1
  baseline                     800             1.5960
  reweight a=.2 b=.8           800             2.5955
  pin cat='b'                  400             2.9890
  restrict to ['a']            400             0.9963
  replace x law                800             3.1960
```

Read the `scenarios` column (branches × 400 samples): pinning and restricting
**halve** the ensemble, because an outcome was removed from the cross product
entirely. Reweighting keeps all branches but changes their weights. Replacing a
distribution leaves the branch structure alone and changes what is drawn inside
it.

That column is the tell that a scenario is not a parameter: a parameter can
never change the number of branches, because it is not part of the ensemble at
all.

---

## PART 4 — they compose: sweep inside each scenario

This is the normal shape of a real study. Each scenario is its own run; the
parameter sweep happens *within* it:

```python
for label, sc in study.items():
    ens = CrossProductEnsemble(sc, n_samples_per_combo=400, rng=np.random.default_rng(0))
    res = Evaluation(sc).evaluate(
        ensemble=ens,
        parameters={p1: sweep_values, p2: np.array([1.0])},
    )
```

```text
  scenario              p1=1      p1=2      p1=3
  baseline             1.596     3.192     4.788
  pessimistic          2.595     5.191     7.786
  cat pinned 'a'       0.996     1.993     2.989
```

The result is a table you read two ways:

- **Across a row** — the parameter sweep: what changes as we turn the dial. One
  run produced the whole row.
- **Down a column** — the scenario comparison: what changes if our
  *assumptions* were different. Each row needed its own run and its own
  ensemble.

The uncertainty never appears in the table: `cat` and `x` were integrated away
into every single cell. Three different things, three different treatments —
and only the first two are under your control.

---

## PART 5 — the boundary: what a Scenario cannot do

### 5a. A Scenario acts *on* a declared index — it cannot invent one

There is no override that **adds** a new categorical. The model author decides
*what* is uncertain; the analyst decides *which assumptions* to test about it.
Adding a new uncertainty means editing the model.

### 5b. An index cannot be both a parameter and an override

```python
clash = Scenario(model, overrides={p1: 5.0}, parameter_axes=[p1, p2])
Evaluation(clash).evaluate(...)
# ValueError: The following indexes appear in both parameters= and Scenario.overrides: 'p1'
```

Sensible: one says "sweep this", the other says "fix this".

### 5c. Parameters carry no probability, and are never averaged over

```python
weighted.expected_value(model.outputs.y).shape     # (3, 1) -- one value PER p1
```

The **ensemble** axis was integrated away; the **parameter** axis was kept. If
you want a single number across `p1`, you must decide the weights yourself — the
library will not guess them, because a swept lever has no probability
distribution.

---

## In one line each

| quantity           | treatment                                                                                    |
|--------------------|----------------------------------------------------------------------------------------------|
| **parameter axis** | a value you sweep. One ensemble, one run, one new result dimension per parameter. No probability. |
| **Scenario**       | an assumption you override. A different ensemble, a separate run, compared side by side.     |
| **uncertainty**    | neither: declared in the model, integrated away by the ensemble (see [ensembles](../ensembles/Ensemble_Tutorial.md)). |

Ask which of the three a quantity is, and the API follows.
