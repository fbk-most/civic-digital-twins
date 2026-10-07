# Choosing an ensemble

An ensemble turns a model's uncertainty into concrete numbers to compute over.

We will use deliberately abstract names so that the example remains as general as possible:

| name       | index type                       | what it represents                                                 |
| ---------- | -------------------------------- | ------------------------------------------------------------------ |
| `x1, x2` | `DistributionIndex`            | continuous uncertainties                                           |
| `cat1`   | `CategoricalIndex`             | discrete probabilistic outcomes`"a"` / `"b"`                  |
| `cat2`   | `CategoricalIndex`             | discrete probabilistic outcomes`"p"` / `"q"`                  |
| `xc`     | `ConditionalDistributionIndex` | a distribution whose*parameters* depend on `cat1` and `cat2` |
| `y`      |                                  | the model output                                                   |

The same model is grown to show the purpose of each ensemble:

| Part | Model                         | Ensemble                 | What we get                                                     | What is answered                                                             |
| ---- | ----------------------------- | ------------------------ | --------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 0    | `y = k1 * k2`, constants    | **none**           | the single value of`y`                                        |                                                                              |
| 1    | `y = x1 * x2`               | `DistributionEnsemble` | the samples of`y` resulting from sampling `x1` and `x2`   | The distribution of`y` when inputs are uncertain                           |
| 2    | same`y = x1 * x2`           | `PartitionedEnsemble`  | the value of`y` in a grid made from `x1` and `x2` samples | How do`x1` and `x2` contribute to `y`                                  |
| 3    | add`cat1`, `cat2`, `xc` | `CrossProductEnsemble` | Samples of`y` **per branch**                            | How the distribution of`y` depends on the categories introduced (branches) |
| 4    | —                            | any                      | practical traps                                                 |                                                                              |

An ensemble is picked based on *what we want to read from the results*. Some ensembles may be equivalent in the expected value we obtain but do not allow for easy access to useful information.

For example:

* `DistributionEnsemble`: All draws come straight off a plain
  array. Thus, every source of uncertainty is mixed into one flat axis: you
  cannot tell which input drove a draw, and a rare branch gets only the few
  draws chance gives it.
* `PartitionedEnsemble`: Gives the same E[`y`] as sampling, but keeps each
  input on its own axis. Averaging out one axis shows how much `y` moves
  with the other, so you learn which input drives `y`. It assumes the inputs
  are independent.
* `CrossProductEnsemble`: Gives the same E[`y`] again, but enumerates every
  branch with its *exact* probability and a guaranteed number of samples, so
  `y` can be read **per branch**, even a branch with 0.2 % probability. It is
  also the only built-in choice when a distribution depends on a category.

---

## PART 0 — no uncertainty: no ensemble at all

An ensemble exists to integrate over uncertainty. If every input is a fixed value, there is nothing to integrate over: omit ensembles and evaluate directly on the scenarios.

```python
@define("fixed")
class CertainModel(Model):
    @inputs
    class Inputs:
        k1: Index
        k2: Index

    @outputs
    class Outputs:
        y: Index

    def compute(self, inputs: Inputs) -> Outputs:
        return CertainModel.Outputs(y=Index("y", inputs.k1 * inputs.k2))

k1 = Index("k1", 100.0)                                       # plain constants,
k2 = Index("k2", 1.0)                                         # nothing to sample

cert_model = CertainModel(inputs=CertainModel.Inputs(k1=k1, k2=k2))
cert_scenario = Scenario(cert_model)

cert_res = Evaluation(cert_scenario).evaluate()             # no ensemble=
cert_y = np.asarray(cert_res[cert_model.outputs.y])
```

```
>>> cert_scenario.abstract_indexes()
[]
>>> cert_y
array(100.)
```

The scenario has no abstract indexes, and `y` comes back as a plain scalar —
no ENSEMBLE axis, no weights, no `expected_value` needed. Sweeping inputs you
*control* (`parameters=` / `parameter_axes=`) is still deterministic: it adds
PARAMETER axes, not an ensemble (see [parameters_vs_scenarios_tutorial](../parameters_vs_scenarios_tutorial/Parameters_vs_Scenarios_Tutorial.md)).

---

## PART 1 — `DistributionEnsemble`: continuous noise only

The simplest version of the model with uncertainty simply passes the uncertainty of its inputs to the outputs. There are no categories or branches, it simply takes samples from the inputs and it results in an array of outputs based on those samples, all equally weighted.

```python
@define("basic")
class BasicModel(Model):
    @inputs
    class Inputs:
        x1: DistributionIndex
        x2: DistributionIndex

    @outputs
    class Outputs:
        y: Index

    def compute(self, inputs: Inputs) -> Outputs:
        return BasicModel.Outputs(y=Index("y", inputs.x1 * inputs.x2))

x1 = DistributionIndex("x1", stats.norm, {"loc": 100.0, "scale": 25.0})   # Mean 100
x2 = DistributionIndex("x2", stats.norm, {"loc": 1.0, "scale": 0.2})      # Mean 1

basic_model = BasicModel(inputs=BasicModel.Inputs(x1=x1, x2=x2))          # Create the model
basic_scenario = Scenario(basic_model)                                    # Create the scenario

basic_ens = DistributionEnsemble(basic_scenario, size=50_000,
                                  rng=np.random.default_rng(0))

basic_res = Evaluation(basic_scenario).evaluate(ensemble=basic_ens)   # Evaluate the scenario

basic_draws = np.asarray(basic_res[basic_model.outputs.y]).ravel()    # Access values through 'y' Index

(basic_w,) = basic_ens.ensemble_weights                               # Access the weights of each sample in ensemble

basic_x1 = np.asarray(basic_res[x1]).ravel()                          # Access values x1 took using directly the 'x1' Index
```

```
>>> basic_draws
array([ 97.24151122, 110.24919288, 107.80984825, ..., 122.87875432,
        90.37481924,  70.87188049], shape=(50000,))

>>> basic_w
array([2.e-05, 2.e-05, 2.e-05, ..., 2.e-05, 2.e-05, 2.e-05],
      shape=(50000,))

>>> basic_x1
array([103.14325553,  96.69737842, 116.01056626, ..., 122.88568002,
       115.75979687,  78.66634566], shape=(50000,))
```

Every draw carries the **same** weight (`1/size`), so `np.percentile` can be used to explore the distribution. Quantiles, tail probabilities, histograms — anything you would compute on a plain array of samples.

**Why this ensemble:** all the uncertainty is continuous, so there is no
structure to exploit. Use `DistributionEnsemble` when the question is "what does
the distribution of the output look like" and nothing categorical is in play.

---

## PART 2 — `PartitionedEnsemble`: which uncertainty drives the output?

Same `y = x1 * x2`, but a different question: not "what is `y`'s
distribution" but "which of `x1` and `x2` drives it more" — because you might
pay to measure one of them better.

A flat sample cannot answer that: every draw welds one `x1` value to one `x2`
value, and you can never un-mix them. Separate axes keep them distinguishable:

```python
# Model and scenario stay the same

part_ens = PartitionedEnsemble(
    basic_scenario,
    axes=[
        EnsembleAxisSpec("x1_axis", indexes=[x1], size=60),
        EnsembleAxisSpec("x2_axis", indexes=[x2], size=40),
    ],
    rng=np.random.default_rng(0),
)

part_res = Evaluation(basic_scenario).evaluate(ensemble=part_ens)
grid = np.asarray(part_res[basic_model.outputs.y])   # shape (60, 40): a GRID
```

```
>>> part_ens.ensemble_axes
(Axis('x1_axis', role='ENSEMBLE'), Axis('x2_axis', role='ENSEMBLE'))

>>> grid
array([[ 94.14018508,  79.01182011, 139.02406861, ...,  91.0645857 ,
         75.47570199,  74.231784  ],
       [ 88.25694957,  74.07402287, 130.33584118, ...,  85.37355796,
         70.75889237,  69.5927123 ],
       [105.88434623,  88.86868991, 156.36757673, ..., 102.42506017,
         84.89143456,  83.49233551],
       ..., shape=(60, 40))

>>> part_res.expected_value(basic_model.outputs.y)
np.float64(103.70190089717185)
```

Thus we draw a total of 60 + 40 = 100 samples but obtain 2400 combinations: every `x1` draw is combined with every `x2` draw.

Row `i` holds `x1[i]` fixed; column `j` holds `x2[j]` fixed. Marginalising one axis at a time isolates each source:

```python
by_x1 = grid.mean(axis=1)   # average out x2
by_x2 = grid.mean(axis=0)   # average out x1
```

```
>>> by_x1
array([104.93377951,  98.37600466, 118.0244614 , 104.40398686,
        88.11174911, 110.93276324, 134.90188195, 125.82400573,
        83.83716371,  69.55124185,  85.88360218, 102.7870429 ,
        ...])
   
>>> by_x1.shape
(60,)
```

`by_x1` shows how much `y` moves when only `x1` varies (with `x2` averaged out), and vice versa.

**Why this ensemble:** the grid shape *is* the answer, and it is cheap — 100 draws cover 2400 combinations because the axes broadcast.

Use only when inputs are **independent**: if the two were correlated, this factorial grid would invent combinations that never occur, and a conditional index inside a `CrossProductEnsemble` (next part) would be the honest choice instead.

---

## PART 3 — `CrossProductEnsemble`: per-branch answers and a rare branch

Before introducing the `CrossProductEnsemble`,  let's start by expanding the model to include two independent `CategoricalIndex`. They represent branching possibilities for the model. For example, `cat1` can go to option `a` with 99% probability and to option `b` with 1% probability. `cat2` also has 2 possibilities, resulting in a total of  **4** **branches:**  (`cat1,cat2`) $\in$ { (`a,p`), (`b,p`), (`a,q`), (`b,q`) }. The model adapts a factor depending on the values `cat1` and `cat2` take.

```python
cat1 = CategoricalIndex("cat1", {"a": 0.99, "b": 0.01})   # deliberately rare "b"
cat2 = CategoricalIndex("cat2", {"p": 0.80, "q": 0.20})

@define("cat")
class CatModel(Model):
    @inputs
    class Inputs:
        cat1: CategoricalIndex
        cat2: CategoricalIndex
        x1: DistributionIndex
        x2: DistributionIndex

    @outputs
    class Outputs:
        y: Index

    def compute(self, inputs: Inputs) -> Outputs:
        is_b = inputs.cat1 == "b"
        is_q = inputs.cat2 == "q"
        # factor: 1 if cat1=a, 50 if (b, p), 400 if (b, q)
        factor = 1.0 + (49.0 + 350.0 * is_q) * is_b
        return CatModel.Outputs(y=Index("y", inputs.x1 * inputs.x2 * factor))
```

Because `x1 * x2` has mean 100, the true values are easy to state:

| branch                   | P (exact)            | factor | E[`y` \| branch]   | contribution to E[`y`] |
| ------------------------ | -------------------- | ------ | -------------------- | ------------------------ |
| `cat1=a`, any `cat2` | 0.99                 | 1      | 100                  | 99                       |
| `cat1=b, cat2=p`       | 0.01 × 0.80 = 0.008 | 50     | 5 000                | 40                       |
| `cat1=b, cat2=q`       | 0.01 × 0.20 = 0.002 | 400    | 40 000               | 80                       |
|                          |                      |        | **E[`y`] =** | **219**            |

The `cat1=b, cat2=q` branch holds 0.2 % of the probability but over a third of E[`y`].

### 3a. Same model, same budget: `DistributionEnsemble` vs `CrossProductEnsemble`

Both ensembles accept categoricals, so we run the same scenario through each
with the same budget of 1000 rows.

```python
# Create model and scenario as usual
cat_model = CatModel(inputs=CatModel.Inputs(cat1=cat1, cat2=cat2, x1=x1, x2=x2))
cat_scenario = Scenario(cat_model)
```

**`DistributionEnsemble`** — samples the categoricals along with `x1` and `x2`:

```python
de_ens = DistributionEnsemble(cat_scenario, size=1_000,       # 1000 independent draws for each sampleable quantity
                              rng=np.random.default_rng(0))

de_res = Evaluation(cat_scenario).evaluate(ensemble=de_ens)   # Evaluate to get result
```
The result is a single axis (a 1D array) and so all the information also mirrors that format.

To access the results:
```python
de_y = np.asarray(de_res[cat_model.outputs.y]).ravel()  # y of each row
```

```text
>>> de_y
array([1.32542269e+02, 8.18555097e+01, 1.10510789e+02, 1.04830503e+02,
       1.50108553e+02, 6.15880901e+01, 1.35344666e+02, 7.76217737e+01,
       ...])
```

The following gives us access to the weights that would be used to compute the mean and percentiles:
```python
(de_w,) = de_ens.ensemble_weights                       # weight of each row
```
```text
>>> de_w
array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
       0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001,
       ...])
```
But as we are using a `DistributionEnsemble` they are all the same!

We can access the particular values each category got in the sampling as we would access any other sampled variable in the model:
```python
de_c1 = np.asarray(de_res[cat1]).ravel()                # cat1 label of each row. cat1 is an Index
de_c2 = np.asarray(de_res[cat2]).ravel()                # cat2 label of each row
```
```text
>>> de_c1
array(['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
       'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
       ...])
```

In order to look at a particular combination of the categories (branch), we need to use a **mask** on the category value arrays:

```python
de_bq = (de_c1 == "b") & (de_c2 == "q")         # create mask

de_bq_n = int(de_bq.sum())                      # samples in the branch

de_bq_y = de_y[de_bq]                           # values of 'y' in the branch
```
```
>>> de_bq
array([...
       False, False, False, False, False, False, False, False, False,
       False, False,  True, False, False, False, False, False, False,
       False, False, False, False, False, False, False, False, False,
       ...])

>>> de_bq_n
2

>>> de_bq_y
array([30461.70561002, 61355.36687007])
```

Using the weights we get the (empirical) probability that a sample landed in the  desired branch:

```python
de_bq_prob = float(de_w[de_bq].sum())                             # its probability
de_bq_mean = float(np.average(de_y[de_bq], weights=de_w[de_bq]))  # E[y | branch]
```

If we do this for each branch, we get the following summary:

```text
  DistributionEnsemble: 1000 rows, E[y] = 236.38   (true 219)
    
    cat1  cat2   rows       P  E[y|branch]  True branch mean
    a     p       791  0.7910        99.52               100
    a     q       198  0.1980        99.65               100
    b     p         9  0.0090      5123.74              5000
    b     q         2  0.0020     45908.54             40000
```

Each branch gets rows in proportion to its probability, decided by chance: the
`cat1=b, cat2=q` branch got **2 rows out of 1000**. Its mean rests on two
points, the probabilities are only counts (9/1000 is not the true 0.008), and
the overall E[`y`] lands 8 % too high.

**`CrossProductEnsemble`** — enumerates the branches instead. We go through the
same steps and see what changes.

```python
cp_ens = CrossProductEnsemble(cat_scenario, n_samples_per_combo=250,  # 1000 samples divided into the 4 branches = 250
                              rng=np.random.default_rng(0))

cp_res = Evaluation(cat_scenario).evaluate(ensemble=cp_ens)   # Evaluate to get result
```
The result is again a single axis (a 1D array) of 1000 rows. The difference is
that the rows are no longer in random order: they come **branch by branch**,
250 rows each — rows 0–249 are `a, p`, 250–499 `a, q`, 500–749 `b, p` and
750–999 `b, q`.

To access the results:
```python
cp_y = np.asarray(cp_res[cat_model.outputs.y]).ravel()  # y of each row
```

```text
>>> cp_y
array([1.27565555e+02, 1.02573230e+02, 1.20466633e+02, 1.08080859e+02,
       6.29441528e+01, 1.00544504e+02, 1.07235062e+02, 1.28560131e+02,
       ...])
```

The weights:
```python
(cp_w,) = cp_ens.ensemble_weights                       # weight of each row
```
```text
>>> cp_w
array([...
       3.168e-03, 3.168e-03, 3.168e-03, 3.168e-03, 3.168e-03, 3.168e-03,
       3.168e-03, 3.168e-03, 3.168e-03, 3.168e-03, 7.920e-04, 7.920e-04,
       7.920e-04, 7.920e-04, 7.920e-04, 7.920e-04, 7.920e-04, 7.920e-04,
       ...])

>>> cp_w[[0, 250, 500, 750]]        # the first row of each branch
array([3.168e-03, 7.920e-04, 3.200e-05, 8.000e-06])
```
Unlike the `DistributionEnsemble`, the weights are **not** all the same. Each
branch's *exact* probability is split evenly over its 250 rows:

| branch           | P (exact)            | weight of each row (÷ 250) |
| ---------------- | -------------------- | -------------------------- |
| `cat1=a, cat2=p` | 0.99 × 0.80 = 0.792  | 0.003168                   |
| `cat1=a, cat2=q` | 0.99 × 0.20 = 0.198  | 0.000792                   |
| `cat1=b, cat2=p` | 0.01 × 0.80 = 0.008  | 0.000032                   |
| `cat1=b, cat2=q` | 0.01 × 0.20 = 0.002  | 0.000008                   |

Every branch has the same *number* of rows, so the *weights* are what carry
the probabilities. Inside one branch all rows share the same weight; the
weights start to matter as soon as a mask holds rows of different weight
(see 3d), so always pass them (`np.average(..., weights=...)`).

The category values are accessed exactly as before:
```python
cp_c1 = np.asarray(cp_res[cat1]).ravel()                # cat1 label of each row
cp_c2 = np.asarray(cp_res[cat2]).ravel()                # cat2 label of each row
```
```text
>>> cp_c1[495:505]                  # around the switch from cat1=a to cat1=b
array(['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'], dtype=object)
```

The mask is built the same way:

```python
cp_bq = (cp_c1 == "b") & (cp_c2 == "q")         # create mask

cp_bq_n = int(cp_bq.sum())                      # samples in the branch

cp_bq_y = cp_y[cp_bq]                           # values of 'y' in the branch
```
```text
>>> cp_bq[745:755]                  # the branch starts at row 750
array([False, False, False, False, False,  True,  True,  True,  True,
        True])

>>> cp_bq_n
250

>>> cp_bq_y
array([50068.67093565, 28496.80516136, 58603.45912826, 22970.99267516,
       ...])
```

The rare branch now has **250** samples instead of 2. Summing its weights gives
its *exact* probability (250 × 0.000008 = 0.002), not an estimate:

```python
cp_bq_prob = float(cp_w[cp_bq].sum())                             # its probability
cp_bq_mean = float(np.average(cp_y[cp_bq], weights=cp_w[cp_bq]))  # E[y | branch]
```

If we do this for each branch, we get the following summary:

```text
  CrossProductEnsemble: 1000 rows, E[y] = 215.53   (true 219)

    cat1  cat2   rows       P  E[y|branch]  True branch mean
    a     p       250  0.7920       100.01               100
    a     q       250  0.1980       100.92               100
    b     p       250  0.0080      4938.31              5000
    b     q       250  0.0020     38417.67             40000
```

Every branch gets the same guaranteed number of rows (250), and its weights sum
to its **exact** probability, however rare. With the same budget, the overall
E[`y`] is within 2 % of the truth.

Both results have the same shape: one flat axis of rows, each tagged with its
`cat1`/`cat2` labels, read with a mask. What differs is only **how the rows are
spread across the branches**: by chance (*sampled*) or evenly with exact
weights (*enumerated*).

### 3b. The elegant way: `ConditionalDistributionIndex` with `CrossProductEnsemble`

`CatModel` works, but the branch logic is hard-coded into `compute()` as
arithmetic on 0/1 indicators, and it can only *scale* a distribution that is
the same in every branch. The natural way to say "this input behaves
differently in each branch" is to put the dependence on the input itself.

A `ConditionalDistributionIndex` does exactly that. Instead of one
distribution, it takes a **factory**: a function that receives the outcome of
each parent categorical (as keyword arguments named after them) and returns
the distribution to use in that branch.

```python
def xc_dist(cat1: str, cat2: str):
    if cat1 == "a":
        return stats.lognorm(s=0.5, scale=20.0)       # the common case
    if cat2 == "p":
        return stats.lognorm(s=0.8, scale=1000.0)     # rare, moderate
    return stats.lognorm(s=0.8, scale=8000.0)         # rare AND extreme


xc = ConditionalDistributionIndex("xc", parents=[cat1, cat2], factory=xc_dist)

@define("full")
class FullModel(Model):
    @inputs
    class Inputs:
        cat1: CategoricalIndex
        cat2: CategoricalIndex
        xc: ConditionalDistributionIndex
        x2: DistributionIndex

    @outputs
    class Outputs:
        y: Index

    def compute(self, inputs: Inputs) -> Outputs:
        # x2 is the same PLAIN index from Part 1: it knows nothing about branches.
        return FullModel.Outputs(y=Index("y", inputs.xc * inputs.x2))


full_model = FullModel(inputs=FullModel.Inputs(
    cat1=cat1, cat2=cat2, xc=xc, x2=x2,
))
full_scenario = Scenario(full_model)
```

The price is that **only `CrossProductEnsemble` can evaluate it**. `DistributionEnsemble` draws every input independently from its own fixed distribution. `xc` has no fixed distribution: to draw it, you must first know which branch the draw is in. `CrossProductEnsemble` works in that order — it fixes the branch first, and inside each branch `xc`'s distribution is known.

### 3c. `FullModel` per branch

Reading `FullModel` works exactly as in 3a: evaluate, take the four arrays,
mask each branch. With 5000 samples per branch (20 000 scenarios):

```python
full_ens = CrossProductEnsemble(full_scenario, n_samples_per_combo=5000,
                                rng=np.random.default_rng(0))
full_res = Evaluation(full_scenario).evaluate(ensemble=full_ens)
(full_w,) = full_ens.ensemble_weights
full_y = np.asarray(full_res[full_model.outputs.y]).ravel()
full_c1 = np.asarray(full_res[cat1]).ravel()
full_c2 = np.asarray(full_res[cat2]).ravel()
```

```text
>>> len(full_ens)
20000
>>> np.unique(full_w)               # one weight per branch: P(branch) / 5000
array([4.000e-07, 1.600e-06, 3.960e-05, 1.584e-04])
```

The same mask per branch as in 3a, now also with the 99th percentile:

```text
    cat1  cat2   P (exact)   E[y|branch]         p99
    a     p         0.7920         22.63       65.89
    a     q         0.1980         22.85       67.42
    b     p         0.0080       1381.25     6294.73
    b     q         0.0020      10904.65    52350.07
    E[y] overall = 55.3057
```

Inside one branch every row has the same weight, so a plain
`np.percentile(branch_values, 99)` is correct there. Across branches it would
not be: the weights differ, as `np.unique(full_w)` shows (see 3d).

Inputs can be read from the result the same way as the output: `full_res[xc]`
gives the value `xc` took in each row, drawn from that row's branch
distribution.

**Why this ensemble:** the `cat1=b, cat2=q` branch holds 0.2 % of the
probability and dominates `y`. Enumeration gives it a *guaranteed* 5000 samples
and an *exact* weight, so its mean and p99 are solid. A sampler spending the
same 20 000 evaluations would land roughly 40 there — enough for a rough mean,
far too few for a 99th percentile — and the branch probabilities would
themselves be estimates. Shrink the budget and that corner vanishes, at which
point the conditional answer does not exist at all.

Sampling is also simply unavailable here: `xc` is conditional, and
`DistributionEnsemble` rejects conditional indexes outright (3b).

### 3d. When a mask mixes branches: marginals

So far every mask fixed *both* categoricals (a **joint** branch), and all the
rows inside it had the same weight. A **marginal** fixes only *one*
categorical, e.g. `cat1=b` regardless of `cat2`. The mask then picks up rows
from several branches, with **different** weights, and this is where the
weights really matter.

To see it, a deliberately tiny ensemble — 4 branches × 2 samples = 8 rows,
small enough to print:

```python
tiny_ens = CrossProductEnsemble(full_scenario, n_samples_per_combo=2,
                                rng=np.random.default_rng(0))
tiny_res = Evaluation(full_scenario).evaluate(ensemble=tiny_ens)

(tiny_w,) = tiny_ens.ensemble_weights                            # weight
tiny_y = np.asarray(tiny_res[full_model.outputs.y]).ravel()      # output
tiny_c1 = np.asarray(tiny_res[cat1]).ravel()                     # label 1
tiny_c2 = np.asarray(tiny_res[cat2]).ravel()                     # label 2
```

Mask on `cat1` only:

```python
b_mask = tiny_c1 == "b"
w_sub = tiny_w[b_mask]
y_sub = tiny_y[b_mask]
```

```text
>>> tiny_c2[b_mask]
array(['p', 'p', 'q', 'q'], dtype=object)
>>> w_sub
array([0.004, 0.004, 0.001, 0.001])
>>> y_sub
array([  348.52840863,  1277.02280611, 17048.28011001, 14566.88919037])
>>> w_sub.sum()
np.float64(0.010000000000000002)
```

Two things to notice:

- The mask holds two rows of `cat2=p` and two of `cat2=q`, but `p` is four
  times more probable — and the weights (0.004 vs 0.001) say so. **Sample
  counts are not probabilities.**
- The weights sum to 0.01, the probability of `cat1=b`, not to 1. A
  conditional mean must divide by that sum:

$$
E[y \mid \text{cat1=b}] = \frac{\sum w_i\, y_i}{\sum w_i} \quad \text{over the masked rows}
$$

That division is the renormalisation, and `np.average(values, weights=w)` does
it internally. Computing `E[y | cat1=b]` five ways:

```python
np.average(y_sub, weights=w_sub)                 # right
np.sum(w_sub * y_sub) / np.sum(w_sub)            # right (same thing, spelled out)
np.sum((w_sub / np.sum(w_sub)) * y_sub)          # right (normalise first)
np.sum(w_sub * y_sub)                            # WRONG
y_sub.mean()                                     # WRONG
```

```text
  computing E[y | cat1=b] from those 4 rows:
    np.average(y, weights=w)         =    3811.7374
    sum(w*y) / sum(w)                =    3811.7374
    normalise w first, then sum(w*y) =    3811.7374
    sum(w*y)  [NO division]          =      38.1174  <- WRONG (branch contribution)
    y_sub.mean()  [no weights]       =    8310.1801  <- WRONG (counts != probabilities)
```

- `sum(w*y)` without division is scaled down by `sum(w) = 0.01`. That is the
  branch's **contribution** to the overall mean, not its mean.
- `y_sub.mean()` gives `cat2=q` the same say as `cat2=p`, more than doubling
  the answer.

The same mask on each outcome gives both marginals:

```python
for c1_outcome in ("a", "b"):
    mask = tiny_c1 == c1_outcome          # ONLY cat1 is constrained
```

```text
  MARGINAL over cat2 -- mask on cat1 ONLY:
    cat1    n  P (exact)         E[y]
    a       4     0.9900       17.450
    b       4     0.0100     3811.737

  MARGINAL over cat1 -- mask on cat2 ONLY:
    cat2    n  P (exact)         E[y]
    p       4     0.8000       24.108
    q       4     0.2000      180.532
```

The `cat2=q` marginal shows the effect most strongly: its mask mixes `cat1=a`
rows (weight 0.099) with `cat1=b` rows (weight 0.001). Weighted, the rare
`cat1=b` rows pull it up to 180.5; a plain `.mean()` would give about 7900.

### 3e. The sanity check: law of total expectation

The joint branches decompose the overall mean **exactly**:

$$
E[y] = \sum_{\text{branches}} P(\text{branch}) \cdot E[y \mid \text{branch}]
$$

Each branch contributes its probability times its conditional mean:

```python
contributions = []
for c1_outcome in ("a", "b"):
    for c2_outcome in ("p", "q"):
        mask = (tiny_c1 == c1_outcome) & (tiny_c2 == c2_outcome)
        branch_weights = tiny_w[mask]
        branch_values = tiny_y[mask]
        contributions.append(
            branch_weights.sum() * np.average(branch_values, weights=branch_weights)
        )

tiny_overall = float(tiny_res.expected_value(full_model.outputs.y))
```

```text
>>> contributions                   # (a, p), (a, q), (b, p), (b, q)
[np.float64(12.784291416431172), np.float64(4.491188516320415), np.float64(6.502204858984646), np.float64(31.615169300381478)]
```

```text
  check (law of total expectation, on the 8 rows above):
    sum of P(branch) * E[y|branch] = 55.3929
    expected_value()               = 55.3929
```

The `cat1=b, cat2=q` branch, with 0.2 % of the weight, contributes 31.6 of the
55.4 — more than the two `cat1=a` branches together (12.8 + 4.5). Run this
check whenever you slice a result by branch: if it fails, your masks do not
partition the ensemble or your weights are mishandled.

---

## PART 4 — traps that apply whichever ensemble you chose

### 4a. `max_categorical_size` silently abandons exactness above 20

The model uses a categorical `cat_big` with 25 outcomes `L0`…`L24`, where
outcome `Li` has weight proportional to `i + 1` and contributes the value `i`
to `y`. Its true mean is exactly 16. Each setting is evaluated with 4 different
seeds:

```python
CrossProductEnsemble(many_scenario, max_categorical_size=mx,
                     rng=np.random.default_rng(sd))
```

```text
  4a. max_categorical_size (support 25, true mean 16)
      max_categorical_size= 20 -> size=20 SAMPLED (noisy)     [15.7  15.55 15.4  15.65]
      max_categorical_size= 25 -> size=25 enumerated (exact)  [16. 16. 16. 16.]
```

With the default limit of 20, a 25-outcome categorical is **sampled** rather
than enumerated: 20 random draws, so each seed gives a different, wrong mean.
Raise the limit to 25 and every outcome is enumerated with its exact weight —
the answer is exactly 16 regardless of seed. Nothing warns you about the
switch; set `max_categorical_size` above your largest support when you need
exactness.

### 4b. Support-only categoricals cannot be enumerated

```python
cat_bare = CategoricalIndex("cat_bare", ["p", "q"])   # no weights
CrossProductEnsemble(Scenario(...))                    # -> ValueError
```

```text
  4b. a categorical built from a bare list has no weights
      raises: CategoricalIndex 'cat_bare' was constructed without weights (support-only); '.outcomes' ...
```

Enumeration needs a probability per outcome, and a bare list provides none.
Give weights at construction, or via a `Scenario` dict override.

### 4c. Reproducibility needs a *fresh* generator each time

The helper `basic_mean(generator)` runs the Part 1 model with 400 draws and
returns `E[y]`. Each line calls it twice:

```python
basic_mean(None), basic_mean(None)
basic_mean(np.random.default_rng(0)), basic_mean(np.random.default_rng(0))
shared = np.random.default_rng(0)
basic_mean(shared), basic_mean(shared)
```

```text
  4c. reproducibility needs a FRESH generator each time
      no rng           :  99.6176  101.2565   <- global state, differs
      fresh default_rng:  99.2009   99.2009   <- reproducible
      ONE rng reused   :  99.2009  100.0243   <- state advances, differs
```

(The `no rng` numbers change on every run, by design.) Note the last line: the
*first* call with a reused generator matches the fresh one (99.2009), but the
second does not, because the generator's state advanced. To reproduce a
result, construct a new `default_rng(seed)` for each ensemble.

### 4d. An ensemble is a recipe; the two kinds differ on reuse

Evaluating **one** ensemble object twice and comparing the draws:

```text
  4d. reusing ONE ensemble object across two evaluate() calls
      DistributionEnsemble draws identical? False  <- RE-SAMPLES
      CrossProductEnsemble draws identical? True  <- STABLE
```

- `DistributionEnsemble` **re-samples** on every `evaluate()`.
- `CrossProductEnsemble` draws once and returns the same values each time.

So a shared `DistributionEnsemble` does *not* pin two scenarios to the same
noise — the draws move underneath you.

### 4e. `n_samples_per_combo` is per branch, and the weight is split

```python
CrossProductEnsemble(full_scenario, n_samples_per_combo=n, rng=np.random.default_rng(0))
```

```text
  4e. n_samples_per_combo is PER BRANCH, and the weight is split
      n=1: size= 4  weights=[0.792 0.198 0.008 0.002]
      n=3: size=12  weights=[0.264  0.264  0.264  0.066  0.066  0.066  0.0027 0.0027 0.0027 0.0007
 0.0007 0.0007]
```

With `n=1` there is one scenario per branch and the weights are exactly the
branch probabilities. With `n=3` every branch — even the 0.2 % one — gets 3
scenarios, and its probability is split three ways (`0.792 / 3 = 0.264`). That
equal allocation is what protects the rare `cat1=b` branch in Part 3. Cost =
branches × n × shape.

### 4f. `sample_across`, for plotting a weighted ensemble

A weighted ensemble cannot be histogrammed directly — rows have different
weights. `sample_across` resamples it into equally-weighted draws:

```python
drawn = sample_across(ensemble, [xc], total=2000, rng=np.random.default_rng(0))[xc]
```

The script measures what fraction of the resampled values come from the
`cat1=b` branch (identified as `xc > 500`). The true weight of that branch is
0.01:

```text
  4f. sample_across turns a WEIGHTED ensemble into plottable samples
      8 scenarios      ->  2000 samples, fraction from the cat1=b branch = 0.0090
      20000 scenarios  -> 20000 samples, fraction from the cat1=b branch = 0.4518
      (true cat1=b weight is 0.01)
```

Each scenario contributes `max(1, round(w_i × total))` samples. The 8-scenario
ensemble respects the weights (0.009 ≈ 0.01). The 20 000-scenario ensemble has
**more scenarios than `total`**, so the `max(1, …)` floor gives every scenario
exactly one sample: you get 20 000 samples instead of 2000, and since half of
the scenarios belong to `cat1=b` branches, ~45 % of the plot comes from a 1 %
branch. Use a *coarse* ensemble, or raise `total` well above the scenario
count.

---

## Choosing, in one line each

| Ensemble                 | Use when                                                                                        |
| ------------------------ | ----------------------------------------------------------------------------------------------- |
| none (`ensemble=None`) | no uncertainty: every input is fixed, so evaluate deterministically.                            |
| `DistributionEnsemble` | continuous noise, one marginal answer, and you want equal-weight draws for quantiles and tails. |
| `CrossProductEnsemble` | per-branch answers, rare branches, exact probabilities, or**any** conditional index.      |
| `PartitionedEnsemble`  | independent sources varied one at a time; the result is a grid reducible along either axis.     |

Decide by what you must *read off* the result. That, not speed, is what actually
separates them.

See also: [parameters_vs_scenarios_tutorial](../parameters_vs_scenarios_tutorial/Parameters_vs_Scenarios_Tutorial.md)
for the things you control from *outside* the model.
