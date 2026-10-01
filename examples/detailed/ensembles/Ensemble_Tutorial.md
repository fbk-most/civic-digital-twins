<!-- SPDX-License-Identifier: Apache-2.0 -->

# Choosing an ensemble: one model, four questions, four right answers

> Script: [`ensembles_compared.py`](ensembles_compared.py) — run it with
> `uv run python examples/detailed/ensembles/ensembles_compared.py`.
> Each section below matches a `PART N` banner in the script.

An ensemble turns a model's uncertainty into concrete numbers to compute over.
There are three built-in strategies plus an open protocol. They are **not**
interchangeable routes to the same number: each makes a different *kind* of
question answerable.

Everything uses deliberately abstract names, so that nothing but the ensemble
mechanics needs to be held in mind:

| name     | what it is                                                        |
|----------|-------------------------------------------------------------------|
| `x1, x2` | continuous uncertainties (`DistributionIndex`)                    |
| `cat1`   | categorical, outcomes `"a"` / `"b"`, **skewed** 0.99 / 0.01       |
| `cat2`   | categorical, outcomes `"p"` / `"q"`, 0.80 / 0.20                  |
| `xc`     | a distribution whose *parameters* depend on `cat1` and `cat2`     |
| `y`      | the model output                                                  |

The same model is grown and shrunk to suit each question:

| Part | Model                        | Ensemble               | Question                                   |
|------|------------------------------|------------------------|--------------------------------------------|
| 1    | `y = x1 * x2`                | `DistributionEnsemble` | the whole distribution of `y`              |
| 2    | add `cat1`, `cat2`, `xc`     | `CrossProductEnsemble` | `y` **per branch**, one branch rare        |
| 3    | back to `y = x1 * x2`        | `PartitionedEnsemble`  | how much does *each* source drive `y`?     |
| 4    | `y = f(x2)`, smooth, costly  | your own `AxisEnsemble`| exact expectation in 5 evaluations         |
| 5    | —                            | any                    | practical traps                            |

**The through-line:** pick by what you need to *read off* the result, not by
what is fastest. Cost differences are usually small; the difference in what you
can ask afterwards is not.

---

## PART 1 — `DistributionEnsemble`: continuous noise only

The simplest version of the model has no categorical structure, nothing to
enumerate, and no conditional dependence. We want the *shape* of `y`'s
distribution, so sampling is not a compromise here — it is the right tool.

```python
@define("basic")
class BasicModel(Model):
    ...
    def compute(self, inputs: Inputs) -> Outputs:
        return BasicModel.Outputs(y=Index("y", inputs.x1 * inputs.x2))

basic_res = Evaluation(basic_scenario).evaluate(
    ensemble=DistributionEnsemble(basic_scenario, size=50_000,
                                  rng=np.random.default_rng(0)),
)
basic_draws = np.asarray(basic_res[basic_model.outputs.y]).ravel()
```

First, the mean converges on the true value (100 × 1.0 = 100) as the sample
grows, at the usual Monte Carlo rate:

```text
  convergence of the mean (true value 100 x 1.0 = 100):
    size=    100 -> E[y] =  101.250
    size=   1000 -> E[y] =   98.934
    size=  10000 -> E[y] =  100.147
    size= 100000 -> E[y] =  100.010
```

Every draw carries the **same** weight (`1/size`). That is what makes plain
`np.percentile` correct here — no weighted-quantile machinery is needed — and
it means the whole distribution is available, not just the mean:

```python
for q in (5, 25, 50, 75, 95):
    print(f"    p{q:<2d} = {np.percentile(basic_draws, q):8.2f}")
print(f"    P(y > 150) = {float((basic_draws > 150).mean()):.4f}")
```

```text
  50000 equally-weighted draws -> the whole distribution:
    p5  =    50.74
    p25 =    77.10
    p50 =    97.57
    p75 =   120.25
    p95 =   157.41
    P(y > 150) = 0.0717
```

Quantiles, tail probabilities, histograms — anything you would compute on a
plain array of samples — come straight off `basic_draws`.

**Why this ensemble:** all the uncertainty is continuous, so there is no
structure to exploit. Use `DistributionEnsemble` when the question is "what does
the distribution of the output look like" and nothing categorical is in play.

---

## PART 2 — `CrossProductEnsemble`: per-branch answers and a rare branch

We extend the *same* model with two independent categoricals. The cross product
pairs every outcome of one with every outcome of the other: 2 × 2 = 4 branches.
The output must now be readable **per branch**, and the branch that dominates
`y` is the rarest one. That is what enumeration is for.

```python
cat1 = CategoricalIndex("cat1", {"a": 0.99, "b": 0.01})   # deliberately rare "b"
cat2 = CategoricalIndex("cat2", {"p": 0.80, "q": 0.20})
```

`xc` depends on *both* categoricals, so it is a **conditional** distribution:
the factory returns a different distribution per branch, and receives the
parents as keyword arguments named after them. (A conditional index is also
something `DistributionEnsemble` cannot represent at all — it raises.)

```python
def xc_dist(cat1: str, cat2: str):
    if cat1 == "a":
        return stats.lognorm(s=0.5, scale=20.0)       # the common case
    if cat2 == "p":
        return stats.lognorm(s=0.8, scale=1000.0)     # rare, moderate
    return stats.lognorm(s=0.8, scale=8000.0)         # rare AND extreme

xc = ConditionalDistributionIndex("xc", parents=[cat1, cat2], factory=xc_dist)
```

### 2a. A deliberately tiny ensemble

4 branches × 2 samples = 8 scenarios: small enough to print in full, which is
the only way the weights and the alignment between arrays become concrete.
Everything pedagogical in this part runs on these 8 rows.

There is no built-in `.by_branch()` helper. Instead **four arrays line up
one-to-one, one entry per scenario**, and you select with a boolean mask:

```python
(tiny_w,) = tiny_ens.ensemble_weights                            # weight
tiny_y = np.asarray(tiny_res[full_model.outputs.y]).ravel()      # output
tiny_c1 = np.asarray(tiny_res[cat1]).ravel()                     # label 1
tiny_c2 = np.asarray(tiny_res[cat2]).ravel()                     # label 2
```

```text
     i  cat1  cat2    weight          y
     0  a     p      0.39600     18.300
     1  a     p      0.39600     13.983
     2  a     q      0.09900     24.114
     3  a     q      0.09900     21.251
     4  b     p      0.00400    348.528
     5  b     p      0.00400   1277.023
     6  b     q      0.00100  17048.280
     7  b     q      0.00100  14566.889
    weights sum to 1.0000  <- over the WHOLE ensemble
```

Where each weight comes from: a branch's weight is the **product** of the two
categorical probabilities, then split across its replicates.

| branch            | product                | per replicate (÷ 2) |
|-------------------|------------------------|---------------------|
| `cat1=a, cat2=p`  | 0.99 × 0.80 = 0.7920   | 0.39600             |
| `cat1=a, cat2=q`  | 0.99 × 0.20 = 0.1980   | 0.09900             |
| `cat1=b, cat2=p`  | 0.01 × 0.80 = 0.0080   | 0.00400             |
| `cat1=b, cat2=q`  | 0.01 × 0.20 = 0.0020   | 0.00100             |

That product is what "cross product" means, and it is why adding a categorical
**multiplies** the branch count rather than adding to it.

### 2b. Weights and renormalisation — the part that is easy to get wrong

The weights sum to 1.0 across the **whole** ensemble. Selecting one branch
therefore gives weights that sum to that branch's *probability*, not to 1. A
conditional mean must divide by that sum:

$$E[y \mid \text{branch}] = \frac{\sum w_i\, y_i}{\sum w_i} \quad \text{over that branch only}$$

That division *is* the renormalisation, and `np.average(values, weights=w)` does
it internally. The script computes `E[y | cat1=b]` five ways:

```python
b_mask = tiny_c1 == "b"
w_sub = tiny_w[b_mask]
y_sub = tiny_y[b_mask]

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

The three correct forms agree exactly. You can verify by hand from rows 4–7 of
the table above: `(0.004·348.5 + 0.004·1277.0 + 0.001·17048.3 + 0.001·14566.9) / 0.01 = 3811.7`.

- `sum(w*y)` without division is scaled down by `sum(w) = 0.01`. That is the
  branch's **contribution** to the overall mean, not the branch's mean.
- `y_sub.mean()` treats `cat2=q` as equally likely as `cat2=p`, because the mask
  holds the same *number* of rows from each — though `p` is four times more
  probable. **Sample counts are not probabilities.**

### 2c. Joint and marginal views

A **joint** branch uses one mask per categorical, combined with `&`:

```python
mask = (tiny_c1 == c1_outcome) & (tiny_c2 == c2_outcome)
branch_weights = tiny_w[mask]
branch_values = tiny_y[mask]
np.average(branch_values, weights=branch_weights)
```

```text
  JOINT branches -- one mask per categorical, combined with &:
    cat1  cat2    n  P (exact)   E[y|branch]
    a     p       2     0.7920        16.142
    a     q       2     0.1980        22.683
    b     p       2     0.0080       812.776
    b     q       2     0.0020     15807.585
```

Each joint branch holds 2 rows; `P (exact)` is the sum of their weights, i.e.
the product of the two categorical probabilities. The rarest branch has by far
the largest conditional mean.

A **marginal** constrains only one categorical; the other varies freely inside
the mask, mixed according to its own weights (see `marginal_report`). Those
weights are *uneven*, which is where `np.average`'s renormalisation does real
work.

```python
mask = label_array == outcome          # ONLY this categorical is constrained
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

Note how `cat1=b`'s marginal (3811.7) is the same number computed in 2b, and
how the rare `cat1=b` branch, though it is only 1 % of the weight, pulls the
`cat2=q` marginal up to 180.5 — far above any `cat1=a` value.

Both marginals partition the whole ensemble, so each recomposes to the same
overall mean. Which one you report is a modelling decision: marginalising over
the *rare* variable barely moves the rows, while marginalising over the
*common* one changes them a lot.

### 2d. Conditional vs plain distribution index

```text
    cat1  cat2      xc (conditional)         x2 (plain)
    a     p              [21.3 18.7]      [0.859 0.747]
    a     q              [27.5 21.1]      [0.875 1.008]
    b     p          [ 651.5 1335.5]      [0.535 0.956]
    b     q        [22706.3 17066.3]      [0.751 0.854]
```

- `xc` is **conditional** — the factory hands each branch its own
  distribution, so the magnitudes jump by orders of magnitude across rows.
- `x2` is **plain** — one N(1.0, 0.2), redrawn independently in every branch.
  With only 2 draws per branch the sample scatter is visible, but no row is
  systematically higher than another.

Both interact with the categoricals in `compute()` — `y` is their product. The
difference is *where* the branch dependence lives: in the distribution itself
(conditional) or only in the arithmetic (plain).

### 2e. The sanity check: law of total expectation

The branches decompose the overall mean **exactly**:

$$E[y] = \sum_{\text{branches}} P(\text{branch}) \cdot E[y \mid \text{branch}]$$

```python
contributions.append(
    branch_weights.sum() * np.average(branch_values, weights=branch_weights)
)
...
sum(contributions)  ==  tiny_res.expected_value(full_model.outputs.y)
```

```text
  check (law of total expectation, on the 8 rows above):
    sum of P(branch) * E[y|branch] = 55.3929
    expected_value()               = 55.3929
```

Note that the overall mean (55.4) is more than three times the common-case
value (~16–23): almost all of the gap comes from the `cat1=b` branches that
carry 1 % of the weight. Run this check whenever you slice a result by branch: if it fails, your masks do
not partition the ensemble or your weights are mishandled.

### 2f. The reportable run

The same code with `n_samples_per_combo=5000` (20 000 scenarios) produces real
numbers:

```text
    cat1  cat2   P (exact)   E[y|branch]         p99
    a     p         0.7920         22.63       65.89
    a     q         0.1980         22.85       67.42
    b     p         0.0080       1381.25     6294.73
    b     q         0.0020      10904.65    52350.07
    E[y] overall = 55.3057
```

`P (exact)` is **identical** to the tiny run — probabilities come from
enumeration, not from sampling. Only the conditional means and quantiles needed
the extra draws.

**Why this ensemble:** the `cat1=b, cat2=q` branch holds 0.2 % of the
probability and dominates `y`. Enumeration gives it a *guaranteed* 5000 samples
and an *exact* weight, so its mean and p99 are solid. A sampler spending the
same 20 000 evaluations would land roughly 40 there — enough for a rough mean,
far too few for a 99th percentile — and the branch probabilities would
themselves be estimates. Shrink the budget and that corner vanishes, at which
point the conditional answer does not exist at all.

Sampling is also simply unavailable here: `xc` is conditional, and
`DistributionEnsemble` rejects conditional indexes outright.

---

## PART 3 — `PartitionedEnsemble`: which uncertainty drives the output?

Back to `y = x1 * x2`, but a different question: not "what is `y`'s
distribution" but "which of `x1` and `x2` drives it more" — because you might
pay to measure one of them better.

A flat sample cannot answer that: every draw welds one `x1` value to one `x2`
value, and you can never un-mix them. Separate axes keep them distinguishable:

```python
part_ens = PartitionedEnsemble(
    basic_scenario,
    axes=[
        EnsembleAxisSpec("x1_axis", indexes=[x1], size=60),
        EnsembleAxisSpec("x2_axis", indexes=[x2], size=40),
    ],
    rng=np.random.default_rng(0),
)
grid = np.asarray(part_res[basic_model.outputs.y])   # shape (60, 40): a GRID
```

```text
  ensemble axes : ['x1_axis', 'x2_axis']
  raw result    : shape (60, 40)  <- a GRID, not a flat sample
  draws used    : 60 + 40 = 100, covering 2400 combinations
  E[y]          : 103.70   (true 100 x 1.0 = 100)
```

The raw result is not a flat list of 100 draws but a 60 × 40 grid: every `x1`
draw is combined with every `x2` draw. Row `i` holds `x1[i]` fixed; column `j`
holds `x2[j]` fixed.

Marginalising one axis at a time isolates each source:

```python
by_x1 = grid.mean(axis=1)   # average out x2
by_x2 = grid.mean(axis=0)   # average out x1
```

```text
  marginalising ONE axis at a time isolates each driver:
    vary x1, average over x2 -> (60,), sd 22.83
    vary x2, average over x1 -> (40,), sd 21.44
```

`by_x1` shows how much `y` moves when only `x1` varies (with `x2` averaged
out), and vice versa. The two spreads are close here, so neither dominates —
measuring either one better would help about equally. That makes sense: `x1`
has a 25 % relative spread (sd 25 around 100), `x2` has 20 % (sd 0.2 around
1.0), and `y` is their product. It is an actionable answer a single flat axis
simply cannot produce.

**Why this ensemble:** the grid shape *is* the answer, and it is cheap — 100
draws cover 2400 combinations because the axes broadcast. The assumption it
encodes is **independence**: if the two were correlated, this factorial grid
would invent combinations that never occur, and a conditional index inside a
`CrossProductEnsemble` would be the honest choice instead.

---

## PART 4 — your own `AxisEnsemble`: exact integration of a smooth function

`y` is now a smooth function of `x2` alone, and each evaluation is expensive.
We want the expectation in as few evaluations as possible.

Monte Carlo converges at 1/√n. For a smooth function of a normal, Gaussian
quadrature is exact with a handful of nodes. The library ships no quadrature
ensemble, so we write one. `AxisEnsemble` is a structural `Protocol`: three
members, no inheritance, nothing to register.

```python
class QuadratureEnsemble:
    def __init__(self, index, mu: float, sigma: float, n: int = 5):
        nodes, weights = np.polynomial.hermite_e.hermegauss(n)
        self._index = index
        self._values = mu + sigma * nodes
        self._weights = weights / weights.sum()     # MUST sum to 1.0
        self._axis = Axis("quadrature", ENSEMBLE)   # MUST be role ENSEMBLE

    @property
    def ensemble_axes(self):
        return (self._axis,)

    @property
    def ensemble_weights(self):
        return (self._weights,)

    def assignments(self):
        return {self._index: self._values}
```

No inheritance is needed — the class satisfies the protocol structurally:

```text
  isinstance(quad, AxisEnsemble) -> True  (runtime_checkable Protocol)
```

With `y = 1000 · x2²`, the true value is `1000 · (1.0² + 0.2²) = 1040`:

```text
    quadrature,     5 evaluations -> 1040.000000  EXACT
    Monte Carlo,    500 evaluations -> 1030.368315  (error -9.631685)
    Monte Carlo,   5000 evaluations -> 1037.820332  (error -2.179668)
    Monte Carlo,  50000 evaluations -> 1040.451679  (error +0.451679)
```

**Why write your own:** 5 model evaluations against 50 000, and the 5 are exact.
When evaluations are expensive and the response is smooth, that is decisive —
and unreachable with the built-in ensembles.

**The contract**, if you write one:

- **Weights must sum to 1.0.** `expected_value()` is a weighted average and
  silently mis-scales otherwise.
- **Every array from `assignments()` must carry all ensemble dims, in order**,
  size 1 where the index does not vary along an axis. A wrong length surfaces
  as a raw numpy broadcast error at `evaluate()` time, not as a helpful message
  about the protocol.
- **Axes must have role `ENSEMBLE`** — that is what marks them for integration.

Latin hypercube, historical records replayed as scenarios, or a fixed list of
hand-picked cases all drop in the same way.

---

## PART 5 — traps that apply whichever ensemble you chose

### 5a. `max_categorical_size` silently abandons exactness above 20

The model uses a categorical `cat_big` with 25 outcomes `L0`…`L24`, where
outcome `Li` has weight proportional to `i + 1` and contributes the value `i`
to `y`. Its true mean is exactly 16. Each setting is evaluated with 4 different
seeds:

```python
CrossProductEnsemble(many_scenario, max_categorical_size=mx,
                     rng=np.random.default_rng(sd))
```

```text
  5a. max_categorical_size (support 25, true mean 16)
      max_categorical_size= 20 -> size=20 SAMPLED (noisy)     [15.7  15.55 15.4  15.65]
      max_categorical_size= 25 -> size=25 enumerated (exact)  [16. 16. 16. 16.]
```

With the default limit of 20, a 25-outcome categorical is **sampled** rather
than enumerated: 20 random draws, so each seed gives a different, wrong mean.
Raise the limit to 25 and every outcome is enumerated with its exact weight —
the answer is exactly 16 regardless of seed. Nothing warns you about the
switch; set `max_categorical_size` above your largest support when you need
exactness.

### 5b. Support-only categoricals cannot be enumerated

```python
cat_bare = CategoricalIndex("cat_bare", ["p", "q"])   # no weights
CrossProductEnsemble(Scenario(...))                    # -> ValueError
```

```text
  5b. a categorical built from a bare list has no weights
      raises: CategoricalIndex 'cat_bare' was constructed without weights (support-only); '.outcomes' ...
```

Enumeration needs a probability per outcome, and a bare list provides none.
Give weights at construction, or via a `Scenario` dict override.

### 5c. Reproducibility needs a *fresh* generator each time

The helper `basic_mean(generator)` runs the Part 1 model with 400 draws and
returns `E[y]`. Each line calls it twice:

```python
basic_mean(None), basic_mean(None)
basic_mean(np.random.default_rng(0)), basic_mean(np.random.default_rng(0))
shared = np.random.default_rng(0)
basic_mean(shared), basic_mean(shared)
```

```text
  5c. reproducibility needs a FRESH generator each time
      no rng           :  99.6176  101.2565   <- global state, differs
      fresh default_rng:  99.2009   99.2009   <- reproducible
      ONE rng reused   :  99.2009  100.0243   <- state advances, differs
```

(The `no rng` numbers change on every run, by design.) Note the last line: the
*first* call with a reused generator matches the fresh one (99.2009), but the
second does not, because the generator's state advanced. To reproduce a
result, construct a new `default_rng(seed)` for each ensemble.

### 5d. An ensemble is a recipe; the two kinds differ on reuse

Evaluating **one** ensemble object twice and comparing the draws:

```text
  5d. reusing ONE ensemble object across two evaluate() calls
      DistributionEnsemble draws identical? False  <- RE-SAMPLES
      CrossProductEnsemble draws identical? True  <- STABLE
```

- `DistributionEnsemble` **re-samples** on every `evaluate()`.
- `CrossProductEnsemble` draws once and returns the same values each time.

So a shared `DistributionEnsemble` does *not* pin two scenarios to the same
noise — the draws move underneath you.

### 5e. `n_samples_per_combo` is per branch, and the weight is split

```python
CrossProductEnsemble(full_scenario, n_samples_per_combo=n, rng=np.random.default_rng(0))
```

```text
  5e. n_samples_per_combo is PER BRANCH, and the weight is split
      n=1: size= 4  weights=[0.792 0.198 0.008 0.002]
      n=3: size=12  weights=[0.264  0.264  0.264  0.066  0.066  0.066  0.0027 0.0027 0.0027 0.0007
 0.0007 0.0007]
```

With `n=1` there is one scenario per branch and the weights are exactly the
branch probabilities. With `n=3` every branch — even the 0.2 % one — gets 3
scenarios, and its probability is split three ways (`0.792 / 3 = 0.264`). That
equal allocation is what protects the rare `cat1=b` branch in Part 2. Cost =
branches × n × shape.

### 5f. `sample_across`, for plotting a weighted ensemble

A weighted ensemble cannot be histogrammed directly — rows have different
weights. `sample_across` resamples it into equally-weighted draws:

```python
drawn = sample_across(ensemble, [xc], total=2000, rng=np.random.default_rng(0))[xc]
```

The script measures what fraction of the resampled values come from the
`cat1=b` branch (identified as `xc > 500`). The true weight of that branch is
0.01:

```text
  5f. sample_across turns a WEIGHTED ensemble into plottable samples
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

| Ensemble               | Use when                                                                                             |
|------------------------|------------------------------------------------------------------------------------------------------|
| `DistributionEnsemble` | continuous noise, one marginal answer, and you want equal-weight draws for quantiles and tails.      |
| `CrossProductEnsemble` | per-branch answers, rare branches, exact probabilities, or **any** conditional index.                |
| `PartitionedEnsemble`  | independent sources varied one at a time; the result is a grid reducible along either axis.          |
| your own               | a better integration rule for your structure; three members, no inheritance.                         |

Decide by what you must *read off* the result. That, not speed, is what actually
separates them.

See also: [parameters_vs_scenarios](../parameters_vs_scenarios/Parameters_vs_Scenarios_Tutorial.md)
for the things you control from *outside* the model.
