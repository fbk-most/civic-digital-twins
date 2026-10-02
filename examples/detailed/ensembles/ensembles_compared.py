# SPDX-License-Identifier: Apache-2.0

"""Choosing an ensemble: ONE model, four questions, four right answers.

Narrative and explanations: see Ensemble_Tutorial.md in this directory.
Each "PART N" banner below matches a section of the same name there.

    x1, x2    continuous uncertainties (DistributionIndex)
    cat1      a categorical uncertainty, outcomes "a" / "b", SKEWED 0.99/0.01
    cat2      a categorical uncertainty, outcomes "p" / "q", 0.80/0.20
    xc        a distribution whose PARAMETERS depend on cat1 and cat2
    y         the model output
"""

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    AxisEnsemble,
    CategoricalIndex,
    ConditionalDistributionIndex,
    CrossProductEnsemble,
    DistributionEnsemble,
    DistributionIndex,
    EnsembleAxisSpec,
    Evaluation,
    Index,
    Model,
    PartitionedEnsemble,
    Scenario,
    define,
    inputs,
    outputs,
    sample_across,
)
from civic_digital_twins.dt_model.axes import ENSEMBLE, Axis

# =============================================================================
# PART 0 -- no uncertainty: no ensemble at all
# =============================================================================
print("=" * 74)
print("PART 0 -- no uncertainty: evaluate without an ensemble")
print("=" * 74)


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

print(f"\n  abstract indexes : {list(cert_scenario.abstract_indexes())}")
print(f"  y                : {float(cert_y)}   (shape {cert_y.shape})")


# =============================================================================
# PART 1 -- DistributionEnsemble: continuous noise only
# =============================================================================
print("\n" + "=" * 74)
print("PART 1 -- DistributionEnsemble: continuous noise, distribution wanted")
print("=" * 74)


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

print("\n  convergence of the mean (true value 100 x 1.0 = 100):")
for size in (100, 1_000, 10_000, 100_000):
    conv_res = Evaluation(basic_scenario).evaluate(
        ensemble=DistributionEnsemble(basic_scenario, size=size,
                                      rng=np.random.default_rng(0)),
    )
    print(f"    size={size:7d} -> E[y] = "
          f"{float(conv_res.expected_value(basic_model.outputs.y)):8.3f}")

# Every draw carries the SAME weight (1/size), so plain numpy percentiles work.
print(f"\n  {basic_draws.shape[0]} equally-weighted draws -> the whole distribution:")
for q in (5, 25, 50, 75, 95):
    print(f"    p{q:<2d} = {np.percentile(basic_draws, q):8.2f}")
print(f"    P(y > 150) = {float((basic_draws > 150).mean()):.4f}")


# =============================================================================
# PART 2 -- PartitionedEnsemble: which uncertainty drives the output?
# =============================================================================
print("\n" + "=" * 74)
print("PART 2 -- PartitionedEnsemble: attributing the spread to each source")
print("=" * 74)

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

print(f"\n  ensemble axes : {[a.name for a in part_ens.ensemble_axes]}")
print(f"  raw result    : shape {grid.shape}  <- a GRID, not a flat sample")
print(f"  draws used    : 60 + 40 = 100, covering {60 * 40} combinations")
print(f"  E[y]          : "
      f"{float(part_res.expected_value(basic_model.outputs.y)):.2f}"
      f"   (true 100 x 1.0 = 100)")

# Marginalise one axis at a time to isolate each source.
by_x1 = grid.mean(axis=1)   # average out x2
by_x2 = grid.mean(axis=0)   # average out x1
print("\n  marginalising ONE axis at a time isolates each driver:")
print(f"    vary x1, average over x2 -> {by_x1.shape}, sd {by_x1.std():5.2f}")
print(f"    vary x2, average over x1 -> {by_x2.shape}, sd {by_x2.std():5.2f}")


# =============================================================================
# PART 3 -- CrossProductEnsemble: the same model, plus categorical structure
# =============================================================================
print("\n" + "=" * 74)
print("PART 3 -- CrossProductEnsemble: per-branch answers, and a rare branch")
print("=" * 74)

cat1 = CategoricalIndex("cat1", {"a": 0.99, "b": 0.01})   # deliberately rare "b"
cat2 = CategoricalIndex("cat2", {"p": 0.80, "q": 0.20})


# The Part 1 model, scaled by the branch. (cat == "b") is 1 in that branch
# and 0 otherwise, so the categoricals enter the arithmetic directly.
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


# Create model and scenario as usual
cat_model = CatModel(inputs=CatModel.Inputs(cat1=cat1, cat2=cat2, x1=x1, x2=x2))
cat_scenario = Scenario(cat_model)

# True values: E[y | branch] = 100 x factor (100, 100, 5000, 40000), and
# E[y] = 100 x (0.99 x 1 + 0.008 x 50 + 0.002 x 400) = 219.

# --- 3a. Same model, same budget (1000 rows), two ensembles -----------------
# (i) DistributionEnsemble: SAMPLES the categoricals.
de_ens = DistributionEnsemble(cat_scenario, size=1_000,       # 1000 independent draws for each sampleable quantity
                              rng=np.random.default_rng(0))

de_res = Evaluation(cat_scenario).evaluate(ensemble=de_ens)   # Evaluate to get result

# The result is ONE flat axis of rows. Four arrays line up one-to-one:
de_y = np.asarray(de_res[cat_model.outputs.y]).ravel()  # y of each row
(de_w,) = de_ens.ensemble_weights                       # weight of each row
de_c1 = np.asarray(de_res[cat1]).ravel()                # cat1 label of each row. cat1 is an Index
de_c2 = np.asarray(de_res[cat2]).ravel()                # cat2 label of each row

# A branch is a mask on the labels. Here: the rare cat1=b, cat2=q branch.
de_bq = (de_c1 == "b") & (de_c2 == "q")         # create mask

de_bq_n = int(de_bq.sum())                      # samples in the branch

de_bq_y = de_y[de_bq]                           # values of 'y' in the branch

de_bq_prob = float(de_w[de_bq].sum())                             # its probability
de_bq_mean = float(np.average(de_y[de_bq], weights=de_w[de_bq]))  # E[y | branch]

# The same mask, for every branch.
print(f"\n  DistributionEnsemble: {len(de_w)} rows, "
      f"E[y] = {float(de_res.expected_value(cat_model.outputs.y)):.2f}   (true 219)")
print(f"\n    {'cat1':5s} {'cat2':5s} {'rows':>5s} {'P':>7s} {'E[y|branch]':>12s}  True branch mean")
for c1_outcome, c2_outcome, true_mean in (("a", "p", 100), ("a", "q", 100),
                                          ("b", "p", 5_000), ("b", "q", 40_000)):
    mask = (de_c1 == c1_outcome) & (de_c2 == c2_outcome)
    print(f"    {c1_outcome:5s} {c2_outcome:5s} {mask.sum():5d} "
          f"{de_w[mask].sum():7.4f} "
          f"{np.average(de_y[mask], weights=de_w[mask]):12.2f} {true_mean:17d}")

# (ii) CrossProductEnsemble: ENUMERATES the branches, 250 rows each.
cp_ens = CrossProductEnsemble(cat_scenario, n_samples_per_combo=250,  # 1000 samples divided into the 4 branches = 250
                              rng=np.random.default_rng(0))

cp_res = Evaluation(cat_scenario).evaluate(ensemble=cp_ens)   # Evaluate to get result

# The result is ONE flat axis of rows. Four arrays line up one-to-one:
cp_y = np.asarray(cp_res[cat_model.outputs.y]).ravel()  # y of each row
(cp_w,) = cp_ens.ensemble_weights                       # weight of each row
cp_c1 = np.asarray(cp_res[cat1]).ravel()                # cat1 label of each row
cp_c2 = np.asarray(cp_res[cat2]).ravel()                # cat2 label of each row

# A branch is a mask on the labels. Here: the rare cat1=b, cat2=q branch.
cp_bq = (cp_c1 == "b") & (cp_c2 == "q")         # create mask

cp_bq_n = int(cp_bq.sum())                      # samples in the branch

cp_bq_y = cp_y[cp_bq]                           # values of 'y' in the branch

cp_bq_prob = float(cp_w[cp_bq].sum())                             # its probability
cp_bq_mean = float(np.average(cp_y[cp_bq], weights=cp_w[cp_bq]))  # E[y | branch]

# The same mask, for every branch.
print(f"\n  CrossProductEnsemble: {len(cp_w)} rows, "
      f"E[y] = {float(cp_res.expected_value(cat_model.outputs.y)):.2f}   (true 219)")
print(f"\n    {'cat1':5s} {'cat2':5s} {'rows':>5s} {'P':>7s} {'E[y|branch]':>12s}  True branch mean")
for c1_outcome, c2_outcome, true_mean in (("a", "p", 100), ("a", "q", 100),
                                          ("b", "p", 5_000), ("b", "q", 40_000)):
    mask = (cp_c1 == c1_outcome) & (cp_c2 == c2_outcome)
    print(f"    {c1_outcome:5s} {c2_outcome:5s} {mask.sum():5d} "
          f"{cp_w[mask].sum():7.4f} "
          f"{np.average(cp_y[mask], weights=cp_w[mask]):12.2f} {true_mean:17d}")


# --- 3b. Add xc: a distribution that depends on the branch ------------------
# xc depends on BOTH categoricals: a CONDITIONAL distribution. The factory
# receives parents as keyword arguments named after them.
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

print("\n  DistributionEnsemble on the FULL model (xc is conditional):")
try:
    DistributionEnsemble(full_scenario, size=1_000, rng=np.random.default_rng(0))
except ValueError as err:
    print(f"    ValueError: {err}")

# --- 3c. FullModel per branch: 5000 samples per branch -----------------------
full_ens = CrossProductEnsemble(full_scenario, n_samples_per_combo=5000,
                                rng=np.random.default_rng(0))
full_res = Evaluation(full_scenario).evaluate(ensemble=full_ens)
(full_w,) = full_ens.ensemble_weights
full_y = np.asarray(full_res[full_model.outputs.y]).ravel()
full_c1 = np.asarray(full_res[cat1]).ravel()
full_c2 = np.asarray(full_res[cat2]).ravel()

print(f"\n  FullModel: {len(full_ens)} scenarios (4 branches x 5000), "
      f"one mask per branch:")
print(f"    {'cat1':5s} {'cat2':5s} {'P (exact)':>10s} {'E[y|branch]':>13s} "
      f"{'p99':>11s}")
for c1_outcome in ("a", "b"):
    for c2_outcome in ("p", "q"):
        mask = (full_c1 == c1_outcome) & (full_c2 == c2_outcome)
        branch_weights = full_w[mask]
        branch_values = full_y[mask]
        print(f"    {c1_outcome:5s} {c2_outcome:5s} "
              f"{branch_weights.sum():10.4f} "
              f"{np.average(branch_values, weights=branch_weights):13.2f} "
              f"{np.percentile(branch_values, 99):11.2f}")
print(f"    E[y] overall = "
      f"{float(full_res.expected_value(full_model.outputs.y)):.4f}")

# --- 3d. When a mask mixes branches: marginals, on a tiny ensemble ----------
# 4 branches x 2 samples = 8 scenarios, small enough to print.
tiny_ens = CrossProductEnsemble(full_scenario, n_samples_per_combo=2,
                                rng=np.random.default_rng(0))
tiny_res = Evaluation(full_scenario).evaluate(ensemble=tiny_ens)

(tiny_w,) = tiny_ens.ensemble_weights                            # weight
tiny_y = np.asarray(tiny_res[full_model.outputs.y]).ravel()      # output
tiny_c1 = np.asarray(tiny_res[cat1]).ravel()                     # label 1
tiny_c2 = np.asarray(tiny_res[cat2]).ravel()                     # label 2

# A mask on cat1 ONLY holds rows of several branches, with different weights.
#     E[y | cat1=b] = sum(w_i * y_i) / sum(w_i)        over the masked rows
b_mask = tiny_c1 == "b"
w_sub = tiny_w[b_mask]
y_sub = tiny_y[b_mask]

print("\n  computing E[y | cat1=b] from those 4 rows:")
print(f"    np.average(y, weights=w)         = "
      f"{np.average(y_sub, weights=w_sub):12.4f}")
print(f"    sum(w*y) / sum(w)                = "
      f"{np.sum(w_sub * y_sub) / np.sum(w_sub):12.4f}")
print(f"    normalise w first, then sum(w*y) = "
      f"{np.sum((w_sub / np.sum(w_sub)) * y_sub):12.4f}")
print(f"    sum(w*y)  [NO division]          = "
      f"{np.sum(w_sub * y_sub):12.4f}  <- WRONG (branch contribution)")
print(f"    y_sub.mean()  [no weights]       = {y_sub.mean():12.4f}  <- WRONG (counts != probabilities)")

# MARGINAL: the mask constrains ONE categorical; the other varies freely inside
# it with UNEVEN weights, so np.average's renormalisation matters.
print("\n  MARGINAL over cat2 -- mask on cat1 ONLY:")
print(f"    {'cat1':5s} {'n':>3s} {'P (exact)':>10s} {'E[y]':>12s}")
for c1_outcome in ("a", "b"):
    mask = tiny_c1 == c1_outcome          # ONLY cat1 is constrained
    print(f"    {c1_outcome:5s} {mask.sum():3d} {tiny_w[mask].sum():10.4f} "
          f"{np.average(tiny_y[mask], weights=tiny_w[mask]):12.3f}")

print("\n  MARGINAL over cat1 -- mask on cat2 ONLY:")
print(f"    {'cat2':5s} {'n':>3s} {'P (exact)':>10s} {'E[y]':>12s}")
for c2_outcome in ("p", "q"):
    mask = tiny_c2 == c2_outcome          # ONLY cat2 is constrained
    print(f"    {c2_outcome:5s} {mask.sum():3d} {tiny_w[mask].sum():10.4f} "
          f"{np.average(tiny_y[mask], weights=tiny_w[mask]):12.3f}")

# --- 3e. Law of total expectation: the branches decompose the mean EXACTLY ---
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
print("\n  check (law of total expectation, on the 8 rows above):")
print(f"    sum of P(branch) * E[y|branch] = {sum(contributions):.4f}")
print(f"    expected_value()               = {tiny_overall:.4f}")


# =============================================================================
# PART 4 -- your own AxisEnsemble: exact integration of a smooth function
# =============================================================================
print("\n" + "=" * 74)
print("PART 4 -- a custom AxisEnsemble: exact where sampling only approximates")
print("=" * 74)


class QuadratureEnsemble:
    """Integrate a normal uncertainty exactly, with n quadrature nodes."""

    def __init__(self, index, mu: float, sigma: float, n: int = 5):
        # Probabilists' Hermite nodes integrate against a standard normal;
        # shift and scale them onto N(mu, sigma).
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
        # One axis, one index varying along it -> shape (n,).
        return {self._index: self._values}


@define("smooth")
class SmoothModel(Model):
    @inputs
    class Inputs:
        x2: DistributionIndex

    @outputs
    class Outputs:
        y: Index

    def compute(self, inputs: Inputs) -> Outputs:
        # E[y] = 1000 * E[x2^2] = 1000 * (1.0^2 + 0.2^2) = 1040
        return SmoothModel.Outputs(
            y=Index("y", inputs.x2 * inputs.x2 * 1000.0)
        )


smooth_model = SmoothModel(inputs=SmoothModel.Inputs(x2=x2))
smooth_scenario = Scenario(smooth_model)

quad = QuadratureEnsemble(x2, mu=1.0, sigma=0.2, n=5)
print(f"\n  isinstance(quad, AxisEnsemble) -> {isinstance(quad, AxisEnsemble)}"
      f"  (runtime_checkable Protocol)")

quad_res = Evaluation(smooth_scenario).evaluate(ensemble=quad)
print("\n  E[y], true value 1000 * (1.0^2 + 0.2^2) = 1040")
print(f"    quadrature,     5 evaluations -> "
      f"{float(quad_res.expected_value(smooth_model.outputs.y)):11.6f}  EXACT")
for size in (500, 5_000, 50_000):
    mc_res = Evaluation(smooth_scenario).evaluate(
        ensemble=DistributionEnsemble(smooth_scenario, size=size,
                                      rng=np.random.default_rng(0)),
    )
    got = float(mc_res.expected_value(smooth_model.outputs.y))
    print(f"    Monte Carlo, {size:6d} evaluations -> {got:11.6f}  "
          f"(error {got - 1040.0:+.6f})")


# =============================================================================
# PART 5 -- traps that apply whichever ensemble you chose
# =============================================================================
print("\n" + "=" * 74)
print("PART 5 -- practical traps, independent of which ensemble you picked")
print("=" * 74)

# --- 5a. max_categorical_size silently abandons exactness above 20 ----------
levels = {f"L{i}": (i + 1) for i in range(25)}
_total = sum(levels.values())
levels = {k: v / _total for k, v in levels.items()}
TRUE_LEVEL = sum(float(k[1:]) * p for k, p in levels.items())
cat_big = CategoricalIndex("cat_big", levels)


@define("manylevels")
class ManyLevels(Model):
    @inputs
    class Inputs:
        cat_big: CategoricalIndex

    @outputs
    class Outputs:
        y: Index

    def compute(self, inputs: Inputs) -> Outputs:
        acc = 0.0
        for key in levels:
            acc = acc + float(key[1:]) * (inputs.cat_big == key)
        return ManyLevels.Outputs(y=Index("y", acc))


many_model = ManyLevels(inputs=ManyLevels.Inputs(cat_big=cat_big))
many_scenario = Scenario(many_model)
print(f"\n  5a. max_categorical_size (support 25, true mean {TRUE_LEVEL:g})")
for mx in (20, 25):
    vals = [
        float(Evaluation(many_scenario).evaluate(
            ensemble=CrossProductEnsemble(many_scenario, max_categorical_size=mx,
                                          rng=np.random.default_rng(sd)),
        ).expected_value(many_model.outputs.y))
        for sd in range(4)
    ]
    size = len(CrossProductEnsemble(many_scenario, max_categorical_size=mx))
    tag = "SAMPLED (noisy)" if mx < 25 else "enumerated (exact)"
    print(f"      max_categorical_size={mx:3d} -> size={size:2d} {tag:19s} "
          f"{np.round(vals, 2)}")

# --- 5b. support-only categoricals cannot be enumerated ---------------------
print("\n  5b. a categorical built from a bare list has no weights")
try:
    cat_bare = CategoricalIndex("cat_bare", ["p", "q"])

    @define("bare")
    class BareModel(Model):
        @inputs
        class Inputs:
            cat_bare: CategoricalIndex

        @outputs
        class Outputs:
            y: Index

        def compute(self, inputs: Inputs) -> Outputs:
            return BareModel.Outputs(y=Index("y", 1.0 * (inputs.cat_bare == "p")))

    CrossProductEnsemble(Scenario(BareModel(inputs=BareModel.Inputs(cat_bare=cat_bare))))
except ValueError as exc:
    print(f"      raises: {str(exc)[:88]}...")

# --- 5c. rng semantics ------------------------------------------------------
print("\n  5c. reproducibility needs a FRESH generator each time")


def basic_mean(generator):
    return float(Evaluation(basic_scenario).evaluate(
        ensemble=DistributionEnsemble(basic_scenario, size=400, rng=generator),
    ).expected_value(basic_model.outputs.y))


print(f"      no rng           : {basic_mean(None):8.4f}  {basic_mean(None):8.4f}"
      f"   <- global state, differs")
print(f"      fresh default_rng: {basic_mean(np.random.default_rng(0)):8.4f}  "
      f"{basic_mean(np.random.default_rng(0)):8.4f}   <- reproducible")
shared = np.random.default_rng(0)
print(f"      ONE rng reused   : {basic_mean(shared):8.4f}  {basic_mean(shared):8.4f}"
      f"   <- state advances, differs")

# --- 5d. an ensemble is a recipe; the two kinds differ on reuse -------------
print("\n  5d. reusing ONE ensemble object across two evaluate() calls")
reuse_de = DistributionEnsemble(basic_scenario, size=800,
                                rng=np.random.default_rng(0))
reuse_r1 = Evaluation(basic_scenario).evaluate(ensemble=reuse_de)
reuse_r2 = Evaluation(basic_scenario).evaluate(ensemble=reuse_de)
print(f"      DistributionEnsemble draws identical? "
      f"{np.allclose(np.ravel(reuse_r1[x2]), np.ravel(reuse_r2[x2]))}"
      f"  <- RE-SAMPLES")
reuse_x1 = Evaluation(full_scenario).evaluate(ensemble=tiny_ens)
reuse_x2 = Evaluation(full_scenario).evaluate(ensemble=tiny_ens)
print(f"      CrossProductEnsemble draws identical? "
      f"{np.allclose(np.ravel(reuse_x1[xc]), np.ravel(reuse_x2[xc]))}"
      f"  <- STABLE")

# --- 5e. sampling budget and cost ------------------------------------------
print("\n  5e. n_samples_per_combo is PER BRANCH, and the weight is split")
for n in (1, 3):
    be = CrossProductEnsemble(full_scenario, n_samples_per_combo=n,
                              rng=np.random.default_rng(0))
    (bwt,) = be.ensemble_weights
    print(f"      n={n}: size={len(be):2d}  weights={np.round(bwt, 4)}")

# --- 5f. sample_across, for plotting a weighted ensemble --------------------
# Each scenario contributes max(1, round(w_i x total)) samples.
print("\n  5f. sample_across turns a WEIGHTED ensemble into plottable samples")
for label, ensemble in ((f"{len(tiny_ens)} scenarios", tiny_ens),
                        (f"{len(full_ens)} scenarios", full_ens)):
    drawn = sample_across(ensemble, [xc], total=2000,
                          rng=np.random.default_rng(0))[xc]
    print(f"      {label:16s} -> {drawn.shape[0]:5d} samples, "
          f"fraction from the cat1=b branch = {float((drawn > 500).mean()):.4f}")
print("      (true cat1=b weight is 0.01)")
