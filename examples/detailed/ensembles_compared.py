"""Choosing an ensemble: ONE model, four questions, four right answers.

An ensemble turns a model's uncertainty into concrete numbers to compute over.
There are three built-in strategies plus an open protocol, and they are NOT
interchangeable routes to the same number -- each makes a different KIND of
question answerable.

Everything here uses deliberately abstract names, so that nothing but the
ensemble mechanics needs to be held in mind:

    x1, x2    continuous uncertainties (DistributionIndex)
    cat1      a categorical uncertainty, outcomes "a" / "b", SKEWED 0.99/0.01
    cat2      a categorical uncertainty, outcomes "p" / "q", 0.80/0.20
    xc        a distribution whose PARAMETERS depend on cat1 and cat2
    y         the model output

The same model is grown and shrunk to suit each question:

  PART 1  y = x1 * x2, continuous only.        -> DistributionEnsemble
          We want the whole distribution of y: mean, quantiles, tail risk.

  PART 2  add cat1 and cat2, and xc.           -> CrossProductEnsemble
          Now y must be readable PER BRANCH, and one branch is rare.
          The longest part: how to read a branch out of a result, how the
          weights work, and joint vs marginal views.

  PART 3  back to y = x1 * x2.                 -> PartitionedEnsemble
          How much does EACH of x1 and x2 drive y? That needs the two kept
          on separate axes, not merged into one.

  PART 4  y = f(x2), smooth and expensive.     -> your own AxisEnsemble
          Gaussian quadrature is exact with 5 nodes; we write the ensemble.

PART 5 collects the practical traps that apply whichever ensemble you pick.

The through-line: pick by what you need to READ OFF the result, not by what is
fastest. Cost differences are usually small; the difference in what you can ask
afterwards is not.
"""

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    define, inputs, outputs, Index, Model, CategoricalIndex, DistributionIndex,
    ConditionalDistributionIndex, Scenario,
    DistributionEnsemble, CrossProductEnsemble, Evaluation,
    PartitionedEnsemble, EnsembleAxisSpec, sample_across, AxisEnsemble,
)
from civic_digital_twins.dt_model.axes import Axis, ENSEMBLE

# The two continuous uncertainties used throughout the file.
x1 = DistributionIndex("x1", stats.norm, {"loc": 100.0, "scale": 25.0})
x2 = DistributionIndex("x2", stats.norm, {"loc": 1.0, "scale": 0.2})


# =============================================================================
# PART 1 -- DistributionEnsemble: continuous noise only
# =============================================================================
# The simplest version of the model: y = x1 * x2. No categorical structure,
# nothing to enumerate, no conditional dependence. We want the shape of y's
# distribution, so sampling is not a compromise here -- it is the right tool.
print("=" * 74)
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


basic_model = BasicModel(inputs=BasicModel.Inputs(x1=x1, x2=x2))
basic_scenario = Scenario(basic_model)

print("\n  convergence of the mean (true value 100 x 1.0 = 100):")
for size in (100, 1_000, 10_000, 100_000):
    basic_res = Evaluation(basic_scenario).evaluate(
        ensemble=DistributionEnsemble(basic_scenario, size=size,
                                      rng=np.random.default_rng(0)),
    )
    print(f"    size={size:7d} -> E[y] = "
          f"{float(basic_res.expected_value(basic_model.outputs.y)):8.3f}")

# Every draw carries the SAME weight (1/size), which is what makes plain numpy
# percentiles correct here -- no weighted-quantile machinery needed.
basic_res = Evaluation(basic_scenario).evaluate(
    ensemble=DistributionEnsemble(basic_scenario, size=50_000,
                                  rng=np.random.default_rng(0)),
)
basic_draws = np.asarray(basic_res[basic_model.outputs.y]).ravel()
print(f"\n  the raw result is {basic_draws.shape[0]} equally-weighted draws, so")
print("  the whole distribution is available, not just the mean:")
for q in (5, 25, 50, 75, 95):
    print(f"    p{q:<2d} = {np.percentile(basic_draws, q):8.2f}")
print(f"    P(y > 150) = {float((basic_draws > 150).mean()):.4f}")

print("""
  Why this ensemble: all the uncertainty is continuous, so there is no
  structure to exploit. Use DistributionEnsemble when the question is "what
  does the distribution of the output look like" and nothing categorical
  is in play.""")


# =============================================================================
# PART 2 -- CrossProductEnsemble: the same model, plus categorical structure
# =============================================================================
# Extend the SAME model with two categorical uncertainties:
#   cat1   outcomes "a" / "b", probabilities 0.99 / 0.01   <- deliberately rare
#   cat2   outcomes "p" / "q", probabilities 0.80 / 0.20
# They are independent, so the cross product pairs every outcome of one with
# every outcome of the other: 2 x 2 = 4 branches.
#
# The output must now be readable PER BRANCH, and the branch that dominates y
# is the rarest one. That is what enumeration is for.
print("\n" + "=" * 74)
print("PART 2 -- CrossProductEnsemble: per-branch answers, and a rare branch")
print("=" * 74)

cat1 = CategoricalIndex("cat1", {"a": 0.99, "b": 0.01})
cat2 = CategoricalIndex("cat2", {"p": 0.80, "q": 0.20})


# xc depends on BOTH categoricals, so it is a CONDITIONAL distribution: the
# factory returns a different distribution per branch. (A conditional index is
# also something DistributionEnsemble cannot represent at all -- it raises.)
# The factory receives parents as keyword arguments named after them.
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
        # x2 is the same PLAIN index from Part 1: it still just multiplies,
        # and it knows nothing about the branches.
        return FullModel.Outputs(y=Index("y", inputs.xc * inputs.x2))


full_model = FullModel(inputs=FullModel.Inputs(
    cat1=cat1, cat2=cat2, xc=xc, x2=x2,
))
full_scenario = Scenario(full_model)

# -----------------------------------------------------------------------------
# A DELIBERATELY TINY ENSEMBLE -- 4 branches x 2 samples = 8 scenarios.
# Small enough to print in full, which is the only way the weights and the
# alignment between arrays become concrete. Everything in this part is shown
# on these 8 rows; a larger run appears once at the end for real numbers.
# -----------------------------------------------------------------------------
tiny_ens = CrossProductEnsemble(full_scenario, n_samples_per_combo=2,
                                rng=np.random.default_rng(0))
tiny_res = Evaluation(full_scenario).evaluate(ensemble=tiny_ens)

# There is no built-in .by_branch() helper. Instead FOUR arrays line up
# one-to-one, ONE ENTRY PER SCENARIO, and you select with a boolean mask:
(tiny_w,) = tiny_ens.ensemble_weights                            # weight
tiny_y = np.asarray(tiny_res[full_model.outputs.y]).ravel()      # output
tiny_c1 = np.asarray(tiny_res[cat1]).ravel()                     # label 1
tiny_c2 = np.asarray(tiny_res[cat2]).ravel()                     # label 2

print(f"\n  the whole ensemble, all {len(tiny_ens)} scenarios "
      f"(4 branches x 2 samples):")
print(f"    {'i':>2s}  {'cat1':5s} {'cat2':5s} {'weight':>8s} {'y':>10s}")
for i in range(len(tiny_w)):
    print(f"    {i:2d}  {tiny_c1[i]:5s} {tiny_c2[i]:5s} "
          f"{tiny_w[i]:8.5f} {tiny_y[i]:10.3f}")
print(f"    weights sum to {tiny_w.sum():.4f}  <- over the WHOLE ensemble")

print("""
    Where each weight comes from: a branch's weight is the PRODUCT of the
    two categorical probabilities, then split across its replicates.
      cat1=a + cat2=p -> 0.99 * 0.80 = 0.7920, / 2 = 0.39600 each
      cat1=a + cat2=q -> 0.99 * 0.20 = 0.1980, / 2 = 0.09900 each
      cat1=b + cat2=p -> 0.01 * 0.80 = 0.0080, / 2 = 0.00400 each
      cat1=b + cat2=q -> 0.01 * 0.20 = 0.0020, / 2 = 0.00100 each
    That product is what "cross product" means, and it is why adding a
    categorical MULTIPLIES the branch count rather than adding to it.""")

# -----------------------------------------------------------------------------
# WEIGHTS AND RENORMALISATION -- the part that is easy to get wrong
# -----------------------------------------------------------------------------
# The weights sum to 1.0 across the WHOLE ensemble. So selecting one branch
# gives weights that sum to that branch's PROBABILITY, not to 1. A conditional
# mean must divide by that sum:
#
#     E[y | branch] = sum(w_i * y_i) / sum(w_i)        over that branch only
#
# That division IS the renormalisation, and np.average(values, weights=w) does
# it internally. Spelled out on the cat1="b" rows of the table above:
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
      f"{np.sum(w_sub * y_sub):12.4f}  <- WRONG")
print(f"      scaled down by sum(w) = {np.sum(w_sub):.4f}; that is the branch's")
print("      CONTRIBUTION to the overall mean, not the branch's mean.")
print(f"    y_sub.mean()  [no weights]       = {y_sub.mean():12.4f}  <- WRONG")
print("      treats cat2=q as equally likely as cat2=p, because the mask holds")
print("      the same NUMBER of rows from each -- though p is four times more")
print("      probable. Sample counts are not probabilities.")

# -----------------------------------------------------------------------------
# JOINT AND MARGINAL VIEWS -- still on the tiny ensemble, so you can check
# every number against the 8 printed rows by hand.
# -----------------------------------------------------------------------------
print("\n  JOINT branches -- one mask per categorical, combined with &:")
print(f"    {'cat1':5s} {'cat2':5s} {'n':>3s} {'P (exact)':>10s} "
      f"{'E[y|branch]':>13s}")
for c1_outcome in ("a", "b"):
    for c2_outcome in ("p", "q"):
        # one mask per categorical, ANDed together
        mask = (tiny_c1 == c1_outcome) & (tiny_c2 == c2_outcome)
        branch_weights = tiny_w[mask]
        branch_values = tiny_y[mask]
        print(f"    {c1_outcome:5s} {c2_outcome:5s} {mask.sum():3d} "
              f"{branch_weights.sum():10.4f} "
              f"{np.average(branch_values, weights=branch_weights):13.3f}")


def marginal_report(label_array, outcomes, title, heading, weights, values):
    """Print E[y | one categorical], averaging over all the others."""
    print(f"\n  {title}")
    print(f"    {heading:5s} {'n':>3s} {'P (exact)':>10s} {'E[y]':>12s}")
    for outcome in outcomes:
        # ONLY this categorical is constrained; the other varies freely inside
        # the mask, mixed according to its own weights. Those weights are
        # UNEVEN, which is where np.average's renormalisation does real work.
        mask = label_array == outcome
        subset_weights = weights[mask]
        subset_values = values[mask]
        print(f"    {outcome:5s} {mask.sum():3d} {subset_weights.sum():10.4f} "
              f"{np.average(subset_values, weights=subset_weights):12.3f}")


marginal_report(tiny_c1, ("a", "b"),
                "MARGINAL over cat2 -- mask on cat1 ONLY:",
                "cat1", tiny_w, tiny_y)
marginal_report(tiny_c2, ("p", "q"),
                "MARGINAL over cat1 -- mask on cat2 ONLY:",
                "cat2", tiny_w, tiny_y)

print("""
    Both marginals partition the whole ensemble, so each recomposes to the
    same overall mean. Which one you report is a modelling decision:
    marginalising over the RARE variable barely moves the rows, while
    marginalising over the COMMON one changes them a lot.""")

# -----------------------------------------------------------------------------
# CONDITIONAL vs PLAIN distribution index, across the branches.
# -----------------------------------------------------------------------------
tiny_xc = np.asarray(tiny_res[xc]).ravel()
tiny_x2 = np.asarray(tiny_res[x2]).ravel()
print("\n  the two kinds of distribution index behave differently by branch:")
print(f"    {'cat1':5s} {'cat2':5s} {'xc (conditional)':>20s} "
      f"{'x2 (plain)':>18s}")
for c1_outcome in ("a", "b"):
    for c2_outcome in ("p", "q"):
        mask = (tiny_c1 == c1_outcome) & (tiny_c2 == c2_outcome)
        print(f"    {c1_outcome:5s} {c2_outcome:5s} "
              f"{str(np.round(tiny_xc[mask], 1)):>20s} "
              f"{str(np.round(tiny_x2[mask], 3)):>18s}")
print("""    xc is CONDITIONAL -- the factory hands each branch its own
    distribution, so the magnitudes jump by orders of magnitude across rows.
    x2 is PLAIN -- one N(1.0, 0.2), redrawn independently in every branch;
    with only 2 draws per branch the sample scatter is visible, but no row is
    systematically higher than another.

    Both interact with the categoricals in compute() -- y is their product.
    The difference is WHERE the branch dependence lives: in the distribution
    itself (conditional) or only in the arithmetic (plain).""")

# -----------------------------------------------------------------------------
# The branches decompose the overall mean EXACTLY (law of total expectation):
#     E[y] = sum over branches of  P(branch) * E[y | branch]
# Still on the tiny ensemble, so you can verify it against the 8 rows above.
# This is the check to run whenever you slice a result by branch: if it fails,
# your masks do not partition the ensemble or your weights are mishandled.
# -----------------------------------------------------------------------------
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

# A larger run of the SAME model, used only where sample size actually matters:
# the reportable table just below, and a couple of the traps in PART 5.
# Everything pedagogical stays on the 8-row ensemble above.
full_ens = CrossProductEnsemble(full_scenario, n_samples_per_combo=5000,
                                rng=np.random.default_rng(0))
full_res = Evaluation(full_scenario).evaluate(ensemble=full_ens)
(full_w,) = full_ens.ensemble_weights
full_y = np.asarray(full_res[full_model.outputs.y]).ravel()
full_c1 = np.asarray(full_res[cat1]).ravel()
full_c2 = np.asarray(full_res[cat2]).ravel()

print(f"\n  the reportable version: {len(full_ens)} scenarios "
      f"(4 branches x 5000). Same code, more samples:")
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
print("    Note P(exact) is IDENTICAL to the tiny run -- probabilities come")
print("    from enumeration, not from sampling. Only the conditional means")
print("    and quantiles needed the extra draws.")

print("""
  Why this ensemble: the cat1=b, cat2=q branch holds 0.2% of the probability
  and dominates y. Enumeration gives it a GUARANTEED 5000 samples and an
  EXACT weight, so its mean and p99 are solid. A sampler spending the same
  20000 evaluations would land roughly 40 there -- enough for a rough mean,
  far too few for a 99th percentile -- and the branch probabilities would
  themselves be estimates. Shrink the budget and that corner vanishes, at
  which point the conditional answer does not exist at all.

  Sampling is also simply unavailable here: xc is conditional, and
  DistributionEnsemble rejects conditional indexes outright.""")


# =============================================================================
# PART 3 -- PartitionedEnsemble: which uncertainty drives the output?
# =============================================================================
# Back to the Part 1 model -- y = x1 * x2, no categoricals -- but a different
# question. Not "what is y's distribution" but "which of x1 and x2 drives it
# more", because you might pay to measure one of them better.
#
# A flat sample cannot answer that: every draw welds one x1 value to one x2
# value, and you can never un-mix them. Separate axes keep them distinguishable.
print("\n" + "=" * 74)
print("PART 3 -- PartitionedEnsemble: attributing the spread to each source")
print("=" * 74)

part_ens = PartitionedEnsemble(
    basic_scenario,
    axes=[
        EnsembleAxisSpec("x1_axis", indexes=[x1], size=60),
        EnsembleAxisSpec("x2_axis", indexes=[x2], size=40),
    ],
)
part_res = Evaluation(basic_scenario).evaluate(ensemble=part_ens)
grid = np.asarray(part_res[basic_model.outputs.y])

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
print("\n  the two are close here, so neither dominates -- measuring either")
print("  one better would help about equally. That is an actionable answer a")
print("  single flat axis simply cannot produce.")

print("""
  Why this ensemble: the grid shape IS the answer, and it is cheap -- 100
  draws cover 2400 combinations because the axes broadcast. The assumption it
  encodes is INDEPENDENCE; if the two were correlated this factorial grid
  would invent combinations that never occur, and a conditional index inside
  a CrossProductEnsemble would be the honest choice instead.""")


# =============================================================================
# PART 4 -- your own AxisEnsemble: exact integration of a smooth function
# =============================================================================
# One more question about the same ingredients: y is now a smooth function of
# x2 alone, and each evaluation is expensive. We want the expectation in as few
# evaluations as possible.
#
# Monte Carlo converges at 1/sqrt(n). For a smooth function of a normal,
# Gaussian quadrature is exact with a handful of nodes. The library ships no
# quadrature ensemble -- so we write one. AxisEnsemble is a structural
# Protocol: three members, no inheritance, nothing to register.
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
        # y scales with the SQUARE of x2 -- smooth, so quadrature integrates
        # it exactly.  E[y] = 1000 * E[x2^2] = 1000 * (1.0^2 + 0.2^2) = 1040
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

print("""
  Why write your own: 5 model evaluations against 50000, and the 5 are exact.
  When evaluations are expensive and the response is smooth, that is decisive
  -- and unreachable with the built-in ensembles.

  The contract, if you write one:
    * weights must sum to 1.0 -- expected_value() is a weighted average and
      silently mis-scales otherwise.
    * every array from assignments() must carry ALL ensemble dims in order,
      size 1 where the index does not vary along an axis. A wrong length
      surfaces as a raw numpy broadcast error at evaluate() time, not as a
      helpful message about the protocol.
    * axes must have role ENSEMBLE -- that is what marks them for integration.
  Latin hypercube, historical records replayed as scenarios, or a fixed list
  of hand-picked cases all drop in the same way.""")


# =============================================================================
# PART 5 -- traps that apply whichever ensemble you chose
# =============================================================================
print("\n" + "=" * 74)
print("PART 5 -- practical traps, independent of which ensemble you picked")
print("=" * 74)

# --- 5a. max_categorical_size silently abandons exactness above 20 ----------
# A categorical with 25 outcomes instead of 2.
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
print("      The default of 20 silently samples a larger support. Raise it")
print("      above your largest support when you need exactness.")

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
print("      Give weights at construction, or via a Scenario dict override.")

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
print("      So a shared DistributionEnsemble does NOT pin two scenarios to")
print("      the same noise -- the draws move underneath you.")

# --- 5e. sampling budget and cost ------------------------------------------
print("\n  5e. n_samples_per_combo is PER BRANCH, and the weight is split")
for n in (1, 3):
    be = CrossProductEnsemble(full_scenario, n_samples_per_combo=n,
                              rng=np.random.default_rng(0))
    (bwt,) = be.ensemble_weights
    print(f"      n={n}: size={len(be):2d}  weights={np.round(bwt, 4)}")
print("      Each branch gets n samples regardless of probability -- which is")
print("      what protects the rare cat1=b branch in PART 2. The branch weight")
print("      is split across its replicates. Cost = branches x n x shape.")

# --- 5f. sample_across, for plotting a weighted ensemble --------------------
print("\n  5f. sample_across turns a WEIGHTED ensemble into plottable samples")
# Each scenario contributes max(1, round(w_i x total)) samples. That max(1, ...)
# matters: once the ensemble has more scenarios than `total`, every scenario is
# forced to contribute one and the weights stop being respected.
for label, ensemble in ((f"{len(tiny_ens)} scenarios", tiny_ens),
                        (f"{len(full_ens)} scenarios", full_ens)):
    drawn = sample_across(ensemble, [xc], total=2000,
                          rng=np.random.default_rng(0))[xc]
    print(f"      {label:16s} -> {drawn.shape[0]:5d} samples, "
          f"fraction from the cat1=b branch = {float((drawn > 500).mean()):.4f}")
print("      True cat1=b weight is 0.01. The small ensemble reproduces it; the")
print("      large one does NOT -- with more scenarios than total=2000, the")
print("      max(1, ...) floor gives every scenario a sample regardless of")
print("      weight. Use a COARSE ensemble, or raise total well above the")
print("      scenario count.")

print("""
======================================================================
  Choosing, in one line each:
    DistributionEnsemble  continuous noise, one marginal answer, and you
                          want equal-weight draws for quantiles and tails.
    CrossProductEnsemble  per-branch answers, rare branches, exact
                          probabilities, or ANY conditional index.
    PartitionedEnsemble   independent sources you need to vary one at a
                          time -- the result keeps a grid you can reduce
                          along either axis.
    your own              a better integration rule for your structure;
                          three members, no inheritance.
  Decide by what you must READ OFF the result. That, not speed, is what
  actually separates them.
======================================================================""")
