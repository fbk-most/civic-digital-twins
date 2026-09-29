"""Parameter axes and Scenarios: the two things you control from OUTSIDE.

Uncertainty inside a model is handled by indexes and integrated away by an
ensemble -- that is ensembles_compared.py's subject. This file is about the
other half: the things you, the analyst, set deliberately and compare.

There are exactly two such mechanisms, and they are not interchangeable:

  PARAMETER AXIS   a value you SWEEP. Declared on the Scenario via
                   parameter_axes=[...], supplied at evaluate() time as an
                   array. The result gains a real dimension per parameter, so
                   one run answers every combination at once.

  SCENARIO         an assumption you OVERRIDE. It shadows an index the model
                   already declared, changing what the ensemble itself is.
                   Each scenario is a separate run, compared side by side.

The dividing line is mechanical, not stylistic:

    a parameter axis changes the VALUES fed into one fixed ensemble;
    a scenario changes the ENSEMBLE, so the weights themselves move.

That is why a sweep is one evaluation and a scenario comparison is several --
and why you cannot express one with the other.

  PART 1  A parameter sweep: one run, one dimension added per parameter.
  PART 2  Two parameters at once -- a genuine grid, not nested loops.
  PART 3  Scenarios: the four override forms, and what each does to the
          ensemble (reweight, pin, restrict, replace).
  PART 4  Why they compose: sweep INSIDE each scenario, and read the table
          two ways.
  PART 5  The boundary -- what a Scenario can and cannot do, and what to
          reach for instead when it cannot.

Abstract naming throughout: p1/p2 are parameters, cat is a categorical, x is
continuous noise, y is the output.
"""

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    define, inputs, outputs, Index, Model, CategoricalIndex, DistributionIndex,
    Scenario, CrossProductEnsemble, DistributionEnsemble, Evaluation,
)

# -----------------------------------------------------------------------------
# One model for the whole file.
#   p1, p2  PARAMETERS -- abstract Index with no value; supplied at evaluate()
#   cat     a categorical uncertainty the ensemble will integrate away
#   x       continuous noise, likewise integrated away
# -----------------------------------------------------------------------------
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


# =============================================================================
# PART 1 -- a PARAMETER sweep: one run, one new dimension
# =============================================================================
# Declaring an index in parameter_axes= tells the Scenario "do not sample this,
# I will supply it". The ensemble then SKIPS it, and evaluate(parameters={...})
# provides an array of values. The result gains a PARAMETER axis of that length.
print("=" * 74)
print("PART 1 -- parameter axes: values you sweep, in a single run")
print("=" * 74)

base_scenario = Scenario(model, parameter_axes=[p1, p2])
base_ens = CrossProductEnsemble(base_scenario, n_samples_per_combo=400,
                                rng=np.random.default_rng(0))

one_d = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: np.array([1.0, 2.0, 3.0]), p2: np.array([1.0])},
)

print(f"\n  ensemble: {len(base_ens)} scenarios "
      f"(2 cat branches x 400 samples) -- built ONCE")
print("  the result layout now carries the parameters as real dimensions:")
for ax, size in one_d.layout.entries:
    print(f"    {ax.name:10s} {ax.role:10s} size={size}")

print("\n  sweeping p1 over [1, 2, 3] with p2 = 1:")
for value, y in zip(one_d.parameter_values_for(p1),
                    np.ravel(one_d.expected_value(model.outputs.y))):
    print(f"    p1={value:4.1f} -> E[y] = {y:7.3f}")

print("""
  Note what did NOT happen: we did not re-run the model three times, and the
  ensemble was not rebuilt. One ensemble, one evaluate(), three answers. The
  categorical was integrated away identically in all of them, so the three
  numbers are directly comparable -- they differ ONLY by p1.""")


# =============================================================================
# PART 2 -- two parameters: a grid, not nested loops
# =============================================================================
# Every parameter axis is independent, so supplying two arrays gives their full
# cross product in one shot. The result is genuinely 2-D and labelled by name.
print("\n" + "=" * 74)
print("PART 2 -- several parameters sweep together into one grid")
print("=" * 74)

grid_res = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: np.array([1.0, 2.0, 3.0]), p2: np.array([10.0, 20.0])},
)

labelled = grid_res.labeled(model.outputs.y)
print(f"\n  expected_value shape : {grid_res.expected_value(model.outputs.y).shape}"
      f"   dims {labelled.dims}")
print(f"  p1 values : {grid_res.parameter_values_for(p1)}")
print(f"  p2 values : {grid_res.parameter_values_for(p2)}")

print(f"\n    {'':>6s}" + "".join(f"{f'p2={v:g}':>12s}"
                                  for v in grid_res.parameter_values_for(p2)))
for i, v1 in enumerate(grid_res.parameter_values_for(p1)):
    row = "".join(f"{labelled.values[i, j]:12.3f}"
                  for j in range(labelled.values.shape[1]))
    print(f"    p1={v1:<3g}{row}")

print("""
  Six answers, still ONE evaluation and ONE ensemble. Because the axes are
  named, you select by meaning rather than position:
    labelled.sel(p1=0, p2=1)   -- indices into each parameter axis
  which is what makes a sweep safe to extend: adding p3 does not renumber
  anything you already wrote.""")
print(f"  labelled.sel(p1=0, p2=1) = {labelled.sel(p1=0, p2=1).values:.3f}"
      f"   (p1={grid_res.parameter_values_for(p1)[0]:g}, "
      f"p2={grid_res.parameter_values_for(p2)[1]:g})")


# =============================================================================
# PART 3 -- SCENARIOS: overrides that change the ensemble itself
# =============================================================================
# A Scenario override shadows an index the MODEL declared. Unlike a parameter,
# it is not a value fed in at evaluate() time -- it changes what the ensemble
# is, which is why each scenario needs its own ensemble and its own run.
#
# Four override forms, and what each does to a CategoricalIndex:
#   dict[str, float]  REWEIGHT  -- same outcomes, new probabilities
#   str               PIN       -- collapse to one outcome (branches shrink)
#   list[str]         RESTRICT  -- keep a subset, renormalise its weights
#   Distribution      REPLACE   -- for a DistributionIndex, swap the law
print("\n" + "=" * 74)
print("PART 3 -- scenarios: four override forms, four different ensembles")
print("=" * 74)

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

print(f"\n  {'scenario':22s} {'branches':>9s} {'E[y] at p1=p2=1':>18s}")
for label, sc in scenarios.items():
    ens = CrossProductEnsemble(sc, n_samples_per_combo=400,
                               rng=np.random.default_rng(0))
    res = Evaluation(sc).evaluate(
        ensemble=ens,
        parameters={p1: np.array([1.0]), p2: np.array([1.0])},
    )
    value = float(np.ravel(res.expected_value(model.outputs.y))[0])
    print(f"  {label:22s} {len(ens):9d} {value:18.4f}")

print("""
  Read the branches column: pinning and restricting HALVE the ensemble,
  because an outcome was removed from the cross product entirely. Reweighting
  keeps all branches but changes their weights. Replacing a distribution
  leaves the branch structure alone and changes what is drawn inside it.

  That column is the tell that a scenario is not a parameter: a parameter can
  never change the number of branches, because it is not part of the ensemble
  at all.""")


# =============================================================================
# PART 4 -- they COMPOSE: sweep inside each scenario
# =============================================================================
# This is the normal shape of a real study. Each scenario is its own run; the
# parameter sweep happens within it. The result is a table you read two ways.
print("\n" + "=" * 74)
print("PART 4 -- sweeping INSIDE each scenario: the two-way table")
print("=" * 74)

sweep_values = np.array([1.0, 2.0, 3.0])
study = {
    "baseline":       Scenario(model, parameter_axes=[p1, p2]),
    "pessimistic":    Scenario(model, overrides={cat: {"a": 0.2, "b": 0.8}},
                               parameter_axes=[p1, p2]),
    "cat pinned 'a'": Scenario(model, overrides={cat: "a"},
                               parameter_axes=[p1, p2]),
}

print(f"\n  {'scenario':16s}" + "".join(f"{f'p1={v:g}':>10s}" for v in sweep_values))
for label, sc in study.items():
    ens = CrossProductEnsemble(sc, n_samples_per_combo=400,
                               rng=np.random.default_rng(0))
    res = Evaluation(sc).evaluate(
        ensemble=ens,
        parameters={p1: sweep_values, p2: np.array([1.0])},
    )
    cells = "".join(f"{v:10.3f}"
                    for v in np.ravel(res.expected_value(model.outputs.y)))
    print(f"  {label:16s}{cells}")

print("""
  ACROSS a row  -- the parameter sweep: what changes as we turn the dial.
                   ONE run produced the whole row.
  DOWN a column -- the scenario comparison: what changes if our ASSUMPTIONS
                   were different. Each row needed its own run and its own
                   ensemble.

  The uncertainty never appears in the table: cat and x were integrated away
  into every single cell. Three different things, three different treatments
  -- and only the first two are under your control.""")


# =============================================================================
# PART 5 -- the boundary: what a Scenario cannot do
# =============================================================================
print("\n" + "=" * 74)
print("PART 5 -- where the mechanism stops, and what to use instead")
print("=" * 74)

# 5a. A Scenario can only act on an index the MODEL declared. It cannot invent
#     an uncertainty that is not already in the model.
print("\n  5a. a Scenario acts ON a declared index -- it cannot invent one.")
print("      There is no override that ADDS a new categorical: the model author")
print("      decides WHAT is uncertain, the analyst decides WHICH ASSUMPTIONS")
print("      to test about it. Adding a new uncertainty means editing the model.")

# 5b. Parameters and overrides must not collide on the same index.
print("\n  5b. an index cannot be both a parameter and an override")
try:
    clash = Scenario(model, overrides={p1: 5.0}, parameter_axes=[p1, p2])
    Evaluation(clash).evaluate(
        ensemble=CrossProductEnsemble(clash, n_samples_per_combo=10),
        parameters={p1: np.array([1.0]), p2: np.array([1.0])},
    )
except ValueError as exc:
    print(f"      raises: {str(exc)[:96]}")
print("      Sensible: one says 'sweep this', the other says 'fix this'.")

# 5c. A parameter axis is NOT integrated away, so it carries no probability.
print("\n  5c. parameters carry no probability, and are never averaged over")
weighted = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: sweep_values, p2: np.array([1.0])},
)
print(f"      expected_value over a swept p1 -> shape "
      f"{weighted.expected_value(model.outputs.y).shape}, one value PER p1")
print("      The ENSEMBLE axis was integrated away; the PARAMETER axis was")
print("      kept. If you want a single number across p1 you must decide the")
print("      weights yourself -- the library will not guess them, because a")
print("      swept lever has no probability distribution.")

print("""
======================================================================
  In one line each:
    parameter axis   a value you sweep. One ensemble, one run, one new
                     result dimension per parameter. No probability.
    Scenario         an assumption you override. A different ensemble,
                     a separate run, compared side by side.
    uncertainty      neither: declared in the model, integrated away by
                     the ensemble (see ensembles_compared.py).
  Ask which of the three a quantity is, and the API follows.
======================================================================""")
