# SPDX-License-Identifier: Apache-2.0

"""Parameter axes and Scenarios: the two things you control from OUTSIDE.

Narrative and explanations: see Parameters_vs_Scenarios_Tutorial.md in this
directory. Each "PART N" banner below matches a section of the same name there.

  PART 1  A parameter sweep: one run, one dimension added per parameter.
  PART 2  Two parameters at once -- a genuine grid, not nested loops.
  PART 3  Scenarios: the four override forms (reweight, pin, restrict, replace).
  PART 4  Why they compose: sweep INSIDE each scenario.
  PART 5  The boundary -- what a Scenario can and cannot do.

Abstract naming throughout: p1/p2 are parameters, cat is a categorical, x is
continuous noise, y is the output.
"""

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    define, inputs, outputs, Index, Model, CategoricalIndex, DistributionIndex,
    Scenario, CrossProductEnsemble, Evaluation,
)

# -----------------------------------------------------------------------------
# One model for the whole file.
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


# =============================================================================
# PART 2 -- two parameters: a grid, not nested loops
# =============================================================================
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

print(f"\n  labelled.sel(p1=0, p2=1) = {labelled.sel(p1=0, p2=1).values:.3f}"
      f"   (p1={grid_res.parameter_values_for(p1)[0]:g}, "
      f"p2={grid_res.parameter_values_for(p2)[1]:g})")


# =============================================================================
# PART 3 -- SCENARIOS: overrides that change the ensemble itself
# =============================================================================
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

print(f"\n  {'scenario':22s} {'scenarios':>9s} {'E[y] at p1=p2=1':>18s}")
for label, sc in scenarios.items():
    ens = CrossProductEnsemble(sc, n_samples_per_combo=400,
                               rng=np.random.default_rng(0))
    res = Evaluation(sc).evaluate(
        ensemble=ens,
        parameters={p1: np.array([1.0]), p2: np.array([1.0])},
    )
    value = float(np.ravel(res.expected_value(model.outputs.y))[0])
    print(f"  {label:22s} {len(ens):9d} {value:18.4f}")


# =============================================================================
# PART 4 -- they COMPOSE: sweep inside each scenario
# =============================================================================
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


# =============================================================================
# PART 5 -- the boundary: what a Scenario cannot do
# =============================================================================
print("\n" + "=" * 74)
print("PART 5 -- where the mechanism stops, and what to use instead")
print("=" * 74)

# 5a. A Scenario can only act on an index the MODEL declared (no code needed:
#     there is simply no override that ADDS a new uncertainty).

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

# 5c. A parameter axis is NOT integrated away, so it carries no probability.
print("\n  5c. parameters carry no probability, and are never averaged over")
weighted = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: sweep_values, p2: np.array([1.0])},
)
print(f"      expected_value over a swept p1 -> shape "
      f"{weighted.expected_value(model.outputs.y).shape}, one value PER p1")
