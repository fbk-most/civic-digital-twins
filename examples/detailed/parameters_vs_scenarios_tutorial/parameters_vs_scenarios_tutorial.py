# SPDX-License-Identifier: Apache-2.0

"""Parameter axes and Scenarios: the two things you control from OUTSIDE.

Narrative and explanations: see Parameters_vs_Scenarios_Tutorial.md in this
directory. Each "PART N" banner below matches a section of the same name there.

  PART 1  A parameter sweep: what the result looks like, and how to read it.
  PART 2  Two parameters at once: a grid, read by name.
  PART 3  A correlated sweep: several parameters moving along ONE axis.
  PART 4  Scenarios: the four override forms and what each does to the result.
  PART 5  They compose: sweep inside each scenario, compare across scenarios.
  PART 6  Traps and boundaries.

Abstract naming throughout: p1/p2 are parameters, cat is a categorical, x is
continuous noise, y is the output.

Results are read BY NAME (LabeledArray.sel) wherever possible, so the code does
not depend on the order of the dimensions.
"""

import numpy as np
from scipy import stats

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    CrossProductEnsemble,
    DistributionIndex,
    Evaluation,
    EvaluationResult,
    GenericIndex,
    Index,
    LabeledArray,
    Model,
    Scenario,
    define,
    inputs,
    outputs,
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
    """y = p1 * p2 * x * factor(cat), where outcome "b" triples the output."""

    @inputs
    class Inputs:
        """Model inputs."""

        p1: Index
        p2: Index
        cat: CategoricalIndex
        x: DistributionIndex

    @outputs
    class Outputs:
        """Model outputs."""

        y: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute y = p1 * p2 * x * factor(cat)."""
        # outcome "b" triples the output, so the categorical's weighting is
        # visible in every number below.
        factor = 1.0 + 2.0 * (inputs.cat == "b")
        return SweepModel.Outputs(
            y=Index("y", inputs.p1 * inputs.p2 * inputs.x * factor)
        )


model = SweepModel(inputs=SweepModel.Inputs(p1=p1, p2=p2, cat=cat, x=x))
y_idx = model.outputs.y


def labeled_raw(res: EvaluationResult, idx: GenericIndex) -> LabeledArray:
    """Return the raw values of *idx* with every dimension named.

    ``res[idx]`` has size-1 dimensions wherever *idx* does not vary, so it is
    broadcast to the full result shape before the layout is attached.
    """
    return LabeledArray(np.broadcast_to(np.asarray(res[idx]), res.full_shape),
                        res.layout)


# =============================================================================
# PART 1 -- a PARAMETER sweep: what the result looks like, and how to read it
# =============================================================================
print("=" * 74)
print("PART 1 -- parameter axes: values you sweep, in a single run")
print("=" * 74)

base_scenario = Scenario(model, parameter_axes=[p1, p2])     # "I will supply p1, p2"
base_ens = CrossProductEnsemble(base_scenario, n_samples_per_combo=400,
                                rng=np.random.default_rng(0))

sweep_res = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: np.array([1.0, 2.0, 3.0]), p2: np.array([1.0])},
)

print(f"\n  abstract indexes left for the ensemble: "
      f"{[ix.name for ix in base_scenario.abstract_indexes()]}")
print(f"  ensemble: {len(base_ens)} rows (2 cat branches x 400 samples) -- built ONCE")

# 1a. The layout: which dimension is which.
print("\n  1a. the result layout (one entry per dimension, in order):")
for ax, size in sweep_res.layout.entries:
    print(f"    {ax.name:15s} {ax.role:10s} size={size}")

# 1b. Raw arrays: every index keeps the FULL rank, size-1 where it does not vary.
sweep_y = np.asarray(sweep_res[y_idx])          # (p1, p2, ensemble)
sweep_p1 = np.asarray(sweep_res[p1])            # (p1, 1, 1)
sweep_cat = np.asarray(sweep_res[cat])          # (1, 1, ensemble)
sweep_x = np.asarray(sweep_res[x])              # (1, 1, ensemble)
(sweep_w,) = base_ens.ensemble_weights      # (ensemble,) -- shared by every p1

print("\n  1b. raw shapes:")
for name, arr in (("y", sweep_y), ("p1", sweep_p1), ("cat", sweep_cat), ("x", sweep_x),
                  ("weights", sweep_w)):
    print(f"    {name:8s} {str(arr.shape):14s}")

# 1c. Reading by name: wrap the raw arrays, then .sel() by parameter name.
y_raw = labeled_raw(sweep_res, y_idx)         # dims ('p1', 'p2', '_cross_product')
cat_raw = labeled_raw(sweep_res, cat)         # same dims, labels broadcast
p1_values = sweep_res.parameter_values_for(p1)
i_p1 = list(p1_values).index(2.0)         # value -> position along p1

y_at_p1_2 = y_raw.sel(p1=i_p1, p2=0).values       # (ensemble,)
cat_labels = cat_raw.sel(p1=0, p2=0).values       # (ensemble,) -- any p1 will do
print(f"\n  1c. y_raw dims: {y_raw.dims}")
print(f"      p1 values: {p1_values}  -> p1=2.0 is position {i_p1}")
print(f"      y_raw.sel(p1={i_p1}, p2=0): shape {y_at_p1_2.shape}, "
      f"E = {np.average(y_at_p1_2, weights=sweep_w):.4f}")

# 1d. expected_value: the weighted mean over the ensemble, parameters kept.
ev = sweep_res.labeled(y_idx)                         # dims ('p1', 'p2')
manual = np.average(sweep_y, axis=-1, weights=sweep_w)    # ensemble is the last axis
print(f"\n  1d. sweep_res.labeled(y_idx) dims: {ev.dims}")
print("      sweeping p1 over [1, 2, 3] with p2 = 1:")
for i, value in enumerate(p1_values):
    print(f"    p1={value:4.1f} -> ev.sel(p1={i}, p2=0) = "
          f"{float(ev.sel(p1=i, p2=0).values):7.4f}   "
          f"np.average(axis=-1) = {manual[i, 0]:7.4f}")

# 1e. Branch masks act on the ensemble axis; wrap the reduced result again.
is_b = cat_labels == "b"
e_b = LabeledArray(np.average(y_raw.values[..., is_b], axis=-1,
                              weights=sweep_w[is_b]),
                   sweep_res.layout_of(y_idx))        # dims ('p1', 'p2') again
print(f"\n  1e. branch cat=b: {int(is_b.sum())} rows, P = {sweep_w[is_b].sum():.1f}")
print(f"      e_b dims {e_b.dims}; E[y | cat=b] per p1 = {e_b.sel(p2=0).values}")
print(f"      one cell, by name: "
      f"{np.average(y_raw.sel(p1=i_p1, p2=0).values[is_b], weights=sweep_w[is_b]):.4f}"
      f"  (p1=2)")

# 1f. The trap: .ravel() on a swept result glues the p1 values together.
print(f"\n  1f. sweep_y.ravel().shape = {sweep_y.ravel().shape}  "
      f"(= 3 p1 values x 800 rows, mixed)")
print(f"      plain mean of ALL of it = {sweep_y.mean():.4f}  <- not any E[y]")


# =============================================================================
# PART 2 -- two parameters: a grid, read by name
# =============================================================================
print("\n" + "=" * 74)
print("PART 2 -- several parameters sweep together into one grid")
print("=" * 74)

grid_res = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p1: np.array([1.0, 2.0, 3.0]), p2: np.array([10.0, 20.0])},
)

labelled = grid_res.labeled(y_idx)                 # dims ('p1', 'p2')
p1_vals = grid_res.parameter_values_for(p1)
p2_vals = grid_res.parameter_values_for(p2)

print(f"\n  raw dims             : {labeled_raw(grid_res, y_idx).dims}")
print(f"  labelled             : {labelled}")
print(f"  p1 values : {p1_vals}")
print(f"  p2 values : {p2_vals}")

# The table is built with .sel(), so it reads the same whatever the order.
print(f"\n    {'':>6s}" + "".join(f"{f'p2={v:g}':>12s}" for v in p2_vals))
for i, v1 in enumerate(p1_vals):
    row = "".join(f"{float(labelled.sel(p1=i, p2=j).values):12.3f}"
                  for j in range(len(p2_vals)))
    print(f"    p1={v1:<3g}{row}")

cell = labelled.sel(p1=0, p2=1)                    # int -> axis dropped
column = labelled.sel(p2=1)                        # every p1 at p2=20
print(f"\n  labelled.sel(p1=0, p2=1) = {float(cell.values):.3f}   "
      f"(p1={p1_vals[0]:g}, p2={p2_vals[1]:g})")
print(f"  labelled.sel(p2=1)       = {column}  values {column.values}")

# 2b. The ORDER of the parameter dimensions follows the parameters= dict.
swapped_res = Evaluation(base_scenario).evaluate(
    ensemble=base_ens,
    parameters={p2: np.array([10.0, 20.0]), p1: np.array([1.0, 2.0, 3.0])},
)
swapped_lab = swapped_res.labeled(y_idx)
print("\n  2b. same scenario, parameters= written as {p2: ..., p1: ...}:")
print(f"    dims                   : {swapped_lab.dims}   (was {labelled.dims})")
print(f"    .values[0]             : {swapped_lab.values[0]}   (p2=10, every p1)")
print(f"    .sel(p1=0)             : {swapped_lab.sel(p1=0).values}   "
      f"(p1=1, every p2 -- as before)")
print(f"    .sel(p1=0, p2=1)       : {float(swapped_lab.sel(p1=0, p2=1).values):.3f}"
      f"   (same cell as before)")

# 2c. Finding a cell by its value: argmax gives positions -> name them via dims.
flat = np.argmax(labelled.values)
best = {dim: int(pos)                              # int(): .sel needs Python ints
        for dim, pos in zip(labelled.dims,
                            np.unravel_index(flat, labelled.values.shape))}
print(f"\n  2c. argmax positions by name: {best}")
print(f"      largest E[y] at p1={p1_vals[best['p1']]:g}, p2={p2_vals[best['p2']]:g}: "
      f"{float(labelled.sel(**best).values):.3f}")

# 2d. Per-branch over the whole grid at once, read back by name.
grid_raw = labeled_raw(grid_res, y_idx)
grid_cat = labeled_raw(grid_res, cat).sel(p1=0, p2=0).values
(grid_w,) = base_ens.ensemble_weights
grid_b = grid_cat == "b"
grid_e_b = LabeledArray(np.average(grid_raw.values[..., grid_b], axis=-1,
                                   weights=grid_w[grid_b]),
                        grid_res.layout_of(y_idx))
print(f"\n  2d. E[y | cat=b] over the grid: {grid_e_b}")
print(grid_e_b.values)
print(f"      grid_e_b.sel(p1=2, p2=1) = {float(grid_e_b.sel(p1=2, p2=1).values):.3f}"
      f"   (p1=3, p2=20)")


# =============================================================================
# PART 3 -- a CORRELATED sweep: several parameters on ONE axis
# =============================================================================
print("\n" + "=" * 74)
print("PART 3 -- a correlated sweep: p1 and p2 move together along one axis")
print("=" * 74)

free_scenario = Scenario(model)                    # p1, p2 stay abstract
free_ens = CrossProductEnsemble(free_scenario, n_samples_per_combo=400,
                                rng=np.random.default_rng(0))

diag_res = Evaluation(free_scenario).evaluate(
    ensemble=free_ens,
    parameter_axes={"level": np.array([1.0, 2.0, 3.0])},   # ONE named axis
    parameters={p1: lambda level: level,                    # p1 = level
                p2: lambda level: 10.0 * level},            # p2 = 10 * level
)
diag_lab = diag_res.labeled(y_idx)

print("\n  layout:")
for ax, size in diag_res.layout.entries:
    print(f"    {ax.name:15s} {ax.role:10s} size={size}")
print(f"  named_axis_values       : {diag_res.named_axis_values}")
print(f"  p1 along the axis       : {labeled_raw(diag_res, p1).sel(_cross_product=0).values}")
print(f"  p2 along the axis       : {labeled_raw(diag_res, p2).sel(_cross_product=0).values}")
print(f"  diag_res.labeled(y_idx) : {diag_lab}")
print(f"  .values                 : {diag_lab.values}")
print(f"  .sel(level=1)           : {float(diag_lab.sel(level=1).values):.3f}"
      f"   (level=2: p1=2, p2=20)")


# =============================================================================
# PART 4 -- SCENARIOS: overrides that change the ensemble itself
# =============================================================================
print("\n" + "=" * 74)
print("PART 4 -- scenarios: four override forms, four different ensembles")
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

print(f"\n  {'scenario':20s} {'rows':>5s} {'cat shape':>12s} "
      f"{'weights (unique)':>26s} {'mean x':>7s} {'E[y]':>7s}")
for label, res in scenario_res.items():
    (w,) = scenario_ens[label].ensemble_weights
    cat_shape = str(np.asarray(res[cat]).shape)
    uniq = np.array2string(np.unique(w), precision=5)
    x_draws = labeled_raw(res, x).sel(p1=0, p2=0).values
    mean_x = float(np.average(x_draws, weights=w))
    e_y = float(res.labeled(y_idx).sel(p1=0, p2=0).values)
    print(f"  {label:20s} {len(scenario_ens[label]):5d} {cat_shape:>12s} "
          f"{uniq:>26s} {mean_x:7.3f} {e_y:7.4f}")

# Pinning removes cat from the ensemble: its raw array is a single constant.
pinned_res = scenario_res["pin cat='b'"]
pinned_abstract = [ix.name for ix in scenarios["pin cat='b'"].abstract_indexes()]
pinned_labels = labeled_raw(pinned_res, cat).sel(p1=0, p2=0).values
print(f"\n  pinned res[cat]  -> {np.asarray(pinned_res[cat])!r}")
print(f"  pinned abstract indexes: {pinned_abstract}")
print(f"  pinned labels via labeled_raw: shape {pinned_labels.shape}, "
      f"unique {np.unique(pinned_labels)}")

# Same seed, same branch structure -> the SAME draws of x (common random numbers).
same_x = np.array_equal(np.asarray(scenario_res["baseline"][x]),
                        np.asarray(scenario_res["reweight a=.2 b=.8"][x]))
print(f"  baseline vs reweight: identical x draws? {same_x}")


# =============================================================================
# PART 5 -- they COMPOSE: sweep inside each scenario, compare across scenarios
# =============================================================================
print("\n" + "=" * 74)
print("PART 5 -- sweeping INSIDE each scenario: the two-way table")
print("=" * 74)

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

# Raw arrays do NOT line up across scenarios: the ensemble sizes differ.
print("\n  raw y per scenario:")
for label, res in study_res.items():
    print(f"    {label:16s} {labeled_raw(res, y_idx)}")

# Expected values DO line up: select the same named cells in every scenario.
table = np.stack([res.labeled(y_idx).sel(p2=0).values      # (p1,) by name
                  for res in study_res.values()])          # (scenario, p1)
print(f"\n  stacked expected values: shape {table.shape}  (scenario, p1)")
print(f"\n  {'scenario':16s}" + "".join(f"{f'p1={v:g}':>10s}" for v in sweep_values))
for label, row in zip(study_res, table):
    print(f"  {label:16s}" + "".join(f"{v:10.3f}" for v in row))

print(f"\n  across a row (p1 sweep, baseline)  : {table[0]}")
print(f"  down a column (scenarios, p1=2)    : {table[:, 1]}")
print(f"  scenario effect vs baseline        : {table[1] - table[0]}")


# =============================================================================
# PART 6 -- traps and boundaries
# =============================================================================
print("\n" + "=" * 74)
print("PART 6 -- where the mechanism stops")
print("=" * 74)

# 6a. A Scenario can only act on an index the MODEL declared (no code needed:
#     there is simply no override that ADDS a new uncertainty).

# 6b. Parameters and overrides must not collide on the same index.
print("\n  6b. an index cannot be both a parameter and an override")
try:
    clash = Scenario(model, overrides={p1: 5.0}, parameter_axes=[p1, p2])
    Evaluation(clash).evaluate(
        ensemble=CrossProductEnsemble(clash, n_samples_per_combo=10),
        parameters={p1: np.array([1.0]), p2: np.array([1.0])},
    )
except ValueError as exc:
    print(f"      raises: {str(exc)[:96]}")

# 6c. A declared parameter axis MUST be supplied.
print("\n  6c. forgetting to supply a declared parameter")
try:
    Evaluation(base_scenario).evaluate(ensemble=base_ens,
                                       parameters={p1: np.array([1.0])})
except ValueError as exc:
    print(f"      raises: {str(exc)[:96]}")

# 6d. A parameter axis is NOT integrated away, so it carries no probability.
print("\n  6d. parameters carry no probability, and are never averaged over")
print(f"      sweep_res.labeled(y_idx) -> {sweep_res.labeled(y_idx)}, one value PER p1")
ev_p1 = sweep_res.labeled(y_idx).sel(p2=0).values       # (3,): one value per p1
my_weights = np.array([0.5, 0.3, 0.2])              # YOUR belief about p1
print(f"      your own average over p1 (weights {my_weights}) = "
      f"{float(np.average(ev_p1, weights=my_weights)):.4f}")
