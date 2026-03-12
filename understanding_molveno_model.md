# How the Molveno Overtourism Digital Twin Works

## What problem does this solve?

Molveno is a small town with a lake, and it gets lots of visitors. The fundamental question is:

> **Given some number of tourists and excursionists, can Molveno's resources handle them?**

"Resources" here means concrete things: parking spaces, beach area, hotel beds, and restaurant seats. The model checks whether each resource is overwhelmed or not, accounting for the fact that we don't know exact values for many things (how many parking spots are really usable, how many tourists come on a rainy Wednesday, etc.).

The script `overtourism_molveno.py` produces a plot that looks like a phase diagram: a 2D map with tourists on one axis, excursionists on the other, coloured by "sustainability" (red = fine, blue = overwhelmed).

---

## Project structure

The model lives in `civic-digital-twins/examples/overtourism_molveno/`:

| File | Role |
|------|------|
| `molveno_model.py` | All variable and constraint definitions; assembles `M_Base` |
| `molveno_presence_stats.py` | Statistical tables for tourist/excursionist counts by season, weather, weekday |
| `overtourism_metamodel.py` | Domain-specific classes: `OvertourismModel`, `PresenceVariable`, `OvertourismEnsemble`, `Constraint` |
| `overtourism_molveno.py` | Main execution script — evaluation, post-processing, plotting |

Core framework classes are in `civic-digital-twins/civic_digital_twins/dt_model/`:

| File | Key classes |
|------|-------------|
| `model/index.py` | `Index`, `UniformDistIndex`, `LognormDistIndex`, `TriangDistIndex`, `Distribution` |
| `model/model.py` | Base `Model` class |
| `simulation/evaluation.py` | `Evaluation`, `EvaluationResult` |
| `simulation/ensemble.py` | `WeightedScenario` type alias |

---

## Key concepts and terminology

Before diving into the variables, it helps to understand the building blocks the model is assembled from.

### Nodes

A **node** is any quantity in the model that can participate in a formula. Nodes form a **computation graph**: each node is either a leaf (a concrete value or distribution) or an internal node (a formula that combines other nodes). When the `Evaluation` engine runs, it resolves every node in the graph by substituting in the values for that particular scenario and grid point.

Examples of nodes:

- `PV_tourists` — a leaf node whose value is drawn from a truncated normal distribution
- `I_U_excursionists_parking` — a leaf node whose value is either 0.55 or 0.80 depending on `CV_weather`
- `I_U_parking` — an internal node (formula) computed from several other nodes

### Indexes (`Index` and subclasses)

An **Index** is a named quantity that represents a parameter of the system — a capacity, a usage fraction, a conversion factor, or a policy lever. Indexes are always scalars (after resolution). Their subclasses differ in *how* their value is resolved:

| Subclass | How the value is determined |
| -------- | --------------------------- |
| `Index` | Fixed scalar constant, or a formula node (arithmetic of other indexes) |
| `UniformDistIndex` | Uniform distribution `U(loc, loc+scale)` — resolved as a distribution for constraint evaluation |
| `LognormDistIndex` | Log-normal distribution — same |
| `TriangDistIndex` | Triangular distribution — same |

Distribution-backed indexes (capacity indexes, some rotation indexes) are **not sampled** during the grid evaluation. Instead, the evaluation computes `P(capacity > usage)` by evaluating the CDF of the distribution at the deterministic usage value.

### Context Variables (`CV_*`)

A **context variable** (CV) represents an external condition that the model does not control but that affects visitor behaviour. CVs are discrete random variables — each has a finite set of named values with associated probabilities (e.g., `CV_weather` takes `good/unsettled/bad` with probabilities 65%/20%/15%).

CVs are **not** optimised or swept in the grid evaluation; they are averaged over via the ensemble (see below).

### Presence Variables (`PV_*`)

A **presence variable** (PV) represents a visitor count. Given fixed CV values, a PV is a continuous distribution (truncated normal). In the grid evaluation, PVs become the **axes** of the grid — the model is evaluated for all combinations of `(tourists, excursionists)` values on the grid, rather than being sampled.

### Ensemble and scenarios

An **ensemble** is the collection of all meaningful CV combinations. For the base scenario, the full Cartesian product is 7 weekdays × 4 seasons × 3 weather types = 84 combinations. Each combination is a **scenario** (or `WeightedScenario`): a dict mapping each CV to a concrete value, plus a probability weight equal to the product of the individual CV probabilities.

When results are aggregated, quantities are averaged over scenarios using these weights (a weighted sum, not uniform averaging).

#### What a WeightedScenario object looks like

Each ensemble member is a `(probability, assignments)` tuple. In Python it looks like:

```python
(
    0.0191,                      # weight = (1/7) × 0.2065 × 0.65
    {
        CV_weekday: "monday",
        CV_season:  "very_high",
        CV_weather: "good",
    }
)
```

The `assignments` dict maps each CV index object to its concrete string value for this scenario. The weight is the joint probability under the independence assumption: each CV's probability is looked up and multiplied together.

#### How weights are computed

For the base scenario (`S_Base`), every CV combination is included. The weight of one scenario is:

```
w = P(weekday) × P(season) × P(weather)
  = (1/7)      × 0.2065    × 0.65      ≈ 0.0191   # monday, very_high, good
  = (1/7)      × 0.2065    × 0.20      ≈ 0.0059   # monday, very_high, unsettled
  = (1/7)      × 0.2065    × 0.15      ≈ 0.0044   # monday, very_high, bad
  = (1/7)      × 0.2609    × 0.65      ≈ 0.0242   # monday, high, good
  ...
  = (1/7)      × 0.2391    × 0.15      ≈ 0.0051   # sunday, low, bad
```

All 84 weights sum exactly to 1.0. This is verified in `doc_overtourism_getting_started.py` §5:

```python
weights = np.array([w for w, _ in scenarios])
assert abs(weights.sum() - 1.0) < 1e-10
```

#### What happens when a scenario filters CVs

`S_Good_Weather = {CV_weather: ["good", "unsettled"]}` drops bad-weather scenarios entirely: 7 × 4 × 2 = 56 members remain. Their raw weights no longer sum to 1.0 (they sum to 0.85, the total probability of non-bad weather). The evaluation code renormalises them internally before averaging.

`S_Bad_Weather = {CV_weather: ["bad"]}` gives 28 members. Each looks like:

```python
(
    0.0044,                      # (1/7) × 0.2065 × 0.15
    {CV_weekday: "monday", CV_season: "very_high", CV_weather: "bad"}
)
```

In every one of these 28 members, `I_U_excursionists_parking = 0.55` (rain value) and `I_U_tourists_beach = 0.25` — the `piecewise()` nodes resolve to their bad-weather branches for all scenarios in this ensemble.

#### What "averaging over the ensemble" means concretely

To get the expected parking demand at a single grid point, say (tourists=5000, excursionists=4000):

```
E[I_U_parking] = Σ_s  w_s × I_U_parking(5000, 4000, s)
               = 0.0191 × parking(monday, very_high, good)
               + 0.0059 × parking(monday, very_high, unsettled)
               + 0.0044 × parking(monday, very_high, bad)   # ← uses 0.55 for exc. parking
               + ...   (84 terms total)
```

This is the `tensordot` call in `plot_scenario()`:

```python
field_elem = np.tensordot(mask, result.weights, axes=([-1], [0]))
```

`mask` has shape `(101, 101, 84)` — one satisfaction probability per grid point per scenario. `result.weights` has shape `(84,)`. The contraction collapses the scenario axis, giving a `(101, 101)` weighted average.

#### A minimal hand-calculable example (2 CVs, 4 members)

With only `CV_season = {high: 0.5, low: 0.5}` and `CV_weather = {good: 0.7, bad: 0.3}`:

| season | weather | weight |
| ------ | ------- | ------ |
| high | good | 0.5 × 0.7 = 0.35 |
| high | bad | 0.5 × 0.3 = 0.15 |
| low | good | 0.5 × 0.7 = 0.35 |
| low | bad | 0.5 × 0.3 = 0.15 |

This is exactly how the real ensemble works — 4 cells instead of 84, same arithmetic. This is the pattern used in `doc_overtourism_getting_started.py` §5, which uses 2 seasons × 3 weather types = 6 members.

### Constraints

A **constraint** pairs a **usage formula** (what demand is placed on a resource) with a **capacity** (how much of that resource exists). Evaluating a constraint at a grid point gives `P(capacity > usage)` — the probability that the resource is not overwhelmed. A value of 1.0 means certain sustainability; 0.0 means certain overload.

### Sustainability field

The **sustainability field** is a 2-D array over the (tourists, excursionists) grid. Each cell contains the probability that *all* constraints are satisfied simultaneously (approximated as the product of per-constraint probabilities, under an independence assumption). Values close to 1.0 are sustainable; values close to 0.0 are unsustainable.

---

## Complete variable reference

All variables below are defined in `molveno_model.py` unless otherwise noted. Python variable names are used throughout.

### 1. Context Variables — "the conditions of the day"

These are things we don't control but that affect how many people show up and what they do:

| Python name | Type | Values | Probability |
|-------------|------|--------|-------------|
| `CV_weekday` | `UniformCategoricalContextVariable` | monday, tuesday, wednesday, thursday, friday, saturday, sunday | Uniform (1/7 each) |
| `CV_season` | `CategoricalContextVariable` | very_high, high, mid, low | 20.65%, 26.09%, 29.35%, 23.91% |
| `CV_weather` | `CategoricalContextVariable` | good, unsettled, bad | 65%, 20%, 15% |

Each context variable is a discrete random variable. When you "sample" it, you draw one of its values with the corresponding probability.

### 2. Presence Variables — "how many people show up"

| Python name | Type | Depends on | Distribution |
|-------------|------|-----------|--------------|
| `PV_tourists` | `PresenceVariable` | CV_weekday, CV_season, CV_weather | Truncated Normal (≥ 0) |
| `PV_excursionists` | `PresenceVariable` | CV_weekday, CV_season, CV_weather | Truncated Normal (≥ 0) |

Given a specific (weekday, season, weather) combination, the number of visitors follows a truncated normal (lower bound at 0, upper bound at 10 standard deviations above the mean) whose parameters are computed by **adding contributions** from each context variable independently:

```
mean(weekday, season, weather) = mean_season + mean_weather + mean_weekday
var(weekday, season, weather)  = std_season² + std_weather² + std_weekday²
std(weekday, season, weather)  = sqrt(var)
```

Note: the tables below list `std` values (not variances); the variance addition step requires squaring them before summing.

This is defined in `molveno_presence_stats.py`. The full tables:

**Season contributions** (from `season_stats`):

| Season | mean_tourists | std_tourists | mean_excursionists | std_excursionists | freq_rel |
|--------|--------------|-------------|-------------------|------------------|----------|
| very_high | 4585.84 | 600.78 | 6001.63 | 910.71 | 0.2065 |
| high | 4019.54 | 425.72 | 3653.04 | 790.85 | 0.2609 |
| mid | 2915.11 | 426.69 | 2476.63 | 829.27 | 0.2935 |
| low | 1798.09 | 480.97 | 1165.91 | 480.76 | 0.2391 |

**Weather contributions** (from `weather_stats`):

| Weather | mean_tourists | std_tourists | mean_excursionists | std_excursionists | freq_rel |
|---------|--------------|-------------|-------------------|------------------|----------|
| bad | 140.31 | 463.14 | -773.24 | 1187.06 | 0.15 |
| unsettled | 128.32 | 319.20 | 31.11 | 824.52 | 0.20 |
| good | 72.86 | 406.09 | 282.91 | 839.91 | 0.65 |

**Weekday contributions** (from `weekday_stats`):

| Weekday | mean_tourists | std_tourists | mean_excursionists | std_excursionists |
|---------|--------------|-------------|-------------------|------------------|
| monday | -362.66 | 112.47 | -391.15 | 791.84 |
| tuesday | -265.34 | 122.14 | -352.27 | 937.65 |
| wednesday | -147.92 | 247.55 | -465.63 | 733.97 |
| thursday | 92.23 | 233.82 | -380.35 | 779.59 |
| friday | 678.05 | 149.45 | -232.91 | 500.51 |
| saturday | 320.47 | 204.42 | 590.99 | 797.29 |
| sunday | -278.93 | 193.02 | 1240.27 | 1149.60 |

Example: on a bad-weather Friday in peak season (`very_high`), the expected tourist count is `mean = 4585.84 + 140.31 + 678.05 ≈ 5404`, with `std = sqrt(600.78² + 463.14² + 149.45²) ≈ 762`. The resulting truncated normal is clipped at 0 (left tail truncated at `−mean/std ≈ −7.1 σ`, i.e., effectively not truncated in practice).

### 3. Capacity Indexes (I_C_*) — "how much of each resource exists"

These are uncertain, modelled as continuous distributions:

| Python name | Type | Parameters | Effective range | Meaning |
|-------------|------|-----------|----------------|---------|
| `I_C_parking` | `UniformDistIndex` | loc=350, scale=100 | 350–450 | Vehicle spots |
| `I_C_beach` | `UniformDistIndex` | loc=6000, scale=1000 | 6000–7000 | Person-equivalents of beach area |
| `I_C_accommodation` | `LognormDistIndex` | s=0.125, loc=0, scale=5000 | ~5000, right-skewed | Accommodation units |
| `I_C_food` | `TriangDistIndex` | loc=3000, scale=1000, c=0.5 | 3000–4000, peak at 3500 | Restaurant seats |

When evaluating constraints, each capacity index is a probability distribution over possible capacity values. The constraint's satisfaction probability for a given point in the grid is computed as `P(capacity > u) = 1 - CDF_capacity(u)`, where `u` is the scalar value of the composite usage index (e.g. `I_U_parking`) evaluated at a specific (tourists, excursionists, scenario) triple. `u` is **not** a sample from `I_U_*`; it is the deterministic output of the usage formula node, resolved by the `Evaluation` engine for that triple. The result is a number in [0, 1]: the probability, under the capacity distribution, that actual capacity exceeds the computed demand.

### 4. Usage Factor Indexes (I_U_*) — "what fraction of visitors uses each resource"

| Python name | Value | Depends on | Description |
|-------------|-------|-----------|-------------|
| `I_U_tourists_parking` | 0.02 | — | Only 2% of tourists use parking (rest walk from hotel) |
| `I_U_excursionists_parking` | 0.55 if bad weather, else 0.80 | CV_weather | Fewer drive when it rains |
| `I_U_tourists_beach` | 0.25 if bad weather, else 0.50 | CV_weather | Beach use drops in rain |
| `I_U_excursionists_beach` | 0.35 if bad weather, else 0.80 | CV_weather | Strong weather effect |
| `I_U_tourists_accommodation` | 0.90 | — | 90% of tourists need beds |
| `I_U_tourists_food` | 0.20 | — | 20% eat at restaurants |
| `I_U_excursionists_food` | 0.80 if bad weather, else 0.40 | CV_weather | Eat indoors more in bad weather |

Weather-dependent values use the `piecewise()` function, which builds a conditional formula node in the computation graph.

### 5. Conversion and Rotation Indexes (I_Xa_*, I_Xo_*) — "people to resource-units"

| Python name | Type | Value | Description |
|-------------|------|-------|-------------|
| `I_Xa_tourists_per_vehicle` | `Index` | 2.5 | Tourists share cars |
| `I_Xa_excursionists_per_vehicle` | `Index` | 2.5 | Excursionists share cars |
| `I_Xo_tourists_parking` | `Index` | 1.02 | Parking rotation for tourists (they stay, so ~1) |
| `I_Xo_excursionists_parking` | `Index` | 3.5 | Parking rotation for excursionists (come and go) |
| `I_Xo_tourists_beach` | `UniformDistIndex` | loc=1.0, scale=2.0 | Beach rotation, uncertain (1–3) |
| `I_Xo_excursionists_beach` | `Index` | 1.02 | Beach rotation for excursionists (minimal) |
| `I_Xa_tourists_accommodation` | `Index` | 1.05 | Accommodation over-allocation factor |
| `I_Xa_visitors_food` | `Index` | 0.9 | Food service allocation factor |
| `I_Xo_visitors_food` | `Index` | 2.0 | Food service rotation (2 seatings/day) |

### 6. Presence Policy Indexes (I_P_*) — "policy levers"

| Python name | Value | Description |
|-------------|-------|-------------|
| `I_P_tourists_reduction_factor` | 1.0 | Multiplier for capping tourists (1.0 = no reduction) |
| `I_P_excursionists_reduction_factor` | 1.0 | Multiplier for capping excursionists |
| `I_P_tourists_saturation_level` | 10000 | Smooth saturation ceiling for tourists |
| `I_P_excursionists_saturation_level` | 10000 | Smooth saturation ceiling for excursionists |

Applied via saturation formula: `p_transformed = (p × rf) × sl / ((p × rf)³ + sl³)^(1/3)`. With rf=1.0 and sl=10000, this barely affects values below 10000. It's a placeholder for future policy scenarios (e.g., "what if we cap tourists at 5000?").

### 7. Composite Usage Indexes (formula nodes) — "total resource demand"

These are `Index` objects whose values are formula nodes built from the variables above. Their Python names are `I_U_parking`, `I_U_beach`, `I_U_accommodation`, and `I_U_food` (distinct from the usage *factor* indexes `I_U_tourists_*` / `I_U_excursionists_*` in section 4). They produce a scalar dimensioned in the same units as the corresponding capacity index (vehicle-slots, person-equivalents, accommodation units, restaurant seats).

**`I_U_parking`** — vehicles simultaneously occupying parking:
```
PV_tourists × I_U_tourists_parking / (I_Xa_tourists_per_vehicle × I_Xo_tourists_parking)
+ PV_excursionists × I_U_excursionists_parking / (I_Xa_excursionists_per_vehicle × I_Xo_excursionists_parking)
```

**`I_U_beach`** — simultaneous person-equivalent occupancy of beach area:
```
PV_tourists × I_U_tourists_beach / I_Xo_tourists_beach
+ PV_excursionists × I_U_excursionists_beach / I_Xo_excursionists_beach
```

**`I_U_accommodation`** — accommodation units required overnight:
```
PV_tourists × I_U_tourists_accommodation / I_Xa_tourists_accommodation
```

**`I_U_food`** — simultaneous restaurant seat demand (accounting for rotation and allocation):
```
(PV_tourists × I_U_tourists_food + PV_excursionists × I_U_excursionists_food)
/ (I_Xa_visitors_food × I_Xo_visitors_food)
```

### 8. Constraints — "the physics of sustainability"

Each constraint pairs a usage formula with a capacity:

| Python name | Usage index | Capacity index |
|-------------|-------------|----------------|
| `C_parking` | `I_U_parking` | `I_C_parking` |
| `C_beach` | `I_U_beach` | `I_C_beach` |
| `C_accommodation` | `I_U_accommodation` | `I_C_accommodation` |
| `C_food` | `I_U_food` | `I_C_food` |

A point (tourists, excursionists) is **sustainable** if ALL four constraints are satisfied simultaneously.

### 9. The Model object — `M_Base`

`M_Base` is an `OvertourismModel` that bundles everything:

```python
M_Base = OvertourismModel(
    "base model",
    cvs=[CV_weekday, CV_season, CV_weather],
    pvs=[PV_tourists, PV_excursionists],
    domain_indexes=[...all 20 scalar/piecewise indexes...],
    capacities=[I_C_parking, I_C_beach, I_C_accommodation, I_C_food],
    constraints=[C_parking, C_beach, C_accommodation, C_food],
)
```

---

## How `overtourism_molveno.py` runs, step by step

### Step 1: Define three scenarios (lines 46–53)

```python
S_Base = {}                                         # sample all weather types
S_Good_Weather = {CV_weather: ["good", "unsettled"]} # exclude bad weather
S_Bad_Weather = {CV_weather: ["bad"]}                # fix weather to bad
```

### Step 2: For each scenario, call `evaluate_scenario` (lines 282–287)

```python
result, scenarios = evaluate_scenario(M_Base, S_Base)
plot_scenario(axs[0], M_Base, result, scenarios, "Base")
```

Inside `evaluate_scenario`:

#### 2a. Create an Ensemble (line 185)

```python
scenarios = list(OvertourismEnsemble(model, situation, cv_ensemble_size=20))
```

An **ensemble** generates all combinations of context variable values by sampling each CV up to `cv_ensemble_size` values. Because the full support of every CV (7 weekdays, 4 seasons, 3 weather types) is smaller than `cv_ensemble_size=20`, the full Cartesian product is always returned: 7 × 4 × 3 = 84 combinations for `S_Base` (fewer when weather is restricted). If a CV had more categories than `cv_ensemble_size`, it would be randomly sampled down to that limit.

Each ensemble member is a `WeightedScenario`: a `(probability, assignments)` tuple where `assignments` is `{index → value, ...}` mapping each CV (and any pre-sampled distribution-backed index) to a concrete scalar, and `probability` is the product of the individual CV probabilities (independence assumption).

#### 2b. Build a grid (lines 186–187)

```python
tt = np.linspace(0, 10000, 101)   # 101 points for tourists
ee = np.linspace(0, 10000, 101)   # 101 points for excursionists
```

#### 2c. Evaluate on the grid (line 188)

```python
result = Evaluation(model).evaluate(scenarios, axes={PV_tourists: tt, PV_excursionists: ee})
```

The `Evaluation` class:
1. Takes each scenario's assignments and resolves all formula nodes in the computation graph
2. Broadcasts grid axes against scenario dimension
3. Returns an `EvaluationResult` with shape `(N_t, N_e, S)` where S = number of scenarios

Access results via `result[some_index]` → numpy array, and scenario weights via `result.weights` → shape (S,).

### Step 3: Compute sustainability field (lines 213–226 in `plot_scenario`)

For each constraint `c` (e.g. `C_parking`):

1. **Resolve usage**: `usage = result[c.usage]`, broadcast to shape `(N_t, N_e, S)`. Each entry is the deterministic scalar value of the composite usage formula (e.g. `I_U_parking`) for a specific `(tourists_grid_point, excursionists_grid_point, scenario)` triple.
2. **Compute per-scenario satisfaction probability** (`mask`, shape `(N_t, N_e, S)`):
   - If `c.capacity` is a `Distribution` (e.g. `I_C_parking ~ Uniform(350,450)`): `mask = 1 - CDF_capacity(usage)` = P(capacity > usage)
   - If `c.capacity` is a scalar constant: `mask = (usage <= capacity).astype(float)` (0 or 1)
3. **Average over scenarios** (weighted sum, not a probabilistic marginalisation): `field_elem = tensordot(mask, result.weights, axes=([-1], [0]))` → shape `(N_t, N_e)`. This is `Σ_s w_s × mask[t,e,s]`, where `w_s` is the scenario probability. The result is the scenario-averaged constraint satisfaction probability for each grid point.
4. **Combine constraints** (independence approximation): `field = field_parking × field_beach × field_accommodation × field_food`. Each element `field[t, e]` ≈ P(all four constraints satisfied) assuming constraints are independent given the visitor counts.

The result `field` is a 101×101 matrix where each value is between 0 (certainly unsustainable) and 1 (certainly sustainable). The multiplication assumes approximate independence between constraints.

### Step 4: Sample realistic visitor counts (lines 238–247)

For each scenario member, draw samples from the presence distributions:
```python
PV_tourists.sample(cvs=assignments, nr=max(1, round(w * 200)))
```
Apply saturation transformation, then scatter on the heatmap.

### Step 5: Compute summary statistics (lines 251–259)

| Statistic | Method | Meaning |
|-----------|--------|---------|
| **Sustainable area** | `field.sum() × cell_area` | How much of (tourists, excursionists) phase space is sustainable |
| **Sustainability index ± CI** | Interpolate field at sampled points, mean ± t-distribution CI (80%) | Expected sustainability given realistic visitor counts |
| **Per-constraint indices** | Same, for each constraint individually | Which resource is most stressed |
| **Critical constraint** | Lowest per-constraint index | The bottleneck resource |
| **Modal lines** | Contour at field = 0.5, linear regression fit | Phase boundaries between sustainable and unsustainable |

### Step 6: Plot (lines 261–288)

- Heatmap of the sustainability field (red = sustainable, blue = unsustainable)
- Black lines: modal lines (phase boundaries)
- Gray dots: sampled visitor counts (where real conditions fall)
- Title: sustainable area, sustainability index ± CI, critical constraint

### Configuration parameters

| Variable | Value | Purpose |
|----------|-------|---------|
| `t_max`, `e_max` | 10000 | Grid extent (tourists, excursionists) |
| `t_sample`, `e_sample` | 100 | Grid resolution → 101 points per axis |
| `target_presence_samples` | 200 | Total presence samples (weighted by scenario probability) |
| `ensemble_size` | 20 | CV ensemble samples per context variable |

---

## Where to find things in the code

### Core files and what lives in each

| File | What you find there |
| ---- | ------------------- |
| `examples/overtourism_molveno/molveno_model.py` | Every `CV_*`, `PV_*`, `I_C_*`, `I_U_*`, `I_Xa_*`, `I_Xo_*`, `I_P_*`, all four constraints, and `M_Base` |
| `examples/overtourism_molveno/molveno_presence_stats.py` | The season, weather, and weekday mean/std tables |
| `examples/overtourism_molveno/overtourism_metamodel.py` | `OvertourismModel`, `PresenceVariable`, `Constraint`, `OvertourismEnsemble`, `CategoricalContextVariable`, `UniformCategoricalContextVariable` |
| `examples/overtourism_molveno/overtourism_molveno.py` | `evaluate_scenario()`, `plot_scenario()`, the grid loop, sustainability field computation, and scatter sampling |
| `civic_digital_twins/dt_model/model/index.py` | `Index`, `ConstIndex`, `UniformDistIndex`, `LognormDistIndex`, `TriangDistIndex`, `TimeseriesIndex`, `Distribution` |
| `civic_digital_twins/dt_model/model/model.py` | Base `Model` class, `abstract_indexes()`, `is_instantiated()` |
| `civic_digital_twins/dt_model/simulation/evaluation.py` | `Evaluation`, `EvaluationResult` — grid resolution and scenario averaging |
| `civic_digital_twins/dt_model/simulation/ensemble.py` | `WeightedScenario` type alias, `DistributionEnsemble` |
| `civic_digital_twins/dt_model/engine/frontend/graph.py` | Low-level DAG construction (`placeholder`, `constant`, `function_call`, etc.) |
| `civic_digital_twins/dt_model/engine/frontend/linearize.py` | Topological sort of the computation graph |
| `civic_digital_twins/dt_model/engine/numpybackend/executor.py` | NumPy execution of the DAG (`State`, `evaluate_nodes`, `LambdaAdapter`) |

### Worked examples (best starting points for reading)

These files in `examples/doc/` are runnable scripts written specifically to explain the framework. Each section maps to a concept:

**`doc_getting_started.py`** — the simplest possible model end-to-end:

- §1: Define indexes and formula nodes → `Index`, `UniformDistIndex`
- §2: Build a `DistributionEnsemble` (samples distribution-backed indexes)
- §3: `Evaluation.evaluate()` and `result[index]` / `result.marginalize()`
- §4: `TimeseriesIndex` and custom functions via `graph.function_call` + `LambdaAdapter`

**`doc_overtourism_getting_started.py`** — minimal overtourism model with ensemble and sustainability field:

- §1: `CategoricalContextVariable` and `UniformCategoricalContextVariable` with probabilities
- §2: `PresenceVariable` with a `(season, weather) → (mean, std)` lookup table
- §3: `Constraint` built from a `TriangDistIndex` capacity and a `piecewise()` usage factor
- §4: Assembling an `OvertourismModel`
- §5: `OvertourismEnsemble` — the 2×3=6 Cartesian product, weights summing to 1.0
- §6: Grid evaluation with `axes={PV_visitors: visitors_axis}`, result shape `(201, 6)`
- §7: The full sustainability field loop — `result[c.usage]`, CDF evaluation, `tensordot` marginalisation

**`doc_model.py`** — index types and `DistributionEnsemble` in isolation:

- `ConstIndex`, distribution-backed `Index`, formula `Index`, placeholder `Index`
- `TimeseriesIndex` as a placeholder
- `DistributionEnsemble`: each scenario is `(weight, {index: scalar_value})`, all weights = 1/N

**`doc_engine.py`** — the low-level computation graph directly:

- `graph.placeholder`, `graph.constant`, arithmetic operators build a DAG
- `linearize.forest()` topologically sorts it
- `executor.State` + `executor.evaluate_nodes()` runs it with NumPy arrays
- `graph.function_call` + `LambdaAdapter` for custom operations (e.g. smoothing a time series)

### Key code locations for specific concepts

| Concept | Where to look |
| ------- | ------------- |
| How `piecewise()` works | `overtourism_metamodel.py` + used in `doc_overtourism_getting_started.py` §3 |
| How ensemble weights are computed | `doc_overtourism_getting_started.py` §5 — the `assert abs(weights.sum() - 1.0) < 1e-10` line |
| How `P(capacity > usage)` is computed | `doc_overtourism_getting_started.py` §7 — the `1.0 - c.capacity.value.cdf(usage)` line |
| How scenarios are averaged with `tensordot` | `doc_overtourism_getting_started.py` §7 — the `np.tensordot(mask, result.weights, ...)` line |
| How `cv_ensemble_size` limits/expands CV sampling | `doc_overtourism_getting_started.py` §5 comment: "cv_ensemble_size=10 >= support sizes 2 and 3, so all CV values are enumerated" |
| Full Molveno ensemble (7×4×3=84) | `overtourism_molveno.py` — `evaluate_scenario()` function |

---

## Key concepts mapped to physics

| Digital Twin concept | Physics analogy |
|---|---|
| (tourists, excursionists) space | Phase space |
| Sustainability field | Order parameter field (0 = disordered/unsustainable, 1 = ordered/sustainable) |
| Constraints | Boundary conditions or equations of state |
| Modal lines | Phase boundaries (critical contours) |
| Ensemble averaging over CVs | Canonical ensemble average over external conditions |
| Capacity distributions | Measurement uncertainties on system parameters |
| Context variables | External fields / control parameters |
| Sustainability index | Expectation value of the order parameter at the "natural" distribution |

---

## Integrating sensitivity analysis

The `sensitivity_analysis_framework.py` provides Sobol indices, PAWN indices, PRIM box peeling, and exceedance analysis for stochastic simulators. This section proposes how to connect the Molveno model to that framework.

### The challenge

The sensitivity framework expects a function with signature:

```python
def simulator_fn(params: dict, n_timesteps: int, rng: np.random.Generator) -> np.ndarray
```

where `params` is a dictionary of scalar parameter values and the return is a 1-D time series of length `n_timesteps`. The Molveno model, however:

1. Produces a **2-D spatial field** (sustainability over a tourists × excursionists grid), not a time series
2. Has **mixed parameter types**: some are constants, some are distributions, some depend on discrete context variables
3. Gets its stochasticity from **ensemble averaging** over discrete context variables, not from a random number generator

### Proposed approach

#### What are the parameters?

The uncertain quantities worth varying in a sensitivity analysis are:

| Parameter | Current value/distribution | Bounds for SA | Why it matters |
|-----------|---------------------------|---------------|----------------|
| `I_C_parking` (capacity) | Uniform(350, 100) | [300, 500] | Directly sets parking constraint |
| `I_C_beach` (capacity) | Uniform(6000, 1000) | [5000, 8000] | Directly sets beach constraint |
| `I_C_accommodation` (capacity) | Lognorm(s=0.125, scale=5000) | [4000, 6000] | Directly sets accommodation constraint |
| `I_C_food` (capacity) | Triang(3000, 1000, c=0.5) | [2500, 4500] | Directly sets food constraint |
| `I_Xo_tourists_beach` (rotation) | Uniform(1.0, 2.0) | [0.5, 4.0] | Uncertain beach turnover rate |
| `I_U_tourists_beach` (usage, good weather) | 0.50 | [0.3, 0.7] | Fraction of tourists at beach |
| `I_U_excursionists_parking` (usage, good weather) | 0.80 | [0.5, 1.0] | Fraction of excursionists driving |
| `I_U_excursionists_beach` (usage, good weather) | 0.80 | [0.5, 1.0] | Fraction of excursionists at beach |
| `I_Xa_tourists_per_vehicle` | 2.5 | [1.5, 4.0] | Car occupancy assumption |
| `I_Xo_excursionists_parking` | 3.5 | [2.0, 5.0] | Parking turnover assumption |

This gives ~10 parameters — tractable for Sobol analysis (Saltelli sampling needs N × (2D + 2) evaluations).

#### How to map the 2-D field to a 1-D output

The framework expects a time series. Several natural mappings exist:

**Option A — Diagonal slice (recommended).** Sweep total visitors along a diagonal in (tourists, excursionists) space, e.g., fix the ratio tourists:excursionists = 1:1 (or any other ratio) and evaluate sustainability from 0 to 10000 total visitors. The "timestep" axis becomes a visitor-count axis. This produces a natural 1-D curve that transitions from sustainable (1.0) to unsustainable (0.0) as load increases.

```python
def molveno_simulator(params, n_timesteps, rng):
    # Override model parameters with SA-sampled values
    # Build ensemble, evaluate on diagonal grid
    # Return sustainability[i] for i = 0..n_timesteps-1
    total = np.linspace(0, 10000, n_timesteps)
    tt = total * ratio_t   # e.g., 0.5
    ee = total * ratio_e   # e.g., 0.5
    # ... evaluate and return sustainability along this line
```

**Option B — Fix one presence variable.** Fix excursionists at a representative value (e.g., 3000) and sweep tourists from 0 to 10000. The "timestep" axis becomes the tourist axis.

**Option C — Scalar output, no time axis.** Compute the sustainability index (the single number currently reported in the plot title) and return it as a constant array. This still works with Sobol/PAWN but loses the spatial structure and makes time-windowed analysis meaningless.

#### Stochasticity for replicas

The framework runs `n_replicas` simulations per parameter set to separate aleatoric from epistemic uncertainty. The Molveno model's natural source of aleatoric noise is the **ensemble over context variables**. For each replica:

1. Draw a random subset of context variable combinations (or resample with replacement)
2. Draw random presence counts from the truncated normal distributions
3. This produces a different sustainability profile each time

The `rng` argument from the framework drives these random draws.

#### Exceedance analysis

The framework's exceedance analysis asks: "what is P(Y > threshold)?" In the Molveno context, the natural question is inverted — we care about **low** sustainability:

> **P(sustainability < c)** — what is the probability that the system becomes unsustainable?

Set the threshold to e.g., c = 0.5 (the point where sustainability drops below 50%). Then:
- **PAWN on exceedance** tells you which parameters most influence whether the system crosses the unsustainability threshold
- **PRIM box peeling** identifies the "danger zone" in parameter space — the combination of low capacities and high usage factors that most reliably produce unsustainable outcomes
- **Pairwise heatmaps** show interaction effects (e.g., does low parking capacity combined with high car occupancy create a compounding risk?)

#### Implementation sketch

```python
from sensitivity_analysis_framework import run_sensitivity_analysis, build_dash_app

def molveno_simulator(params, n_timesteps, rng):
    """Wrap Molveno model as a 1-D stochastic simulator."""
    # 1. Override capacity and usage parameters
    # 2. Build ensemble with random CV sampling (using rng)
    # 3. Evaluate on a diagonal slice through (tourists, excursionists) space
    # 4. Return 1-D sustainability profile

    # ... (override model index values with params dict)
    # ... (create ensemble, evaluate, compute field along diagonal)
    return sustainability_along_diagonal  # shape (n_timesteps,)

problem = {
    "num_vars": 10,
    "names": ["C_parking", "C_beach", "C_accommodation", "C_food",
              "Xo_beach", "U_tourists_beach", "U_exc_parking",
              "U_exc_beach", "Xa_per_vehicle", "Xo_exc_parking"],
    "bounds": [[300,500], [5000,8000], [4000,6000], [2500,4500],
               [0.5,4.0], [0.3,0.7], [0.5,1.0],
               [0.5,1.0], [1.5,4.0], [2.0,5.0]],
}

results = run_sensitivity_analysis(molveno_simulator, problem,
                                    n_timesteps=101, n_samples=1024,
                                    n_replicas=50)
app = build_dash_app(results, initial_threshold=0.5)
app.run(port=8000)
```

#### What would the dashboard show?

- **Sobol S1/ST on E[sustainability|params]**: Which parameters most affect the average sustainability profile. We expect capacities (especially parking — the usual bottleneck) to dominate.
- **Sobol on Var[sustainability|params]**: Which parameters most affect the *variability* of sustainability. Weather-dependent usage factors would likely rank high here.
- **PAWN indices**: Non-linear sensitivity — parameters whose full distribution of outcomes shifts (not just mean/variance).
- **Exceedance (P(sustainability < 0.5))**: The probability of crossing into unsustainability as a function of visitor load. PRIM would identify parameter sub-regions where this probability is highest.
- **Pairwise heatmaps**: Interaction effects between parameters (e.g., parking capacity vs. excursionist parking usage).

#### Key implementation considerations

1. **Parameter override mechanism**: The `Evaluation` class already supports value overrides through scenario assignments. Capacity indexes (which are distributions) need special handling — when the SA framework provides a fixed value, replace the distribution with that scalar for the evaluation.

2. **Performance**: Each SA evaluation needs an ensemble computation. With 101 grid points, 84 ensemble members, and ~20,000 Saltelli samples × 50 replicas, this is ~84 million evaluations. Consider:
   - Reducing grid resolution (e.g., 51 points)
   - Reducing ensemble size
   - Vectorising the evaluation (the current framework already broadcasts well)

3. **Weather conditioning**: Could run SA separately for each weather scenario (base, good, bad) to see how sensitivity changes with weather. This adds interpretive value without increasing computational cost per run.

---

## Changes needed in dt_model for replica support — assessment and implementation

### The problem

The sensitivity analysis framework (`sensitivity_analysis_framework_PAWN.py`) requires a function with signature:

```python
def simulator_fn(params: dict, n_timesteps: int, rng: np.random.Generator) -> np.ndarray
```

The framework calls this function N_REPLICAS times per parameter set with different `rng` values. Each call must return a different stochastic outcome so the framework can separate **epistemic uncertainty** (E[Y|θ]) from **aleatoric uncertainty** (Var[Y|θ]).

The Molveno model produces a 2-D spatial field, not a 1-D time series, and has no concept of replicas. Below is the approach taken and why it is the most faithful mapping.

### Design decision: one random CV combo per replica

Two approaches were considered for what makes each replica different:

**Option A (chosen): Single random CV combo per replica.** Each replica draws one random (weekday, season, weather) combination. The sustainability profile changes because weather switches piecewise usage factors (e.g. beach usage 0.80 in good weather vs 0.35 in bad weather). This means:

- Mean across replicas ≈ ensemble-weighted average (the same quantity the original model computes via weighted tensordot)
- Variance across replicas = day-to-day variability from weather/season/weekday
- Natural interpretation: each replica = "what happens on this particular random day"

**Option B (rejected): Full 84-member ensemble average per replica, re-sample distributions.** Each replica computes the full weighted average over all CV combos. The only variation would come from re-sampling distribution-backed indexes (capacities). But the SA framework already sweeps capacities as parameters. With capacities fixed by SA and only `I_Xo_tourists_beach` (Uniform(1,2)) remaining as a distribution, replicas would be nearly identical — the framework can't separate epistemic from aleatoric uncertainty.

Option A is clearly more faithful: the SA parameters capture epistemic uncertainty (what are the true capacities, usage factors, conversion rates?), while the replicas capture aleatoric uncertainty (what weather/season/weekday is it today?).

### Design decision: line through (tourists, excursionists) space

The framework expects a 1-D "timeseries" output. The Molveno model's output is a 2-D sustainability field. The mapping used: sweep total visitors along a line in (tourists, excursionists) space with a fixed ratio (default 50/50). The "timestep" axis becomes a total-visitor-count axis.

This produces a natural 1-D curve that transitions from 1.0 (sustainable at low visitor counts) to 0.0 (unsustainable at high counts). The transition point depends on both the SA parameters and the random CV combo, which is exactly what the SA framework needs.

### Design decision: bypass OvertourismEnsemble

The wrapper does NOT use `OvertourismEnsemble` at all. Instead, it builds a single `WeightedScenario` manually:

1. Uses `rng.choice()` (numpy) to pick one CV combo with proper probabilities
2. Passes PV values (the line) as arrays in the scenario assignment
3. Passes SA-provided capacity/rotation values as scenario assignments (for abstract indexes) or as `Evaluation.overrides` (for constant indexes)

This avoids changing `OvertourismEnsemble` and keeps all SA-specific logic in the wrapper. The SA framework's outer replica loop handles the multiple calls with different `rng`.

### Actual change to dt_model: `Evaluation.evaluate()` overrides parameter

The one change needed in the core framework is an `overrides` parameter on `Evaluation.evaluate()`:

```python
def evaluate(self, scenarios, nodes_of_interest=None, *, axes=None,
             functions=None, overrides=None):
```

**Why this is needed:** The SA framework needs to override constant indexes (e.g., `I_Xa_tourists_per_vehicle = 2.5`) with SA-sampled values. Constant indexes use `graph.constant` nodes — they are NOT abstract, so they cannot be overridden via scenario assignments. The `overrides` dict injects values directly into the executor's substitution dict, where they take precedence over baked-in constant values.

**How it works:** The executor checks `state.values[node]` before computing any node. If an override is present for a `graph.constant` node, the executor uses the override instead of the constant. Formula nodes that depend on the overridden constant then pick up the new value automatically.

**Implementation** (in `civic_digital_twins/dt_model/simulation/evaluation.py`):

```python
# After building c_subs from axes and abstract indexes:
if overrides:
    for idx, val in overrides.items():
        c_subs[idx.node] = np.asarray(val)
```

With `overrides=None` (default), behaviour is identical to before. No other dt_model changes are required.

### SA parameters

The 10 parameters swept by the sensitivity analysis:

| SA name | Model index | Default value | SA bounds | Type |
|---------|-------------|---------------|-----------|------|
| C_parking | `I_C_parking` | Uniform(350,100) | [300, 500] | distribution → scalar override |
| C_beach | `I_C_beach` | Uniform(6000,1000) | [5000, 8000] | distribution → scalar override |
| C_accommodation | `I_C_accommodation` | Lognorm(s=0.125, scale=5000) | [4000, 6000] | distribution → scalar override |
| C_food | `I_C_food` | Triang(3000,1000,c=0.5) | [2500, 4500] | distribution → scalar override |
| Xo_beach | `I_Xo_tourists_beach` | Uniform(1.0, 2.0) | [0.5, 4.0] | distribution → scalar override |
| Xa_tourists_per_vehicle | `I_Xa_tourists_per_vehicle` | 2.5 | [1.5, 4.0] | constant → `overrides` |
| Xa_exc_per_vehicle | `I_Xa_excursionists_per_vehicle` | 2.5 | [1.5, 4.0] | constant → `overrides` |
| Xo_exc_parking | `I_Xo_excursionists_parking` | 3.5 | [2.0, 5.0] | constant → `overrides` |
| U_tourists_parking | `I_U_tourists_parking` | 0.02 | [0.005, 0.05] | constant → `overrides` |
| U_tourists_accommodation | `I_U_tourists_accommodation` | 0.90 | [0.7, 1.0] | constant → `overrides` |

Distribution-backed indexes (first 5) go into scenario assignments. Constant indexes (last 5) go into `Evaluation.overrides`. Weather-dependent piecewise indexes (e.g. `I_U_excursionists_parking`) are NOT overridden — they change naturally with the random CV combo, which is the desired aleatoric variation.

### Constraint evaluation in the wrapper

When the SA provides scalar capacity values, constraint satisfaction becomes binary (usage <= capacity → 1.0, else 0.0). This is correct because:

- With a scalar capacity, there's no distribution to compute `P(capacity > usage)` against
- The probability is recovered by averaging across replicas: if 70% of replicas have usage <= capacity at a given point, then `E[sustainability] ≈ 0.70`
- This is standard Monte Carlo estimation of the same quantity the original model computes analytically via CDF

### Implementation: `sensitivity_molveno.py`

The wrapper lives at `examples/overtourism_molveno/sensitivity_molveno.py`. Key structure:

```python
def molveno_simulator(params, n_timesteps, rng):
    # 1. Define line: total visitors 0→10000, split 50/50 tourists/excursionists
    total = np.linspace(0, LINE_MAX, n_timesteps)
    tt_line = total * RATIO_T
    ee_line = total * RATIO_E

    # 2. Random CV combo (numpy rng for reproducibility)
    wd = rng.choice(weekday_values, p=weekday_probs)
    sn = rng.choice(season_values, p=season_probs)
    wt = rng.choice(weather_values, p=weather_probs)

    # 3. Single scenario: CV assignments + PV arrays + SA capacity overrides
    assignments = {CV_weekday: wd, CV_season: sn, CV_weather: wt,
                   PV_tourists: tt_line, PV_excursionists: ee_line}
    # Distribution-backed abstracts → scenario assignments
    # Constant indexes → Evaluation overrides

    # 4. Evaluate computation graph (no axes — PVs are scenario values)
    result = Evaluation(M_Base).evaluate(scenarios, overrides=overrides)

    # 5. Binary constraint check: usage <= SA-provided capacity
    sustainability = np.ones(n_timesteps)
    for c in M_Base.constraints:
        usage = result.marginalize(c.usage)
        sustainability *= (usage <= params[cap_name]).astype(float)
    return sustainability
```

### Verified results (N=64, replicas=20, 51 timesteps)

Sobol total-order indices on E[sustainability|θ], time-averaged:

| Parameter | ST |
|-----------|-----|
| Xa_exc_per_vehicle | 0.71 |
| Xo_exc_parking | 0.35 |
| C_parking | 0.26 |
| U_tourists_parking | 0.11 |
| Xa_tourists_per_vehicle | 0.09 |
| Xo_beach | 0.03 |
| C_beach | 0.02 |
| U_tourists_accommodation | 0.02 |
| C_accommodation | 0.02 |
| C_food | 0.01 |

Parking is the binding constraint at the 50/50 tourist/excursionist ratio, dominated by excursionist car occupancy and parking turnover assumptions. Beach, accommodation, and food capacities are non-binding in this parameter range.
