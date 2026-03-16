"""Sensitivity analysis for the Molveno overtourism model.

This script connects the Molveno usage formulas to the Sobol + PAWN
sensitivity analysis framework (sensitivity_analysis_framework_PAWN.py).

The approach is:

    1. Import the model elements: context variables, presence variables,
       indexes, and usage formula nodes from molveno_model.py.

    2. Define which indexes to vary and their sampling ranges.

    3. Write a simulator function that receives SA-sampled parameter values
       and an x-axis, evaluates the computation graph, and returns a usage
       vector.

    4. Hand the simulator to the SA framework.


How dt_model objects are used
------------------------------

The Molveno model is built with the civic_digital_twins dt_model library,
which represents computations as a lazy directed acyclic graph (DAG).

Each Index object owns two things:
  - .node   — a graph node (placeholder or constant/formula) that sits
               inside the computation DAG.
  - .value  — the default value or distribution for that node.

Three kinds of Index nodes matter here:

  scalar constant   Index (e.g. I_U_tourists_parking = 0.02)
      .node is a graph.constant — the DAG already has a fixed value.
      Overriding: put a concrete scalar into subs[idx.node]; the executor
      will use that instead of the baked-in constant.

  piecewise formula Index (e.g. I_U_excursionists_parking)
      .node is a formula node — the DAG computes the value from weather.
      Overriding: put a scalar into subs[idx.node]; the executor short-
      circuits the formula and uses the override directly.

  distribution Index (e.g. I_Xo_tourists_beach, I_C_beach)
      .node is a graph.placeholder with NO default.
      .value is a frozen scipy distribution.
      These MUST be given a concrete value (via subs or subtracted
      separately) — the graph engine cannot evaluate them otherwise.

The graph engine works like this:

    linearize.forest(target_node)
        → returns the topologically sorted list of nodes reachable from
          the target.  Only the current formula's subgraph is visited;
          indexes from other formulas are never touched.

    executor.State(subs)
        → a mutable dict of node→value.  Any node already in subs is
          treated as "already evaluated" and its value is used directly.
          This is how SA-sampled values are injected into the graph.

    executor.evaluate_nodes(state, *plan)
        → walks the plan and evaluates each node, storing results in
          state.values.  After this, state.values[target_node] holds
          the formula result.

Capacity indexes are distribution indexes, but they are NOT part of the
usage formula's DAG (they live one level up, in the Constraint objects).
So they cannot be injected via subs.  Instead they are treated as ordinary
SA parameters with a sampling range derived from their distribution's
support.  The simulator receives the SA-sampled capacity value in `params`
and subtracts it from the usage result explicitly:

    output = usage(visitors, ...) − capacity_sample

The SA framework and dashboard then see usage − capacity as the quantity
of interest.  A threshold of 0 means "usage equals capacity".


How it works — section by section
---------------------------------

Section 1 — Imports
    Loads the Molveno computation graph objects.

Section 2 — X-axis and model configuration
    RATIO_T / RATIO_E split total visitors into tourists / excursionists.
    SERIES_AXIS defines the x-axis values (total visitors) for the sweep.

Section 3 — CV probability tables
    Weekday / season / weather values and their probabilities.
    One combination is drawn per SA replica.

Section 4 — Parameter definitions
    Each usage formula has a registry listing every index that appears in
    its computation graph, PLUS the corresponding capacity index, with SA
    sampling bounds [lo, hi].

    Only the parameters of the SELECTED formula matter.  Indexes from other
    formulas are never evaluated.

Section 5 — User configuration
    Pick which formula to study (ANALYSE) and optionally fix parameters.

Section 6 — SALib problem builder
    Builds the {"num_vars", "names", "bounds"} dict that SALib expects.

Section 7 — Simulator function
    The function the SA framework calls for each (sample, replica).
    Evaluates the graph and returns usage − capacity.

Section 8 — Main
    Runs the SA, prints results, launches the Dash dashboard.


Usage examples:
    python sensitivity_molveno.py                # analyse default formula
    python sensitivity_molveno.py parking        # analyse parking usage
    python sensitivity_molveno.py food           # analyse food usage
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ── Graph engine ──────────────────────────────────────────────────────────────
# linearize.forest(node) — returns the topologically sorted list of nodes
#   reachable from `node`.  Only dependencies of that node are included;
#   unrelated parts of the model (other formulas, capacity indexes) are
#   never visited.
#
# executor.State(subs) — holds a dict mapping graph nodes → concrete values.
#   The executor evaluates each node in the linearized plan.  If a node's
#   value is already in `subs`, it is used directly (this is how we inject
#   SA-sampled parameter values and CV/PV values).
#
# executor.evaluate_nodes(state, *plan) — runs the plan and stores results
#   in state.values.

from civic_digital_twins.dt_model.engine.frontend import linearize
from civic_digital_twins.dt_model.engine.numpybackend import executor


# ============================================================================
# Section 1 — Model imports
# ============================================================================
# Allow running from the repo root or from this directory.
_this_dir = str(Path(__file__).resolve().parent)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from molveno_model import (  # noqa: E402
    # ── Context variables ──
    CV_weekday,
    CV_season,
    CV_weather,
    # ── Presence variables ──
    PV_tourists,
    PV_excursionists,
    # ── Parking usage indexes ──
    I_U_tourists_parking,               # scalar 0.02
    I_U_excursionists_parking,          # piecewise on weather (0.55 / 0.80)
    I_Xa_tourists_per_vehicle,          # scalar 2.5
    I_Xa_excursionists_per_vehicle,     # scalar 2.5
    I_Xo_tourists_parking,             # scalar 1.02
    I_Xo_excursionists_parking,        # scalar 3.5
    # ── Beach usage indexes ──
    I_U_tourists_beach,                 # piecewise on weather (0.25 / 0.50)
    I_U_excursionists_beach,            # piecewise on weather (0.35 / 0.80)
    I_Xo_tourists_beach,               # UniformDist placeholder [1.0, 3.0]
    I_Xo_excursionists_beach,          # scalar 1.02
    # ── Accommodation usage indexes ──
    I_U_tourists_accommodation,        # scalar 0.90
    I_Xa_tourists_accommodation,       # scalar 1.05
    # ── Food usage indexes ──
    I_U_tourists_food,                  # scalar 0.20
    I_U_excursionists_food,             # piecewise on weather (0.40 / 0.80)
    I_Xa_visitors_food,                 # scalar 0.9
    I_Xo_visitors_food,                # scalar 2.0
    # ── Usage formula nodes (expressions over the indexes above) ──
    I_U_parking,
    I_U_beach,
    I_U_accommodation,
    I_U_food,
    # ── Capacity indexes (stochastic — sampled once per replica) ──
    I_C_parking,        # Uniform(350, 100)
    I_C_beach,          # Uniform(6000, 1000)
    I_C_accommodation,  # Lognorm(s=0.125, loc=0, scale=5000)
    I_C_food,           # Triangular(loc=3000, scale=1000, c=0.5)
)
from molveno_presence_stats import season, weather, weekday  # noqa: E402


# ============================================================================
# Section 2 — X-axis and model configuration
# ============================================================================
# This block defines the input domain for the sensitivity sweep. SERIES_AXIS
# is the x-axis: the range of total visitor counts that the simulator will
# evaluate for every SA sample. RATIO_T and RATIO_E scale that total into the
# tourist and excursionist sub-populations fed to the dt_model presence
# variables (PV_tourists, PV_excursionists). MODEL_TYPE tells the SA framework
# whether each x-axis point is evaluated independently ("piecewise") or must
# be run as an ordered trajectory ("sequential") — this affects how the
# framework parallelises and stores results.
#
#   Example: 5000 total visitors, RATIO_T=0.6, RATIO_E=0.4
#            → 3000 tourists, 2000 excursionists

RATIO_T = 1.0
RATIO_E = 1.0

# Model type tag (stored in SA results for downstream use):
#   "piecewise"  — each x-axis point is independent; the framework may
#                  evaluate them in any order or in parallel.
#   "sequential" — the model has internal state (e.g. a traffic sim);
#                  must run as a single trajectory from 0 to max(x).
MODEL_TYPE = "piecewise"

# Series axis: the x-axis values (total visitors).
SERIES_AXIS: int | list | range | np.ndarray = np.linspace(0, 10_000, 200)


# ============================================================================
# Section 3 — CV probability tables
# ============================================================================
# Context variables (CVs) in the dt_model represent external conditions the
# model cannot control — here: day-of-week, season, and weather. Rather than
# fixing them to a single value, the simulator draws a fresh random combination
# for every SA replica (see step 7b). These tables pre-compute the values and
# their sampling weights from molveno_presence_stats so that rng.choice() can
# pick a combination cheaply at runtime. Averaging over many replicas makes the
# SA indices reflect the *expected* sensitivity across realistic operating
# conditions, not just one specific day or weather scenario.

_weekday_values = list(weekday)
_weekday_probs  = np.full(len(_weekday_values), 1.0 / len(_weekday_values))

_season_values = list(season.keys())
_season_probs  = np.array([season[v] for v in _season_values])

_weather_values = list(weather.keys())
_weather_probs  = np.array([weather[v] for v in _weather_values])


# ============================================================================
# Section 4 — Parameter definitions
# ============================================================================
# This block is the bridge between the dt_model Index objects and the SA
# framework. Each PARAMS_* dict maps a human-readable parameter name to a
# (Index_object, [lo, hi]) pair. The Index_object carries the .node reference
# that the simulator injects into the graph executor's substitution dict; the
# bounds tell SALib the range to sample from. There is one dict per usage
# formula — only the dict for the formula selected in Section 5 is ever used,
# so indexes from other formulas are never touched at runtime. The capacity
# index for each formula is also listed here; because it lives outside the
# usage DAG it is subtracted from the result explicitly rather than injected
# into the graph.
#
# Format:  "param_name": (Index_object, [lower_bound, upper_bound])
#
#   Index_object — tells the simulator WHICH graph node to override.
#   [lo, hi]     — tells the SA framework the range to sample from.
#
# You can freely add, remove, or comment out entries:
#   - Removing an entry means that index keeps its model default
#     (constant or weather-dependent), and is NOT explored by the SA.
#   - Distribution placeholders (marked below) MUST be included here
#     or in FIXED_PARAMS — they have no default value.
#
# ── Parking usage ──
# Formula (molveno_model.py lines 104-110):
#   tourists * U_t_park / (Xa_t_veh * Xo_t_park)
#   + excursionists * U_e_park / (Xa_e_veh * Xo_e_park)

PARAMS_PARKING = {
    "U_tourists_parking":      (I_U_tourists_parking,          [0.005, 0.05]),
    "U_exc_parking":           (I_U_excursionists_parking,     [0.30, 0.90]),   # piecewise
    "Xa_tourists_per_vehicle": (I_Xa_tourists_per_vehicle,     [1.5, 4.0]),
    "Xa_exc_per_vehicle":      (I_Xa_excursionists_per_vehicle, [1.5, 4.0]),
    "Xo_tourists_parking":    (I_Xo_tourists_parking,          [0.5, 2.0]),
    "Xo_exc_parking":         (I_Xo_excursionists_parking,     [2.0, 5.0]),
    # Capacity — Uniform(loc=350, scale=100) → support [350, 450]
    "C_parking":               (I_C_parking,                   [350.0, 450.0]),  # distribution — REQUIRED
}

# ── Beach usage ──
# Formula (molveno_model.py lines 112-116):
#   tourists * U_t_beach / Xo_t_beach
#   + excursionists * U_e_beach / Xo_e_beach

PARAMS_BEACH = {
    "U_tourists_beach":   (I_U_tourists_beach,        [0.10, 0.70]),    # piecewise
    "U_exc_beach":        (I_U_excursionists_beach,   [0.15, 0.95]),    # piecewise
    "Xo_tourists_beach":  (I_Xo_tourists_beach,       [0.5, 4.0]),     # distribution — REQUIRED
    "Xo_exc_beach":       (I_Xo_excursionists_beach,  [0.5, 4.0]),
    # Capacity — Uniform(loc=6000, scale=1000) → support [6000, 7000]
    "C_beach":            (I_C_beach,                 [6000.0, 7000.0]),  # distribution — REQUIRED
}

# ── Accommodation usage ──
# Formula (molveno_model.py lines 118-121):
#   tourists * U_t_acc / Xa_t_acc

PARAMS_ACCOMMODATION = {
    "U_tourists_accommodation":  (I_U_tourists_accommodation,  [0.7, 1.0]),
    "Xa_tourists_accommodation": (I_Xa_tourists_accommodation, [0.8, 1.5]),
    # Capacity — Lognorm(s=0.125, loc=0, scale=5000) → 5th–95th pct ≈ [3700, 6800]
    "C_accommodation":           (I_C_accommodation,           [3700.0, 6800.0]),  # distribution — REQUIRED
}

# ── Food usage ──
# Formula (molveno_model.py lines 123-127):
#   (tourists * U_t_food + excursionists * U_e_food) / (Xa_food * Xo_food)

PARAMS_FOOD = {
    "U_tourists_food":  (I_U_tourists_food,         [0.10, 0.40]),
    "U_exc_food":       (I_U_excursionists_food,    [0.20, 0.95]),     # piecewise
    "Xa_visitors_food": (I_Xa_visitors_food,        [0.5, 1.5]),
    "Xo_visitors_food": (I_Xo_visitors_food,        [1.0, 3.0]),
    # Capacity — Triang(loc=3000, scale=1000, c=0.5) → support [3000, 4000]
    "C_food":           (I_C_food,                  [3000.0, 4000.0]),  # distribution — REQUIRED
}

# Maps formula name → (target usage node, parameter registry).
USAGE_FORMULAS = {
    "parking":       (I_U_parking,       PARAMS_PARKING),
    "beach":         (I_U_beach,         PARAMS_BEACH),
    "accommodation": (I_U_accommodation, PARAMS_ACCOMMODATION),
    "food":          (I_U_food,          PARAMS_FOOD),
}


# ============================================================================
# Section 5 — User configuration  *** EDIT THESE ***
# ============================================================================
# These two variables are the only settings you normally need to change.
# ANALYSE selects which usage formula (and its corresponding PARAMS_* dict and
# capacity index) the entire script will work with. FIXED_PARAMS lets you pin
# any subset of parameters to a known value so they are removed from the SA
# sweep — useful when you want to isolate the effect of a specific group of
# parameters or when one index already has a well-established value in your
# adaptation context.

# Which usage formula to study.
# One of: "parking", "beach", "accommodation", "food"
ANALYSE = "beach"

# Fix any parameter to a constant value (removes it from SA exploration).
# Example: {"Xo_tourists_beach": 2.0}
# Leave empty {} to vary all parameters of the selected formula.
FIXED_PARAMS: dict[str, float] = {}


# ============================================================================
# Section 6 — SALib problem builder
# ============================================================================
def build_problem(
    formula_name: str,
    fixed: dict[str, float] | None = None,
) -> dict:
    """Build the SALib problem dict for the selected usage formula.

    This function translates the PARAMS_* registry for the chosen formula into
    the ``{"num_vars", "names", "bounds"}`` dict that SALib's Sobol and PAWN
    samplers expect. It filters out any parameter that appears in ``fixed`` so
    that pinned parameters are invisible to the sampler — the simulator will
    fall back to their constant values instead. The result is stored in the
    module-level ``problem`` variable and passed directly to
    ``run_sensitivity_analysis`` in Section 8.

    Returns {"num_vars", "names", "bounds"} — the format SALib expects.
    Parameters in `fixed` are excluded from the SA sweep.
    """
    if formula_name not in USAGE_FORMULAS:
        raise ValueError(
            f"Unknown formula {formula_name!r}.  "
            f"Choose from: {list(USAGE_FORMULAS)}"
        )
    fixed = fixed or {}
    _, params = USAGE_FORMULAS[formula_name]

    names, bounds = [], []
    for pname, (_, pbnds) in params.items():
        if pname in fixed:
            continue
        names.append(pname)
        bounds.append(pbnds)

    if not names:
        raise ValueError("All parameters are fixed — nothing left to analyse.")

    return {"num_vars": len(names), "names": names, "bounds": bounds}


# ── Module-level setup ──
# These three lines pre-compute everything the simulator needs at import time
# so that no repeated dict lookups happen inside the hot per-sample loop.
# _target_usage holds the root node of the selected formula's DAG; the
# simulator linearises from that node. _PARAM_TO_INDEX maps each SA parameter
# name directly to its Index object so the simulator can resolve subs[idx.node]
# in one step. _capacity_param_name identifies which entry in full_params
# carries the capacity sample that must be subtracted after graph evaluation.
# Pre-build the problem and index lookup for the current configuration.
problem = build_problem(ANALYSE, FIXED_PARAMS)

_target_usage, _target_params = USAGE_FORMULAS[ANALYSE]
_PARAM_TO_INDEX = {pname: idx for pname, (idx, _) in _target_params.items()}
# Capacity param name for the selected formula (e.g. "C_beach").
# Its SA-sampled value is subtracted from usage in the simulator — it is NOT
# a node in the usage formula's graph, so it must not be injected via subs.
_capacity_param_name = f"C_{ANALYSE}"


# ============================================================================
# Section 7 — Simulator function
# ============================================================================
def molveno_simulator(
    params: dict[str, float],
    series_axis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate projected usage for the selected formula along the visitor sweep.

    This is the single function the SA framework calls for every (sample,
    replica) pair. It is the core adapter between the dt_model computation
    graph and SALib: it receives one row of SA-sampled parameter values,
    injects them into the graph executor via a substitution dict, evaluates
    the selected usage formula over the full visitor range, and subtracts the
    sampled capacity to produce a signed "margin" vector. A positive value
    means usage exceeds capacity at that visitor level; zero is the threshold.
    The SA framework aggregates these vectors across thousands of (sample,
    replica) calls to compute Sobol and PAWN sensitivity indices.

    Called by the SA framework for every (sample, replica) pair.

    Steps:
        a) Build the visitor sweep from the series axis
        b) Draw a random context-variable combination (weekday, season, weather)
        c) Map all values into a substitution dict for the graph executor
        d) Linearize the target formula and evaluate the graph
        e) Sample capacity once for this replica and return usage − capacity

    Parameters
    ----------
    params : dict
        SA-sampled values for the varied parameters.
    series_axis : np.ndarray
        X-axis values (total visitors).
    rng : np.random.Generator
        Per-replica random generator.

    Returns
    -------
    np.ndarray of shape (len(series_axis),)
        Usage minus capacity (positive = over capacity) along the visitor sweep.
    """
    # Merge SA-sampled values with any user-fixed constants.
    # FIXED_PARAMS (Section 5) is applied first so that SA-sampled values
    # always win over fixed ones — this prevents a mis-configured FIXED_PARAMS
    # from silently overriding a live SA parameter.
    full_params = {**FIXED_PARAMS, **params}

    # --- 7a. Build the visitor sweep ---
    total = np.asarray(series_axis, dtype=float)
    tt_line = total * RATIO_T          # tourists
    ee_line = total * RATIO_E          # excursionists

    # --- 7b. Draw random context variables ---
    wd = rng.choice(_weekday_values, p=_weekday_probs)
    sn = rng.choice(_season_values,  p=_season_probs)
    wt = rng.choice(_weather_values, p=_weather_probs)

    # --- 7c. Build substitution dict ---
    # Every placeholder node in the target formula's dependency tree must
    # have a concrete value here.  That includes:
    #   - Context variables (drawn above)
    #   - Presence variables (from the x-axis)
    #   - Any SA/fixed parameter whose Index has a placeholder node
    subs: dict = {
        CV_weekday.node:       np.asarray(wd),
        CV_season.node:        np.asarray(sn),
        CV_weather.node:       np.asarray(wt),
        PV_tourists.node:      np.asarray(tt_line),
        PV_excursionists.node: np.asarray(ee_line),
    }
    for pname, value in full_params.items():
        if pname == _capacity_param_name:
            continue  # capacity is not a node in the usage graph; handled below
        idx = _PARAM_TO_INDEX[pname]
        subs[idx.node] = np.asarray(value)

    # --- 7d. Evaluate the computation graph ---
    # linearize.forest(target_node) returns ONLY nodes reachable from the
    # target.  Indexes from other formulas are never touched.
    target_node = _target_usage.node
    state = executor.State(subs)
    executor.evaluate_nodes(state, *linearize.forest(target_node))

    # --- 7e. Subtract SA-sampled capacity and return usage − capacity ---
    capacity = full_params[_capacity_param_name]
    return np.asarray(state.values[target_node]).squeeze() - capacity


# ============================================================================
# Section 8 — Main
# ============================================================================
if __name__ == "__main__":
    # Accept formula name from command line:
    #   python sensitivity_molveno.py beach
    if len(sys.argv) > 1:
        ANALYSE = sys.argv[1]
        problem = build_problem(ANALYSE, FIXED_PARAMS)
        _target_usage, _target_params = USAGE_FORMULAS[ANALYSE]
        _PARAM_TO_INDEX = {pname: idx for pname, (idx, _) in _target_params.items()}

    # ── Summary ──
    print(f"Usage formula : {ANALYSE}")
    print(f"SA params     : {problem['names']}")
    if FIXED_PARAMS:
        print(f"Fixed         : {FIXED_PARAMS}")
    print()

    # Add repo root so the SA framework can be imported.
    _repo = str(Path(__file__).resolve().parent.parent.parent)
    if _repo not in sys.path:
        sys.path.insert(0, _repo)

    from sensitivity_analysis_framework_PAWN import (  # noqa: E402
        build_dash_app,
        print_summary,
        run_sensitivity_analysis,
    )

    results = run_sensitivity_analysis(
        simulator_fn=molveno_simulator,
        problem=problem,
        series_axis=SERIES_AXIS,
        n_samples=512,
        n_replicas=100,
        seed=42,
        model_type=MODEL_TYPE,
    )
    print_summary(results)
    app = build_dash_app(results)
    print("\nDash at http://127.0.0.1:8000")
    app.run(debug=False, port=8000)
