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


How it works — section by section
---------------------------------

Section 1 — Imports
    Loads the Molveno computation graph objects.  Three categories:

    (a) Context variables (CV) — weekday, season, weather.
        Their nodes are placeholders: we draw a random value each replica.

    (b) Presence variables (PV) — tourists, excursionists.
        Their nodes are placeholders: we fill them from the x-axis.

    (c) Indexes — the parameters that appear in each usage formula.
        Three kinds of graph nodes, important for understanding overrides:

          scalar constant  — holds a fixed number (e.g. 0.02).
                             Overriding replaces it with the SA value.

          piecewise formula — depends on weather (e.g. 0.25 if bad, 0.50
                              otherwise).  Overriding REPLACES the weather
                              dependency with a single SA-sampled scalar.

          distribution      — a placeholder with no default (e.g.
                              UniformDistIndex).  It MUST be included in
                              SA_PARAMS or FIXED_PARAMS; otherwise the graph
                              engine cannot evaluate the formula.

    (d) Usage formula nodes — each wraps an arithmetic expression over
        (b) and (c).  These are the SA targets.

Section 2 — X-axis and model configuration
    RATIO_T / RATIO_E split total visitors into tourists / excursionists.
    SERIES_AXIS defines the x-axis values (total visitors) for the sweep.

Section 3 — CV probability tables
    Weekday / season / weather values and their probabilities.
    One combination is drawn per SA replica.

Section 4 — Parameter definitions
    Each usage formula has a registry listing EVERY index that appears in
    its computation graph, with a sampling range [lo, hi].

    Only the parameters of the SELECTED formula matter.  Indexes from other
    formulas are never evaluated — the graph engine (linearize.forest) only
    traverses nodes reachable from the target.  So there is no need for
    "background noise" from other formulas.

Section 5 — User configuration
    Pick which formula to study (ANALYSE) and optionally fix parameters.

Section 6 — SALib problem builder
    Builds the {"num_vars", "names", "bounds"} dict that SALib expects.

Section 7 — Simulator function
    The function the SA framework calls for each (sample, replica).
    It substitutes values into the computation graph, evaluates it,
    and returns the raw usage vector.

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
)
from molveno_presence_stats import season, weather, weekday  # noqa: E402


# ============================================================================
# Section 2 — X-axis and model configuration
# ============================================================================
# The simulator sweeps total visitors along SERIES_AXIS.
# RATIO_T and RATIO_E control the tourist / excursionist split.
#
#   Example: 5000 total visitors, RATIO_T=0.6, RATIO_E=0.4
#            → 3000 tourists, 2000 excursionists

RATIO_T = 1.0
RATIO_E = 0.0

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
# A random (weekday, season, weather) combination is drawn for each SA
# replica.  This captures stochastic variation from context variables
# across replicas.

_weekday_values = list(weekday)
_weekday_probs  = np.full(len(_weekday_values), 1.0 / len(_weekday_values))

_season_values = list(season.keys())
_season_probs  = np.array([season[v] for v in _season_values])

_weather_values = list(weather.keys())
_weather_probs  = np.array([weather[v] for v in _weather_values])


# ============================================================================
# Section 4 — Parameter definitions
# ============================================================================
# Each usage formula depends on a specific set of indexes.  Below we list
# ALL indexes that appear in each formula's computation graph, together
# with their SA sampling bounds [lo, hi].
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
}

# ── Accommodation usage ──
# Formula (molveno_model.py lines 118-121):
#   tourists * U_t_acc / Xa_t_acc

PARAMS_ACCOMMODATION = {
    "U_tourists_accommodation":  (I_U_tourists_accommodation,  [0.7, 1.0]),
    "Xa_tourists_accommodation": (I_Xa_tourists_accommodation, [0.8, 1.5]),
}

# ── Food usage ──
# Formula (molveno_model.py lines 123-127):
#   (tourists * U_t_food + excursionists * U_e_food) / (Xa_food * Xo_food)

PARAMS_FOOD = {
    "U_tourists_food":  (I_U_tourists_food,         [0.10, 0.40]),
    "U_exc_food":       (I_U_excursionists_food,    [0.20, 0.95]),     # piecewise
    "Xa_visitors_food": (I_Xa_visitors_food,        [0.5, 1.5]),
    "Xo_visitors_food": (I_Xo_visitors_food,        [1.0, 3.0]),
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
# Pre-build the problem and index lookup for the current configuration.
problem = build_problem(ANALYSE, FIXED_PARAMS)

_target_usage, _target_params = USAGE_FORMULAS[ANALYSE]
_PARAM_TO_INDEX = {pname: idx for pname, (idx, _) in _target_params.items()}


# ============================================================================
# Section 7 — Simulator function
# ============================================================================
def molveno_simulator(
    params: dict[str, float],
    series_axis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate projected usage for the selected formula along the visitor sweep.

    Called by the SA framework for every (sample, replica) pair.

    Steps:
        a) Build the visitor sweep from the series axis
        b) Draw a random context-variable combination (weekday, season, weather)
        c) Map all values into a substitution dict for the graph executor
        d) Linearize the target formula and evaluate the graph
        e) Return the raw usage vector

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
        Projected usage along the visitor sweep.
    """
    # Merge SA-sampled values with any user-fixed constants.
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
        idx = _PARAM_TO_INDEX[pname]
        subs[idx.node] = np.asarray(value)

    # --- 7d. Evaluate the computation graph ---
    # linearize.forest(target_node) returns ONLY nodes reachable from the
    # target.  Indexes from other formulas are never touched.
    target_node = _target_usage.node
    state = executor.State(subs)
    executor.evaluate_nodes(state, *linearize.forest(target_node))

    # --- 7e. Return raw usage ---
    return np.asarray(state.values[target_node]).squeeze()


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
