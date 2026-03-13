"""Sensitivity analysis for the Molveno overtourism model.

This script connects the Molveno model to the PAWN sensitivity analysis
framework. It is designed to be simple to configure: you pick ONE usage
formula to analyse and can optionally fix any SA parameter to a constant
value (removing it from the analysis).

## How it works — section by section

1. **Model imports** — loads the Molveno overtourism model objects (context
   variables, presence variables, indexes, usage nodes) that define the
   computation graph.

2. **Line configuration** — the simulator sweeps total visitors along the
   values in SERIES_AXIS.  RATIO_T / RATIO_E control how they split into
   tourists vs excursionists.  The sweep produces a 1-D usage profile
   (the SA output).

3. **CV probability tables** — weekday/season/weather are drawn randomly for
   each SA replica so that stochastic variation is captured across replicas.

4. **Per-usage parameter registries** — each usage formula (parking, beach,
   accommodation, food) has its own set of SA parameters: the usage and
   conversion indexes that feed into that formula.  Only the parameters
   relevant to the selected formula enter the SA; parameters of *other*
   formulas become "background noise" — sampled uniformly from their
   registry bounds in each replica (not fixed at defaults).

5. **User configuration** — ``ANALYSE_USAGE`` selects which single usage
   formula to study.  ``FIXED_PARAMS`` lets you pin any SA parameter to a
   constant, removing it from the Sobol/PAWN exploration.

6. **Problem builder** — ``build_problem()`` constructs the SALib problem
   dict from the selected usage formula's parameters minus any fixed ones.

7. **Simulator function** — called by the SA framework for every sample ×
   replica.  It builds a scenario, evaluates the computation graph, and
   returns raw projected usage for the selected formula.  Capacity
   comparison is done later via the dashboard's threshold exceedance.

8. **Main block** — runs the SA, prints a summary, and launches a Dash
   dashboard for interactive exploration.

Usage:
    python sensitivity_molveno.py                      # analyse "parking"
    python sensitivity_molveno.py beach                # analyse "beach"
    python sensitivity_molveno.py food                 # analyse "food"
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

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
    CV_season,
    CV_weather,
    CV_weekday,
    I_U_accommodation,
    I_U_beach,
    I_U_excursionists_parking,
    I_U_food,
    I_U_parking,
    I_U_tourists_accommodation,
    I_U_tourists_parking,
    I_Xa_excursionists_per_vehicle,
    I_Xa_tourists_per_vehicle,
    I_Xo_excursionists_parking,
    I_Xo_tourists_beach,
    PV_excursionists,
    PV_tourists,
)
from molveno_presence_stats import season, weather, weekday  # noqa: E402

# ============================================================================
# Section 2 — Line configuration
# ============================================================================
# The simulator sweeps total visitors along the values in SERIES_AXIS.
# RATIO_T and RATIO_E control the tourist / excursionist split.
RATIO_T = 1.0
RATIO_E = 1.0

# Model type tag: tells the SA framework how the series axis can be evaluated.
#   "piecewise"  — each series_axis point is independent; the framework may
#                  evaluate them in any order or individually.
#   "sequential" — the model has internal state that evolves over time (e.g. a
#                  traffic simulator); it must be run from 0 to max(series_axis)
#                  as a single trajectory.  The framework always passes the full
#                  series_axis and plots the result, but must not assume points
#                  can be evaluated in isolation.
MODEL_TYPE = "piecewise"

# Series axis: the x-axis values (total visitors) for the sweep.
# Passed directly to run_sensitivity_analysis as series_axis.
# Can be an int (number of points from 0 to 10 000), a list, a range, or an
# array — e.g. range(0, 10001, 100) for 101 evenly-spaced points, or
# [0, 500, 1000, 5000, 10000] to pick specific values.
SERIES_AXIS: int | list | range | np.ndarray = np.linspace(0, 10_000, 100)

# ============================================================================
# Section 3 — CV probability tables
# ============================================================================
# Context-variable values and their probabilities, used to draw a random
# (weekday, season, weather) combination for each SA replica.
_weekday_values = list(weekday)
_weekday_probs = np.full(len(_weekday_values), 1.0 / len(_weekday_values))

_season_values = list(season.keys())
_season_probs = np.array([season[v] for v in _season_values])

_weather_values = list(weather.keys())
_weather_probs = np.array([weather[v] for v in _weather_values])

# ============================================================================
# Section 4 — Per-usage parameter registries
# ============================================================================
# Each usage formula has its own set of usage and conversion indexes that
# appear in that formula.  Only these are varied in the SA when that formula
# is selected.  Capacity indexes are deliberately excluded: they do not
# appear in any usage formula and have no effect on projected usage.
#
# Format:  "SA_param_name": (model_Index_object, [lower_bound, upper_bound])

PARAMS_PARKING = {
    "Xa_tourists_per_vehicle":  (I_Xa_tourists_per_vehicle,     [1.5, 4.0]),
    "Xa_exc_per_vehicle":       (I_Xa_excursionists_per_vehicle, [1.5, 4.0]),
    "Xo_exc_parking":           (I_Xo_excursionists_parking,    [2.0, 5.0]),
    "U_tourists_parking":       (I_U_tourists_parking,          [0.005, 0.05]),
    "U_exc_parking":            (I_U_excursionists_parking,     [0.005, 0.05]),
}

PARAMS_BEACH = {
    "Xo_beach":   (I_Xo_tourists_beach, [0.5, 4.0]),
}

PARAMS_ACCOMMODATION = {
    "U_tourists_accommodation": (I_U_tourists_accommodation, [0.7, 1.0]),
}

PARAMS_FOOD = {}  # food usage indexes have no uncertain parameters currently

# Lookup by usage formula name → usage node + parameter registry.
USAGE_PARAMS = {
    "parking":       PARAMS_PARKING,
    "beach":         PARAMS_BEACH,
    "accommodation": PARAMS_ACCOMMODATION,
    "food":          PARAMS_FOOD,
}

# Map usage formula name → the Index node that computes projected usage.
USAGE_NODE = {
    "parking":       I_U_parking,
    "beach":         I_U_beach,
    "accommodation": I_U_accommodation,
    "food":          I_U_food,
}

# ============================================================================
# Section 5 — User configuration  *** EDIT THESE ***
# ============================================================================

# Which single usage formula to analyse.
# One of: "parking", "beach", "accommodation", "food"
ANALYSE_USAGE = "beach"

# Fix any SA parameter to a constant value to remove it from the analysis.
# Example: {"U_tourists_parking": 0.02}
# Leave empty to vary all parameters of the selected usage formula.
FIXED_PARAMS: dict[str, float] = {}


# ============================================================================
# Section 6 — Problem builder
# ============================================================================
def build_problem(
    usage_name: str,
    fixed: dict[str, float] | None = None,
) -> tuple[dict, dict[str, object]]:
    """Build the SALib problem dict for *one* usage formula.

    Parameters
    ----------
    usage_name:
        Name of the usage formula to analyse (key in USAGE_PARAMS).
    fixed:
        Parameters to hold constant (excluded from the SA sweep).

    Returns
    -------
    problem:
        SALib-compatible dict with "num_vars", "names", "bounds".
    background_params:
        Dict mapping parameter name → (Index, bounds) for parameters of
        *other* usage formulas.  These are sampled uniformly from their
        registry bounds in each replica, contributing background noise
        without appearing in the SA exploration.
        User-fixed values from ``fixed`` are excluded (they get baked
        into the problem as constants, not background noise).
    """
    if usage_name not in USAGE_PARAMS:
        raise ValueError(
            f"Unknown usage formula {usage_name!r}.  "
            f"Choose from: {list(USAGE_PARAMS)}"
        )
    fixed = fixed or {}
    all_params = USAGE_PARAMS[usage_name]

    names, bounds = [], []
    for pname, (_, pbnds) in all_params.items():
        if pname in fixed:
            continue  # held constant — skip
        names.append(pname)
        bounds.append(pbnds)

    if not names:
        raise ValueError("All parameters are fixed — nothing left to analyse.")

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}

    # Background parameters: other usage formulas' params, sampled per replica.
    # Each entry is (Index_object, [lo, hi]) — always sampled uniformly from
    # bounds, regardless of whether the Index itself is a Distribution or scalar.
    background_params: dict[str, tuple] = {}
    for other_name, other_params in USAGE_PARAMS.items():
        if other_name == usage_name:
            continue
        for pname, (idx, bnds) in other_params.items():
            if pname not in fixed:
                background_params[pname] = (idx, bnds)

    return problem, background_params


# Build the problem for the current configuration.
problem, _background_params = build_problem(ANALYSE_USAGE, FIXED_PARAMS)

# Map *all* known SA parameter names → their model Index objects.
_ALL_PARAM_TO_INDEX = {}
for _pdict in USAGE_PARAMS.values():
    for _pname, (_idx, _) in _pdict.items():
        _ALL_PARAM_TO_INDEX[_pname] = _idx


# ============================================================================
# Section 7 — Simulator function
# ============================================================================
def molveno_simulator(
    params: dict[str, float],
    series_axis: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate projected usage for ONE formula along the visitor sweep.

    Called by the SA framework for every (sample, replica) pair.
    Returns raw usage values — capacity comparison is done later via
    the dashboard's threshold exceedance feature.

    Parameters
    ----------
    params:
        SA-sampled values for the *varied* parameters only.
    series_axis:
        Array of total-visitor values defining the x-axis of the sweep.
    rng:
        Random generator for this replica (draws weekday/season/weather).

    Returns
    -------
    np.ndarray of shape (len(series_axis),)
        Projected usage along the visitor sweep.
    """
    # Start from SA-sampled params + any user-pinned constants.
    full_params = {**FIXED_PARAMS, **params}

    # Sample background parameters (other usage formulas) uniformly from
    # their registry bounds so they contribute noise per replica.
    for bg_name, (_, bg_bounds) in _background_params.items():
        full_params[bg_name] = rng.uniform(bg_bounds[0], bg_bounds[1])

    # --- 7a. Build the visitor sweep line ---
    total = np.asarray(series_axis, dtype=float)
    tt_line = total * RATIO_T
    ee_line = total * RATIO_E

    # --- 7b. Draw a random context-variable combination ---
    wd = rng.choice(_weekday_values, p=_weekday_probs)
    sn = rng.choice(_season_values, p=_season_probs)
    wt = rng.choice(_weather_values, p=_weather_probs)

    # --- 7c. Build substitution dict ---
    # Map graph placeholder/constant nodes → concrete values.
    # Distribution-backed indexes (placeholders) and constant indexes
    # are both injected directly into the executor state.
    subs: dict = {
        CV_weekday.node: np.asarray(wd),
        CV_season.node:  np.asarray(sn),
        CV_weather.node: np.asarray(wt),
        PV_tourists.node:      np.asarray(tt_line),
        PV_excursionists.node: np.asarray(ee_line),
    }
    for sa_name, idx in _ALL_PARAM_TO_INDEX.items():
        if sa_name not in full_params:
            continue
        subs[idx.node] = np.asarray(full_params[sa_name])

    # --- 7d. Evaluate the computation graph ---
    usage_idx = USAGE_NODE[ANALYSE_USAGE]
    target_node = usage_idx.node
    state = executor.State(subs)
    executor.evaluate_nodes(state, *linearize.forest(target_node))

    # --- 7e. Return raw usage ---
    return np.asarray(state.values[target_node]).squeeze()


# ============================================================================
# Section 8 — Main
# ============================================================================
if __name__ == "__main__":
    # Accept usage formula name from command line:  python sensitivity_molveno.py beach
    if len(sys.argv) > 1:
        ANALYSE_USAGE = sys.argv[1]
        problem, _background_params = build_problem(ANALYSE_USAGE, FIXED_PARAMS)

    print(f"Usage formula : {ANALYSE_USAGE}")
    print(f"SA params     : {problem['names']}")
    if FIXED_PARAMS:
        print(f"Fixed         : {FIXED_PARAMS}")
    if _background_params:
        bg_summary = {k: bnds for k, (_idx, bnds) in _background_params.items()}
        print(f"Background (sampled uniformly per replica): {bg_summary}")
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
