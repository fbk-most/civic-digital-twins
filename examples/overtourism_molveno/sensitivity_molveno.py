"""Sensitivity analysis for the Molveno overtourism model.

This script connects the Molveno model to the PAWN sensitivity analysis
framework. It is designed to be simple to configure: you pick ONE constraint
to analyse and can optionally fix any SA parameter to a constant value
(removing it from the analysis).

## How it works — section by section

1. **Model imports** — loads the Molveno overtourism model objects (context
   variables, presence variables, indexes, constraints) that define the
   computation graph.

2. **Line configuration** — the simulator sweeps total visitors from 0 to
   LINE_MAX.  RATIO_T / RATIO_E control how they split into tourists vs
   excursionists.  The sweep produces a 1-D "sustainability profile" (the
   SA output).

3. **CV probability tables** — weekday/season/weather are drawn randomly for
   each SA replica so that stochastic variation is captured across replicas.

4. **Per-constraint parameter registries** — each constraint (parking, beach,
   accommodation, food) has its own set of SA parameters: the capacity plus
   the usage/conversion indexes that feed into that constraint's usage
   formula.  Only the parameters relevant to the selected constraint enter
   the SA; everything else is held at its default.

5. **User configuration** — ``ANALYSE_CONSTRAINT`` selects which single
   constraint to study.  ``FIXED_PARAMS`` lets you pin any SA parameter to a
   constant, removing it from the Sobol/PAWN exploration.

6. **Problem builder** — ``build_problem()`` constructs the SALib problem
   dict from the selected constraint's parameters minus any fixed ones.

7. **Simulator function** — called by the SA framework for every sample ×
   replica.  It builds a scenario, evaluates the computation graph, and
   returns a binary sustainability profile (1 = sustainable, 0 = not) for
   the single selected constraint.

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

from civic_digital_twins.dt_model import Evaluation
from civic_digital_twins.dt_model.model.index import Distribution

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
    I_C_accommodation,
    I_C_beach,
    I_C_food,
    I_C_parking,
    I_U_tourists_accommodation,
    I_U_tourists_parking,
    I_Xa_excursionists_per_vehicle,
    I_Xa_tourists_per_vehicle,
    I_Xo_excursionists_parking,
    I_Xo_tourists_beach,
    M_Base,
    PV_excursionists,
    PV_tourists,
)
from molveno_presence_stats import season, weather, weekday  # noqa: E402

# ============================================================================
# Section 2 — Line configuration
# ============================================================================
# The simulator sweeps total visitors along a line from 0 to LINE_MAX.
# RATIO_T and RATIO_E control the tourist / excursionist split.
LINE_MAX = 10_000
RATIO_T = 0.5
RATIO_E = 0.5

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
# Section 4 — Per-constraint SA parameter registries
# ============================================================================
# Each constraint has its own capacity index + the usage/conversion indexes
# that appear in its usage formula.  Only these are varied in the SA when
# that constraint is selected.
#
# Format:  "SA_param_name": (model_Index_object, [lower_bound, upper_bound])

PARAMS_PARKING = {
    "C_parking":               (I_C_parking,                    [300, 500]),
    "Xa_tourists_per_vehicle":  (I_Xa_tourists_per_vehicle,     [1.5, 4.0]),
    "Xa_exc_per_vehicle":       (I_Xa_excursionists_per_vehicle, [1.5, 4.0]),
    "Xo_exc_parking":           (I_Xo_excursionists_parking,    [2.0, 5.0]),
    "U_tourists_parking":       (I_U_tourists_parking,          [0.005, 0.05]),
}

PARAMS_BEACH = {
    "C_beach":    (I_C_beach,          [5000, 8000]),
    "Xo_beach":   (I_Xo_tourists_beach, [0.5, 4.0]),
}

PARAMS_ACCOMMODATION = {
    "C_accommodation":          (I_C_accommodation,          [4000, 6000]),
    "U_tourists_accommodation": (I_U_tourists_accommodation, [0.7, 1.0]),
}

PARAMS_FOOD = {
    "C_food": (I_C_food, [2500, 4500]),
}

# Lookup by constraint name.
CONSTRAINT_PARAMS = {
    "parking":       PARAMS_PARKING,
    "beach":         PARAMS_BEACH,
    "accommodation": PARAMS_ACCOMMODATION,
    "food":          PARAMS_FOOD,
}

# ============================================================================
# Section 5 — User configuration  *** EDIT THESE ***
# ============================================================================

# Which single constraint to analyse.
# One of: "parking", "beach", "accommodation", "food"
ANALYSE_CONSTRAINT = "parking"

# Fix any SA parameter to a constant value to remove it from the analysis.
# Example: {"C_parking": 400, "U_tourists_parking": 0.02}
# Leave empty to vary all parameters of the selected constraint.
FIXED_PARAMS: dict[str, float] = {}


# ============================================================================
# Section 6 — Problem builder
# ============================================================================
def build_problem(
    constraint_name: str,
    fixed: dict[str, float] | None = None,
) -> tuple[dict, dict[str, float]]:
    """Build the SALib problem dict for *one* constraint.

    Parameters
    ----------
    constraint_name:
        Name of the constraint to analyse (key in CONSTRAINT_PARAMS).
    fixed:
        Parameters to hold constant (excluded from the SA sweep).

    Returns
    -------
    problem:
        SALib-compatible dict with "num_vars", "names", "bounds".
    effective_fixed:
        Merged dict of all fixed values (user-supplied + defaults for
        parameters of *other* constraints that aren't being analysed).
    """
    if constraint_name not in CONSTRAINT_PARAMS:
        raise ValueError(
            f"Unknown constraint {constraint_name!r}.  "
            f"Choose from: {list(CONSTRAINT_PARAMS)}"
        )
    fixed = fixed or {}
    all_params = CONSTRAINT_PARAMS[constraint_name]

    names, bounds = [], []
    for pname, (_idx, pbnds) in all_params.items():
        if pname in fixed:
            continue  # held constant — skip
        names.append(pname)
        bounds.append(pbnds)

    if not names:
        raise ValueError("All parameters are fixed — nothing left to analyse.")

    problem = {"num_vars": len(names), "names": names, "bounds": bounds}

    # Build the full fixed-value dict: user-fixed + defaults for other
    # constraints' parameters (so the simulator always has every value).
    effective_fixed = dict(fixed)
    for other_name, other_params in CONSTRAINT_PARAMS.items():
        if other_name == constraint_name:
            continue
        for pname, (idx, _bnds) in other_params.items():
            if pname not in effective_fixed:
                val = idx.value
                if isinstance(val, Distribution):
                    effective_fixed[pname] = val.mean()
                else:
                    effective_fixed[pname] = float(val)

    return problem, effective_fixed


# Build the problem for the current configuration.
problem, _effective_fixed = build_problem(ANALYSE_CONSTRAINT, FIXED_PARAMS)

# Map *all* known SA parameter names → their model Index objects.
_ALL_PARAM_TO_INDEX = {}
for _pdict in CONSTRAINT_PARAMS.values():
    for _pname, (_idx, _) in _pdict.items():
        _ALL_PARAM_TO_INDEX[_pname] = _idx

# Which SA param gives the scalar capacity for each constraint.
_CONSTRAINT_CAP_PARAM = {
    "parking":       "C_parking",
    "beach":         "C_beach",
    "accommodation": "C_accommodation",
    "food":          "C_food",
}


# ============================================================================
# Section 7 — Simulator function
# ============================================================================
def molveno_simulator(
    params: dict[str, float],
    n_timesteps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evaluate sustainability for ONE constraint along the visitor sweep.

    Called by the SA framework for every (sample, replica) pair.

    Parameters
    ----------
    params:
        SA-sampled values for the *varied* parameters only.
    n_timesteps:
        Number of points along the visitor sweep.
    rng:
        Random generator for this replica (draws weekday/season/weather).

    Returns
    -------
    np.ndarray of shape (n_timesteps,)
        Binary sustainability profile: 1 = sustainable, 0 = not.
    """
    # Merge SA-varied params with fixed params so every index has a value.
    full_params = {**_effective_fixed, **params}

    # --- 7a. Build the visitor sweep line ---
    total = np.linspace(0, LINE_MAX, n_timesteps)
    tt_line = total * RATIO_T
    ee_line = total * RATIO_E

    # --- 7b. Draw a random context-variable combination ---
    wd = rng.choice(_weekday_values, p=_weekday_probs)
    sn = rng.choice(_season_values, p=_season_probs)
    wt = rng.choice(_weather_values, p=_weather_probs)

    # --- 7c. Build scenario assignments ---
    # Context + presence variables are always assigned.
    # Distribution-backed indexes are overridden via scenario assignments;
    # constant indexes are overridden via Evaluation.overrides.
    assignments: dict = {
        CV_weekday: wd,
        CV_season:  sn,
        CV_weather: wt,
        PV_tourists:      tt_line,
        PV_excursionists: ee_line,
    }
    overrides: dict = {}
    for sa_name, idx in _ALL_PARAM_TO_INDEX.items():
        if sa_name not in full_params:
            continue
        if isinstance(idx.value, Distribution):
            assignments[idx] = full_params[sa_name]
        elif idx.value is not None:
            overrides[idx] = full_params[sa_name]

    scenario = [(1.0, assignments)]

    # --- 7d. Evaluate the computation graph ---
    nodes_of_interest = [c.usage for c in M_Base.constraints]
    result = Evaluation(M_Base).evaluate(
        scenario,
        nodes_of_interest=nodes_of_interest,
        overrides=overrides,
    )

    # --- 7e. Compute sustainability for the selected constraint only ---
    target = [c for c in M_Base.constraints if c.name == ANALYSE_CONSTRAINT][0]
    usage = result.marginalize(target.usage)
    cap_param = _CONSTRAINT_CAP_PARAM.get(target.name)
    if cap_param is not None and cap_param in full_params:
        cap = full_params[cap_param]
        sustainability = (usage <= cap).astype(float)
    elif isinstance(target.capacity.value, Distribution):
        sustainability = (1.0 - target.capacity.value.cdf(usage)).astype(float)
    else:
        cap = float(target.capacity.value)
        sustainability = (usage <= cap).astype(float)

    return sustainability


# ============================================================================
# Section 8 — Main
# ============================================================================
if __name__ == "__main__":
    # Accept constraint name from command line:  python sensitivity_molveno.py beach
    if len(sys.argv) > 1:
        ANALYSE_CONSTRAINT = sys.argv[1]
        problem, _effective_fixed = build_problem(ANALYSE_CONSTRAINT, FIXED_PARAMS)

    print(f"Constraint : {ANALYSE_CONSTRAINT}")
    print(f"SA params  : {problem['names']}")
    if _effective_fixed:
        fixed_display = {k: v for k, v in _effective_fixed.items()
                         if k in {p for d in CONSTRAINT_PARAMS.values() for p in d}}
        if fixed_display:
            print(f"Fixed      : {fixed_display}")
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
        n_timesteps=101,
        n_samples=256,
        n_replicas=100,
        seed=42,
    )
    print_summary(results)
    app = build_dash_app(results, initial_threshold=0.5)
    print("\nDash at http://127.0.0.1:8000")
    app.run(debug=False, port=8000)
