"""Euro-split parking pricing optimisation for the Bologna mobility model.

**Goal**: find the ``(cost_euro0, increment)`` pair that maximises daily
parking revenue ("collected fees"), where the fee schedule is

    cost_k = max(0,  cost_euro0 − k × increment)   for k = 0, 1, …, 6

so Euro-0 vehicles (most polluting) pay the highest fee and each step up
the Euro ladder grants a discount of *increment* euros.

The Bologna CDT model is treated as a **black box**: the *only* interface
used to evaluate the objective is ``Evaluation(model).evaluate(…)``,
wrapped by the existing :func:`evaluate` helper.

Two optimisation strategies are demonstrated and compared:

1. **Grid sweep** — evaluate the objective on a regular 2-D grid of
   ``(cost_euro0, increment)`` parameter pairs.  Grid points are computed
   in parallel with :class:`concurrent.futures.ProcessPoolExecutor`.

2. **Scipy** — refine the grid's best point with
   :func:`scipy.optimize.minimize` (Nelder-Mead), a gradient-free method
   that tolerates the stochastic noise introduced by Monte-Carlo sampling
   of the uncertain ``i_b_p50_cost`` index.

Run from the repository root::

    uv run python examples/mobility_bologna/optimization_euro_split.py
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # must be called before any other matplotlib sub-imports

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.optimize import minimize

from civic_digital_twins.dt_model import Index

try:
    from .mobility_bologna import BolognaModel
    from .mobility_bologna import evaluate as _bologna_evaluate
except ImportError:
    from mobility_bologna import BolognaModel  # type: ignore[no-redef]
    from mobility_bologna import evaluate as _bologna_evaluate  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Monte-Carlo ensemble sizes: smaller = faster, larger = less noisy objective.
ENSEMBLE_SIZE_GRID: int = 50  # used for each grid-point evaluation
ENSEMBLE_SIZE_OPT: int = 200  # used inside the Scipy optimiser

# 2-D search bounds
COST_EURO0_MIN: float = 0.5
COST_EURO0_MAX: float = 20.0
INCREMENT_MIN: float = 0.0
INCREMENT_MAX: float = 4.0

# Grid resolution (COST_EURO0_N × INCREMENT_N evaluations total)
COST_EURO0_N: int = 25
INCREMENT_N: int = 25

# Output directory for plots
OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Black-box objective
# ---------------------------------------------------------------------------
# Both functions are defined at module level so that ProcessPoolExecutor
# can pickle them for dispatch to worker processes.


def build_model(cost_euro0: float, increment: float) -> BolognaModel:
    """Build a :class:`BolognaModel` with the given Euro-class fee schedule.

    The fee for a vehicle of Euro class *k* is::

        cost_k = max(0,  cost_euro0 − k × increment)

    All other model parameters are taken from :meth:`BolognaModel.default_inputs`.

    Parameters
    ----------
    cost_euro0 : float
        Parking fee (€/vehicle) for Euro-0 vehicles (highest cost).
    increment : float
        Fee reduction (€) per Euro-class step.

    Returns
    -------
    BolognaModel
        Fully configured model instance.
    """
    costs = [Index(f"cost euro {k}", float(np.clip(cost_euro0 - k * increment, 0.0, None))) for k in range(7)]
    return BolognaModel(**{**BolognaModel.default_inputs(), "i_p_cost": costs})


def compute_income(cost_euro0: float, increment: float, ensemble_size: int = ENSEMBLE_SIZE_GRID) -> float:
    """Evaluate expected daily parking revenue via the CDT Evaluation interface.

    This is the **only** function that touches the CDT model layer.  Both the
    grid sweep and the Scipy optimiser call it to obtain objective values.

    Parameters
    ----------
    cost_euro0 : float
        Parking fee (€) for Euro-0 vehicles.
    increment : float
        Fee discount (€) per Euro-class step.
    ensemble_size : int, optional
        Number of Monte-Carlo samples drawn by :class:`DistributionEnsemble`
        to marginalise over the uncertain ``i_b_p50_cost`` index.

    Returns
    -------
    float
        Expected total collected fees (€/day), averaged over the ensemble.
    """
    model = build_model(cost_euro0, increment)
    result = _bologna_evaluate(model, size=ensemble_size)
    return float(result.marginalize(model.outputs.total_payed))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cost_euro0_axis = np.linspace(COST_EURO0_MIN, COST_EURO0_MAX, COST_EURO0_N)
    increment_axis = np.linspace(INCREMENT_MIN, INCREMENT_MAX, INCREMENT_N)
    n_workers = os.cpu_count() or 1

    # -----------------------------------------------------------------------
    # 1 — Grid sweep
    # -----------------------------------------------------------------------

    n_evals = COST_EURO0_N * INCREMENT_N
    print(
        f"\n{'=' * 60}\n"
        f"GRID SWEEP  ({COST_EURO0_N} × {INCREMENT_N} = {n_evals} evaluations, "
        f"ensemble_size={ENSEMBLE_SIZE_GRID}, workers={n_workers})\n"
        f"{'=' * 60}"
    )
    t_grid_start = time.perf_counter()

    grid_params = [(c0, incr) for c0 in cost_euro0_axis for incr in increment_axis]
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(compute_income, c0, incr, ENSEMBLE_SIZE_GRID) for c0, incr in grid_params]
        income_flat = [f.result() for f in futures]  # collected in submission order

    income_grid = np.array(income_flat, dtype=float).reshape(COST_EURO0_N, INCREMENT_N)
    t_grid = time.perf_counter() - t_grid_start

    best_ij = np.unravel_index(np.argmax(income_grid), income_grid.shape)
    best_c0_grid = float(cost_euro0_axis[best_ij[0]])
    best_incr_grid = float(increment_axis[best_ij[1]])
    best_income_grid = float(income_grid[best_ij])

    print(
        f"  Best  cost_euro0 = {best_c0_grid:.2f} €   "
        f"increment = {best_incr_grid:.2f} €/class   "
        f"income = {best_income_grid:.0f} €/day\n"
        f"  Elapsed: {t_grid:.1f} s"
    )

    # -----------------------------------------------------------------------
    # 2 — Scipy optimisation  (Nelder-Mead, warm-started from grid best)
    # -----------------------------------------------------------------------

    print(
        f"\n{'=' * 60}\nSCIPY (Nelder-Mead, warm-start from grid best, ensemble_size={ENSEMBLE_SIZE_OPT})\n{'=' * 60}"
    )

    n_scipy_evals = [0]
    _prev_x: list[np.ndarray | None] = [None]

    def _neg_income(x: np.ndarray) -> float:
        """Negated income for minimisation; only prints when parameters move meaningfully."""
        n_scipy_evals[0] += 1
        val = compute_income(float(x[0]), float(x[1]), ENSEMBLE_SIZE_OPT)
        # Suppress log flood when the simplex has collapsed to a single point:
        # only print when at least one parameter changes by more than 0.05 €.
        if _prev_x[0] is None or float(np.max(np.abs(x - _prev_x[0]))) > 0.05:
            print(f"  [{n_scipy_evals[0]:3d}]  cost_euro0={x[0]:.3f}  increment={x[1]:.3f}  income={val:.0f}")
            _prev_x[0] = x.copy()
        return -val

    t_scipy_start = time.perf_counter()
    scipy_result = minimize(
        _neg_income,
        x0=np.array([best_c0_grid, best_incr_grid]),
        method="Nelder-Mead",
        # fatol is set large so convergence is driven purely by xatol (parameter
        # precision).  The Monte-Carlo noise in the objective (~±1 k €/day with
        # ensemble_size=200) would otherwise prevent the simplex from ever
        # satisfying a tight function-value criterion.
        options={"xatol": 0.25, "fatol": 1e9, "maxiter": 100, "disp": False},
    )
    t_scipy = time.perf_counter() - t_scipy_start

    opt_c0 = float(scipy_result.x[0])
    opt_incr = float(scipy_result.x[1])
    opt_income = -float(scipy_result.fun)

    print(
        f"\n  Best  cost_euro0 = {opt_c0:.2f} €   "
        f"increment = {opt_incr:.2f} €/class   "
        f"income = {opt_income:.0f} €/day\n"
        f"  Converged: {scipy_result.success}   "
        f"Evaluations: {n_scipy_evals[0]}   "
        f"Elapsed: {t_scipy:.1f} s"
    )

    # -----------------------------------------------------------------------
    # Reference (default parameters)
    # -----------------------------------------------------------------------

    REF_C0, REF_INCR = 5.0, 0.25
    ref_income = compute_income(REF_C0, REF_INCR, ENSEMBLE_SIZE_OPT)

    print(
        f"\n{'=' * 60}\n"
        f"SUMMARY\n"
        f"{'=' * 60}\n"
        f"  Default  cost_euro0={REF_C0:.2f} €  increment={REF_INCR:.2f} €/class  "
        f"income={ref_income:.0f} €/day\n"
        f"  Grid     cost_euro0={best_c0_grid:.2f} €  increment={best_incr_grid:.2f} €/class  "
        f"income={best_income_grid:.0f} €/day  (+{best_income_grid - ref_income:.0f} €/day vs default)\n"
        f"  Scipy    cost_euro0={opt_c0:.2f} €  increment={opt_incr:.2f} €/class  "
        f"income={opt_income:.0f} €/day  (+{opt_income - ref_income:.0f} €/day vs default)\n"
    )

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------

    cc, ii = np.meshgrid(cost_euro0_axis, increment_axis, indexing="ij")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    fig.suptitle("Euro-split parking pricing optimisation — Bologna model", fontsize=14)

    # 2-D income heatmap
    im = axs[0].pcolormesh(cc, ii, income_grid, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=axs[0], label="Income (€/day)")
    # Feasibility boundary: line where Euro-6 cost = 0  (cost_euro0 = 6 × increment)
    feas_i = np.linspace(INCREMENT_MIN, min(COST_EURO0_MAX / 6, INCREMENT_MAX), 200)
    axs[0].plot(6 * feas_i, feas_i, "w--", linewidth=1.5, label="Euro-6 cost = 0 €")
    axs[0].scatter([best_c0_grid], [best_incr_grid], c="red", marker="*", s=250, zorder=5, label="Grid best")
    axs[0].scatter([opt_c0], [opt_incr], c="orange", marker="X", s=200, zorder=5, label="Scipy best")
    axs[0].scatter([REF_C0], [REF_INCR], c="white", marker="o", s=120, zorder=5, label="Default")
    axs[0].set_xlabel("cost_euro0 (€)")
    axs[0].set_ylabel("increment (€/class)")
    axs[0].set_title("Income surface (€/day)")
    axs[0].legend(fontsize=8)

    # 1-D slice along cost_euro0, at grid-best increment
    axs[1].plot(cost_euro0_axis, income_grid[:, best_ij[1]], color="steelblue", linewidth=2)
    axs[1].axvline(best_c0_grid, color="red", linestyle="--", label=f"Grid best ({best_c0_grid:.2f} €)")
    axs[1].axvline(opt_c0, color="orange", linestyle="--", label=f"Scipy best ({opt_c0:.2f} €)")
    axs[1].axvline(REF_C0, color="gray", linestyle=":", label=f"Default ({REF_C0:.2f} €)")
    axs[1].set_xlabel("cost_euro0 (€)")
    axs[1].set_ylabel("Income (€/day)")
    axs[1].set_title(f"Slice at increment = {best_incr_grid:.2f} €/class")
    axs[1].legend(fontsize=8)

    # 1-D slice along increment, at grid-best cost_euro0
    axs[2].plot(increment_axis, income_grid[best_ij[0], :], color="darkorange", linewidth=2)
    axs[2].axvline(best_incr_grid, color="red", linestyle="--", label=f"Grid best ({best_incr_grid:.2f} €)")
    axs[2].axvline(opt_incr, color="orange", linestyle="--", label=f"Scipy best ({opt_incr:.2f} €)")
    axs[2].axvline(REF_INCR, color="gray", linestyle=":", label=f"Default ({REF_INCR:.2f} €)")
    axs[2].set_xlabel("increment (€/class)")
    axs[2].set_ylabel("Income (€/day)")
    axs[2].set_title(f"Slice at cost_euro0 = {best_c0_grid:.2f} €")
    axs[2].legend(fontsize=8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "optimization_euro_split.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
