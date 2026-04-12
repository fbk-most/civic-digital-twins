"""Euro-split parking pricing optimisation for the Bologna mobility model.

**Goal**: find the ``(cost_euro0, increment)`` pair that maximises daily
parking revenue ("collected fees"), where the fee schedule is

    cost_k = max(0,  cost_euro0 − k × increment)   for k = 0, 1, …, 6

so Euro-0 vehicles (most polluting) pay the highest fee and each step up
the Euro ladder grants a discount of *increment* euros.

The Bologna CDT model is treated as a **black box**: the *only* interface
used to evaluate the objective is ``Evaluation(model).evaluate(…)``.

Two optimisation strategies are demonstrated and compared:

1. **Grid sweep** — evaluate the objective on a regular 2-D grid of
   ``(cost_euro0, increment)`` parameter pairs.  Because ``i_p_cost_euro0``
   and ``i_p_increment`` are explicit formula nodes inside :class:`BolognaModel`,
   the entire ``(COST_EURO0_N × INCREMENT_N)`` grid is evaluated in a **single
   vectorized** :meth:`~dt_model.simulation.evaluation.Evaluation.evaluate` call
   via the ``parameters=`` axis-sweep mechanism — no per-point re-evaluation,
   no worker processes.

2. **Scipy** — refine the grid's best point with
   :func:`scipy.optimize.minimize` (Nelder-Mead), a gradient-free method
   that tolerates the stochastic noise introduced by Monte-Carlo sampling
   of the uncertain ``i_b_p50_cost`` index.

Run from the repository root::

    uv run python examples/mobility_bologna/optimization_euro_split.py
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from pathlib import Path

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # must be called before any other matplotlib sub-imports

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from scipy.optimize import minimize

from civic_digital_twins.dt_model import DistributionEnsemble, Evaluation, Index
from civic_digital_twins.dt_model.engine.numpybackend import executor

try:
    from .mobility_bologna import BolognaModel, _ts_solve
except ImportError:
    from mobility_bologna import BolognaModel, _ts_solve  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Monte-Carlo ensemble sizes: smaller = faster, larger = less noisy objective.
ENSEMBLE_SIZE_GRID: int = 100  # used for the vectorized grid evaluation
ENSEMBLE_SIZE_OPT: int = 100  # used inside the Scipy optimiser

# 2-D search bounds
COST_EURO0_MIN: float = 0.5
COST_EURO0_MAX: float = 60.0
INCREMENT_MIN: float = 0.0
INCREMENT_MAX: float = 8.0

# Grid resolution (COST_EURO0_N × INCREMENT_N grid points evaluated in one pass)
COST_EURO0_N: int = 100
INCREMENT_N: int = 100

# Output directory for plots
OUTPUT_DIR = Path(__file__).parent / "output"


# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------


def pareto_front(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """Return a boolean mask of Pareto-optimal points when maximising both objectives.

    A point *i* is Pareto-optimal if no other point *j* satisfies
    ``f1[j] >= f1[i]`` **and** ``f2[j] >= f2[i]`` with at least one strict
    inequality (strict-dominance criterion).

    The algorithm is O(N log N): sort by ``f1`` descending (ties broken by
    ``f2`` descending), then scan left-to-right keeping a running maximum of
    ``f2``.  A point is non-dominated iff its ``f2`` strictly exceeds every
    ``f2`` seen at higher (or equal) ``f1`` values.

    Parameters
    ----------
    f1 : np.ndarray
        1-D array of objective-1 values (higher is better).
    f2 : np.ndarray
        1-D array of objective-2 values (higher is better), same length as *f1*.

    Returns
    -------
    np.ndarray
        Boolean mask of the same length; ``True`` marks Pareto-optimal points.
    """
    assert f1.shape == f2.shape and f1.ndim == 1, "f1 and f2 must be 1-D arrays of equal length"
    # Sort by f1 descending; break ties by f2 descending so that among
    # equal-f1 points the one with the best f2 is seen first.
    order = np.lexsort((-f2, -f1))
    mask = np.zeros(len(f1), dtype=bool)
    running_max_f2 = -np.inf
    for k in order:
        if f2[k] > running_max_f2:
            mask[k] = True
            running_max_f2 = f2[k]
    return mask


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Build a single model whose cost parameters will be swept as
    # PARAMETER axes by Evaluation.evaluate(parameters=...).
    #
    # The two Index objects below are passed both to BolognaModel (as the
    # default concrete values used in plain evaluate() calls) AND later to
    # Evaluation.evaluate(parameters=...) so the engine can substitute them
    # with full sweep arrays in a single vectorized pass.
    #
    # DistributionEnsemble ignores them because they are concrete (value ≠ None)
    # and therefore not returned by model.abstract_indexes().
    # ------------------------------------------------------------------
    i_p_cost_euro0 = Index("cost euro_0", 5.00)  # default; overridden by parameters=
    i_p_increment = Index("cost increment", 0.25)  # default; overridden by parameters=

    model = BolognaModel(
        **{
            **BolognaModel.default_inputs(),
            "i_p_cost_euro0": i_p_cost_euro0,
            "i_p_increment": i_p_increment,
        }
    )

    cost_euro0_axis = np.linspace(COST_EURO0_MIN, COST_EURO0_MAX, COST_EURO0_N)
    increment_axis = np.linspace(INCREMENT_MIN, INCREMENT_MAX, INCREMENT_N)

    # ------------------------------------------------------------------
    # Single-point evaluation helper (used by the Scipy optimiser and by
    # the reference-scenario evaluation at the end).
    #
    # Passes 1-element parameter arrays so the result shape is (1, 1)
    # after ensemble marginalisation; .item() extracts the Python float.
    # ------------------------------------------------------------------

    def compute_income(cost_euro0: float, increment: float, ensemble_size: int = ENSEMBLE_SIZE_GRID) -> float:
        """Return expected daily parking revenue for one ``(cost_euro0, increment)`` point.

        Parameters
        ----------
        cost_euro0 : float
            Parking fee (€) for Euro-0 vehicles.
        increment : float
            Fee discount (€) per Euro-class step.
        ensemble_size : int, optional
            Number of Monte-Carlo samples for ``i_b_p50_cost``.

        Returns
        -------
        float
            Expected total collected fees (€/day), averaged over the ensemble.
        """
        ensemble = DistributionEnsemble(model, ensemble_size)
        result = Evaluation(model).evaluate(
            ensemble=ensemble,
            parameters={
                i_p_cost_euro0: np.array([cost_euro0]),
                i_p_increment: np.array([increment]),
            },
            nodes_of_interest=[model.outputs.total_payed],
        )
        # marginalize() returns shape (1, 1) for two size-1 PARAMETER axes;
        # .item() converts the single element to a Python float.
        return float(result.marginalize(model.outputs.total_payed).item())

    # -----------------------------------------------------------------------
    # 1 — Grid sweep  (single vectorized evaluation pass)
    # -----------------------------------------------------------------------

    print(
        f"\n{'=' * 60}\n"
        f"GRID SWEEP  ({COST_EURO0_N} × {INCREMENT_N} = {COST_EURO0_N * INCREMENT_N} grid points, "
        f"ensemble_size={ENSEMBLE_SIZE_GRID}, single vectorized pass)\n"
        f"{'=' * 60}"
    )
    t_grid_start = time.perf_counter()

    # One Evaluation.evaluate() call sweeps the full (N_c0, N_incr) grid.
    # The engine broadcasts:
    #   i_p_cost_euro0  → shape (N_c0, 1)    [PARAMETER axis 0]
    #   i_p_increment   → shape (1, N_incr)   [PARAMETER axis 1]
    #   i_b_p50_cost    → shape (S,)           [ENSEMBLE axis]
    # and produces income_grid / emissions_grid of shape (N_c0, N_incr) after marginalisation.
    ensemble_grid = DistributionEnsemble(model, ENSEMBLE_SIZE_GRID)
    result_grid = Evaluation(model).evaluate(
        ensemble=ensemble_grid,
        parameters={
            i_p_cost_euro0: cost_euro0_axis,
            i_p_increment: increment_axis,
        },
        nodes_of_interest=[
            model.outputs.total_payed,
            model.outputs.total_modified_emissions,
            model.outputs.total_emissions,
        ],
        functions={"ts_solve": executor.LambdaAdapter(_ts_solve)},
    )
    income_grid = result_grid.marginalize(model.outputs.total_payed)
    emissions_grid = result_grid.marginalize(model.outputs.total_modified_emissions)
    # total_emissions is policy-independent; all grid cells share the same value
    # total_emissions is policy-independent (depends only on fixed baseline traffic),
    # so the engine computes it as a plain scalar with no PARAMETER/ENSEMBLE dims.
    # marginalize() cannot be used here — use direct state access instead.
    baseline_emissions = float(np.asarray(result_grid[model.outputs.total_emissions]).flat[0])
    emission_reduction_grid = baseline_emissions - emissions_grid
    # income_grid.shape == emissions_grid.shape == (COST_EURO0_N, INCREMENT_N)

    t_grid = time.perf_counter() - t_grid_start

    best_ij = np.unravel_index(np.argmax(income_grid), income_grid.shape)
    best_c0_grid = float(cost_euro0_axis[best_ij[0]])
    best_incr_grid = float(increment_axis[best_ij[1]])
    best_income_grid = float(income_grid[best_ij])
    best_emission_reduction_grid = float(emission_reduction_grid[best_ij])

    print(
        f"  Best  cost_euro0 = {best_c0_grid:.2f} €   "
        f"increment = {best_incr_grid:.2f} €/class   "
        f"income = {best_income_grid:.0f} €/day   "
        f"emission reduction = {best_emission_reduction_grid:.0f} kg/day\n"
        f"  Baseline emissions = {baseline_emissions:.0f} kg/day\n"
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

    fig, axs = plt.subplots(2, 3, figsize=(18, 11), layout="constrained")
    fig.suptitle("Euro-split parking pricing optimisation — Bologna model", fontsize=14)

    # ---- Row 0: Income ----

    # 2-D income heatmap
    im0 = axs[0, 0].pcolormesh(cc, ii, income_grid, cmap="viridis", shading="auto")
    fig.colorbar(im0, ax=axs[0, 0], label="Income (€/day)")
    feas_i = np.linspace(INCREMENT_MIN, min(COST_EURO0_MAX / 6, INCREMENT_MAX), 200)
    axs[0, 0].plot(6 * feas_i, feas_i, "w--", linewidth=1.5, label="Euro-6 cost = 0 €")
    axs[0, 0].scatter([best_c0_grid], [best_incr_grid], c="red", marker="*", s=250, zorder=5, label="Grid best")
    axs[0, 0].scatter([opt_c0], [opt_incr], c="orange", marker="X", s=200, zorder=5, label="Scipy best")
    axs[0, 0].scatter([REF_C0], [REF_INCR], c="white", marker="o", s=120, zorder=5, label="Default")
    axs[0, 0].set_xlabel("cost_euro0 (€)")
    axs[0, 0].set_ylabel("increment (€/class)")
    axs[0, 0].set_title("Income surface (€/day)")
    axs[0, 0].legend(fontsize=8)

    # 1-D income slice along cost_euro0, at grid-best increment
    axs[0, 1].plot(cost_euro0_axis, income_grid[:, best_ij[1]], color="steelblue", linewidth=2)
    axs[0, 1].axvline(best_c0_grid, color="red", linestyle="--", label=f"Grid best ({best_c0_grid:.2f} €)")
    axs[0, 1].axvline(opt_c0, color="orange", linestyle="--", label=f"Scipy best ({opt_c0:.2f} €)")
    axs[0, 1].axvline(REF_C0, color="gray", linestyle=":", label=f"Default ({REF_C0:.2f} €)")
    axs[0, 1].set_xlabel("cost_euro0 (€)")
    axs[0, 1].set_ylabel("Income (€/day)")
    axs[0, 1].set_title(f"Income slice at increment = {best_incr_grid:.2f} €/class")
    axs[0, 1].legend(fontsize=8)

    # 1-D income slice along increment, at grid-best cost_euro0
    axs[0, 2].plot(increment_axis, income_grid[best_ij[0], :], color="darkorange", linewidth=2)
    axs[0, 2].axvline(best_incr_grid, color="red", linestyle="--", label=f"Grid best ({best_incr_grid:.2f} €)")
    axs[0, 2].axvline(opt_incr, color="orange", linestyle="--", label=f"Scipy best ({opt_incr:.2f} €)")
    axs[0, 2].axvline(REF_INCR, color="gray", linestyle=":", label=f"Default ({REF_INCR:.2f} €)")
    axs[0, 2].set_xlabel("increment (€/class)")
    axs[0, 2].set_ylabel("Income (€/day)")
    axs[0, 2].set_title(f"Income slice at cost_euro0 = {best_c0_grid:.2f} €")
    axs[0, 2].legend(fontsize=8)

    # ---- Row 1: Emission reduction ----

    # 2-D emission-reduction heatmap
    im1 = axs[1, 0].pcolormesh(cc, ii, emission_reduction_grid, cmap="YlGn", shading="auto")
    fig.colorbar(im1, ax=axs[1, 0], label="Emission reduction (kg/day)")
    axs[1, 0].plot(6 * feas_i, feas_i, "k--", linewidth=1.5, label="Euro-6 cost = 0 €")
    axs[1, 0].scatter(
        [best_c0_grid], [best_incr_grid], c="red", marker="*", s=250, zorder=5, label="Grid best (income)"
    )
    axs[1, 0].scatter([REF_C0], [REF_INCR], c="white", marker="o", s=120, zorder=5, label="Default")
    axs[1, 0].set_xlabel("cost_euro0 (€)")
    axs[1, 0].set_ylabel("increment (€/class)")
    axs[1, 0].set_title("Emission reduction surface (kg/day)")
    axs[1, 0].legend(fontsize=8)

    # 1-D emission-reduction slice along cost_euro0, at grid-best increment
    axs[1, 1].plot(cost_euro0_axis, emission_reduction_grid[:, best_ij[1]], color="green", linewidth=2)
    axs[1, 1].axvline(best_c0_grid, color="red", linestyle="--", label=f"Grid best income ({best_c0_grid:.2f} €)")
    axs[1, 1].axvline(REF_C0, color="gray", linestyle=":", label=f"Default ({REF_C0:.2f} €)")
    axs[1, 1].set_xlabel("cost_euro0 (€)")
    axs[1, 1].set_ylabel("Emission reduction (kg/day)")
    axs[1, 1].set_title(f"Emission reduction slice at increment = {best_incr_grid:.2f} €/class")
    axs[1, 1].legend(fontsize=8)

    # 1-D emission-reduction slice along increment, at grid-best cost_euro0
    axs[1, 2].plot(increment_axis, emission_reduction_grid[best_ij[0], :], color="darkgreen", linewidth=2)
    axs[1, 2].axvline(best_incr_grid, color="red", linestyle="--", label=f"Grid best income ({best_incr_grid:.2f} €)")
    axs[1, 2].axvline(REF_INCR, color="gray", linestyle=":", label=f"Default ({REF_INCR:.2f} €)")
    axs[1, 2].set_xlabel("increment (€/class)")
    axs[1, 2].set_ylabel("Emission reduction (kg/day)")
    axs[1, 2].set_title(f"Emission reduction slice at cost_euro0 = {best_c0_grid:.2f} €")
    axs[1, 2].legend(fontsize=8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "optimization_euro_split.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")

    # -----------------------------------------------------------------------
    # 3 — Pareto frontier analysis
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 60}\nPARETO FRONTIER\n{'=' * 60}")

    # Flatten both objective grids to 1-D vectors of length N = COST_EURO0_N * INCREMENT_N.
    # cc / ii are the meshgrid coordinate arrays built for the plot above (indexing="ij"),
    # so c0_flat[k] / incr_flat[k] give the exact parameter pair for flat index k.
    f_income = income_grid.ravel()  # shape (N,)
    f_emission = emission_reduction_grid.ravel()  # shape (N,)
    c0_flat = cc.ravel()
    incr_flat = ii.ravel()

    pareto_mask = pareto_front(f_income, f_emission)
    pareto_idx = np.where(pareto_mask)[0]

    # Sort the Pareto front by income ascending so the tradeoff curve reads
    # left-to-right from max-emission-reduction to max-income.
    pareto_order = np.argsort(f_income[pareto_idx])
    pareto_idx = pareto_idx[pareto_order]

    pf_income = f_income[pareto_idx]
    pf_emission = f_emission[pareto_idx]
    pf_c0 = c0_flat[pareto_idx]
    pf_incr = incr_flat[pareto_idx]

    print(f"  Pareto-optimal points: {pareto_mask.sum()} / {len(f_income)}")

    # ------------------------------------------------------------------
    # Knee-point: the Pareto-front point closest to the utopia point
    # after min-max normalisation so both objectives contribute equally.
    # ------------------------------------------------------------------
    utopia_income = float(pf_income.max())
    utopia_emission = float(pf_emission.max())
    nadir_income = float(pf_income.min())
    nadir_emission = float(pf_emission.min())

    range_income = max(utopia_income - nadir_income, 1.0)
    range_emission = max(utopia_emission - nadir_emission, 1.0)

    norm_income = (pf_income - nadir_income) / range_income
    norm_emission = (pf_emission - nadir_emission) / range_emission

    dist_to_utopia = np.hypot(1.0 - norm_income, 1.0 - norm_emission)
    knee_k = int(np.argmin(dist_to_utopia))

    print(
        f"  Knee point  cost_euro0 = {pf_c0[knee_k]:.2f} €   "
        f"increment = {pf_incr[knee_k]:.2f} €/class   "
        f"income = {pf_income[knee_k]:.0f} €/day   "
        f"emission reduction = {pf_emission[knee_k]:.0f} kg/day"
    )

    # Flat indices for annotated reference points
    ref_flat_idx = int(
        np.ravel_multi_index(
            (int(np.argmin(np.abs(cost_euro0_axis - REF_C0))), int(np.argmin(np.abs(increment_axis - REF_INCR)))),
            income_grid.shape,
        )
    )
    best_income_flat_idx = int(np.ravel_multi_index(best_ij, income_grid.shape))

    # -----------------------------------------------------------------------
    # Pareto figure — three panels
    # -----------------------------------------------------------------------

    fig_p, axs_p = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
    fig_p.suptitle("Pareto frontier — income vs emission reduction  (Bologna parking model)", fontsize=14)

    # ---- Panel 0: Objective space (all grid points + highlighted front) ----
    ax = axs_p[0]
    sc = ax.scatter(
        f_income,
        f_emission,
        c=f_emission,
        cmap="YlGn",
        s=8,
        alpha=0.4,
        label="All grid points",
    )
    fig_p.colorbar(sc, ax=ax, label="Emission reduction (kg/day)")
    ax.plot(pf_income, pf_emission, "b-o", markersize=5, linewidth=1.5, label="Pareto front")
    ax.scatter(
        [f_income[best_income_flat_idx]],
        [f_emission[best_income_flat_idx]],
        c="red",
        marker="*",
        s=250,
        zorder=6,
        label="Max-income point",
    )
    ax.scatter(
        [pf_income[knee_k]],
        [pf_emission[knee_k]],
        c="orange",
        marker="D",
        s=180,
        zorder=6,
        label="Knee point",
    )
    ax.scatter(
        [f_income[ref_flat_idx]],
        [f_emission[ref_flat_idx]],
        c="white",
        edgecolors="black",
        marker="o",
        s=120,
        zorder=6,
        label="Default policy",
    )
    ax.set_xlabel("Income (€/day)")
    ax.set_ylabel("Emission reduction (kg/day)")
    ax.set_title("Objective space")
    ax.legend(fontsize=8)

    # ---- Panel 1: Parameter space — where the Pareto-optimal points live ----
    ax = axs_p[1]
    im_p = ax.pcolormesh(cc, ii, income_grid, cmap="viridis", shading="auto", alpha=0.5)
    fig_p.colorbar(im_p, ax=ax, label="Income (€/day)")
    ax.scatter(
        pf_c0,
        pf_incr,
        c=pf_emission,
        cmap="YlGn",
        s=40,
        zorder=5,
        edgecolors="black",
        linewidths=0.4,
        label="Pareto-optimal points",
    )
    ax.scatter([pf_c0[knee_k]], [pf_incr[knee_k]], c="orange", marker="D", s=180, zorder=6, label="Knee point")
    ax.scatter([REF_C0], [REF_INCR], c="white", marker="o", s=120, zorder=6, label="Default")
    feas_i_p = np.linspace(INCREMENT_MIN, min(COST_EURO0_MAX / 6, INCREMENT_MAX), 200)
    ax.plot(6 * feas_i_p, feas_i_p, "w--", linewidth=1.5, label="Euro-6 cost = 0 €")
    ax.set_xlabel("cost_euro0 (€)")
    ax.set_ylabel("increment (€/class)")
    ax.set_title("Parameter space — Pareto-optimal policies")
    ax.legend(fontsize=8)

    # ---- Panel 2: Tradeoff curve — clean 1-D view of the Pareto front ----
    ax = axs_p[2]
    ax.plot(pf_income, pf_emission, "b-o", markersize=5, linewidth=1.5)
    ax.scatter(
        [pf_income[knee_k]],
        [pf_emission[knee_k]],
        c="orange",
        marker="D",
        s=180,
        zorder=6,
        label=f"Knee  ({pf_income[knee_k]:.0f} €/day, {pf_emission[knee_k]:.0f} kg/day)",
    )
    ax.scatter(
        [f_income[best_income_flat_idx]],
        [f_emission[best_income_flat_idx]],
        c="red",
        marker="*",
        s=250,
        zorder=6,
        label=f"Max income  ({f_income[best_income_flat_idx]:.0f} €/day)",
    )
    ax.scatter(
        [f_income[ref_flat_idx]],
        [f_emission[ref_flat_idx]],
        c="gray",
        edgecolors="black",
        marker="o",
        s=120,
        zorder=6,
        label=f"Default  ({f_income[ref_flat_idx]:.0f} €/day)",
    )
    ax.set_xlabel("Income (€/day)")
    ax.set_ylabel("Emission reduction (kg/day)")
    ax.set_title("Tradeoff curve (Pareto front)")
    ax.legend(fontsize=8)

    out_path_p = OUTPUT_DIR / "optimization_pareto.png"
    fig_p.savefig(out_path_p, dpi=150, bbox_inches="tight")
    print(f"Pareto plot saved to {out_path_p}")
