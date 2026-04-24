"""Example to interactively explore the frozen Molveno model.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import time
from functools import reduce
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, ndimage, stats

matplotlib.use("Agg")  # must be called before any other matplotlib sub-imports

from civic_digital_twins.dt_model import Distribution, Evaluation

try:
    from overtourism_molveno.molveno_model import MolvenoModel
    from overtourism_molveno.overtourism_metamodel import OvertourismEnsemble
except ImportError:
    from molveno_model import MolvenoModel
    from overtourism_metamodel import OvertourismEnsemble

model = MolvenoModel()

# Base situation
S_Base = {}

# Good weather situation
S_Good_Weather = {model.cv_weather: ["good", "unsettled"]}

# Bad weather situation
S_Bad_Weather = {model.cv_weather: ["bad"]}

# PLOTTING

(t_max, e_max) = (10000, 10000)
(t_sample, e_sample) = (100, 100)
target_presence_samples = 200
ensemble_size = 20  # TODO: make configurable; may it be a CV parameter?


# ---------------------------------------------------------------------------
# Post-processing helpers (extracted from SustainabilityEvaluation)
# ---------------------------------------------------------------------------


def _compute_sustainable_area(field: np.ndarray, axes: dict) -> float:
    """Compute the sustainable area under the field."""
    return field.sum() * reduce(
        lambda x, y: x * y,
        [axis.max() / (axis.size - 1) + 1 for axis in axes.values()],
    )


def _compute_sustainability_index_with_ci(
    field: np.ndarray, axes: dict, presences: list, confidence: float = 0.9
) -> tuple[float, float]:
    """Return the sustainability index and its confidence half-width."""
    index = interpolate.interpn(
        list(axes.values()),
        field,
        np.array(presences),
        bounds_error=False,
        fill_value=0.0,
    )
    m, se = np.mean(index), stats.sem(index)
    h = se * stats.t.ppf((1 + confidence) / 2.0, index.size - 1)
    return float(m), float(h)


def _compute_sustainability_index_with_ci_per_constraint(
    field_elements: dict, axes: dict, presences: list, confidence: float = 0.9
) -> dict:
    """Return the sustainability index and CI half-width for each constraint."""
    result = {}
    for c, fe in field_elements.items():
        index = interpolate.interpn(
            list(axes.values()),
            fe,
            np.array(presences),
            bounds_error=False,
            fill_value=0.0,
        )
        m, se = np.mean(index), stats.sem(index)
        h = se * stats.t.ppf((1 + confidence) / 2.0, index.size - 1)
        result[c] = (float(m), float(h))
    return result


def _compute_modal_line_per_constraint(field_elements: dict, axes: dict) -> dict:
    """Compute the modal line for each constraint via orthogonal regression (first PC)."""
    axes_list = list(axes.values())  # [tt, ee]
    bounds = [axes_list[0].max(), axes_list[1].max()]
    modal_lines = {}

    for c, fe in field_elements.items():
        matrix = (fe <= 0.5) & (
            (ndimage.shift(fe, (0, 1)) > 0.5)
            | (ndimage.shift(fe, (0, -1)) > 0.5)
            | (ndimage.shift(fe, (1, 0)) > 0.5)
            | (ndimage.shift(fe, (-1, 0)) > 0.5)
        )
        (yi, xi) = np.nonzero(matrix)
        # yi = row indices (tourists axis 0), xi = col indices (excursionists axis 1)

        if len(yi) < 3:
            # Too few boundary points to fit a line.
            continue

        # Orthogonal regression: first principal component of the boundary cloud.
        pts = np.stack([axes_list[0][yi], axes_list[1][xi]], axis=1)  # (N, 2)
        centroid = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
        direction = Vt[0]  # unit vector in [tourists, excursionists] space

        # Clip the infinite line to the grid box [0, t_max] × [0, e_max].
        t_lo, t_hi = -np.inf, np.inf
        for i, bound in enumerate(bounds):
            if abs(direction[i]) > 1e-10:
                ta = -centroid[i] / direction[i]
                tb = (bound - centroid[i]) / direction[i]
                t_lo = max(t_lo, min(ta, tb))
                t_hi = min(t_hi, max(ta, tb))

        if t_lo >= t_hi:
            continue

        p0 = centroid + t_lo * direction  # [tourists, excursionists]
        p1 = centroid + t_hi * direction
        modal_lines[c] = ((p0[0], p1[0]), (p0[1], p1[1]))

    return modal_lines


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def evaluate_scenario(model, situation) -> tuple:
    """Evaluate the sustainability field for *situation*.

    Returns ``(result, ensemble)`` where *result* is an
    :class:`~dt_model.simulation.evaluation.EvaluationResult` and *ensemble*
    is the :class:`~overtourism_molveno.OvertourismEnsemble` used (needed
    downstream for presence-sample generation).
    """
    ensemble = OvertourismEnsemble(model, situation, cv_ensemble_size=ensemble_size)
    tt = np.linspace(0, t_max, t_sample + 1)
    ee = np.linspace(0, e_max, e_sample + 1)
    result = Evaluation(model).evaluate(
        ensemble=ensemble, parameters={model.pv_tourists: tt, model.pv_excursionists: ee}
    )
    return result, ensemble


def plot_scenario(model, result, scenarios, title):
    """Render the sustainability field and KPIs onto a figure.

    Parameters
    ----------
    model:
        The :class:`~overtourism_molveno.molveno_model.MolvenoModel` being plotted.
    result:
        :class:`~dt_model.simulation.evaluation.EvaluationResult` from
        :func:`evaluate_scenario`.
    scenarios:
        The list of weighted scenarios used in the evaluation (needed for
        presence-sample generation).
    title:
        Plot title prefix.

    Returns
    -------
    fig:
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 10), layout="constrained")
    tt = result.parameter_values[model.pv_tourists]
    ee = result.parameter_values[model.pv_excursionists]

    # Compute sustainability field.
    # field[t_idx, e_idx] = P(all constraints satisfied | tourists=tt[t_idx], excursionists=ee[e_idx])
    field = np.ones((tt.size, ee.size))
    field_elements = {}
    for c in model.constraints:
        usage = np.broadcast_to(result[c.usage], result.full_shape)
        if isinstance(c.capacity.value, Distribution):
            mask = (1.0 - c.capacity.value.cdf(usage)).astype(float)
        else:
            cap = np.broadcast_to(result[c.capacity], result.full_shape)
            mask = (usage <= cap).astype(float)
        field_elem = np.tensordot(mask, result.weights, axes=([-1], [0]))  # (N_t, N_e)
        field_elements[c] = field_elem
        field *= field_elem

    # Presence samples for scatter plot.
    def presence_transformation(presence, reduction_factor, saturation_level, sharpness=3):
        tmp = presence * reduction_factor
        return tmp * saturation_level / ((tmp**sharpness + saturation_level**sharpness) ** (1 / sharpness))

    rf_t = float(np.mean(result[model.i_p_tourists_reduction_factor]))
    sl_t = float(np.mean(result[model.i_p_tourists_saturation_level]))
    rf_e = float(np.mean(result[model.i_p_excursionists_reduction_factor]))
    sl_e = float(np.mean(result[model.i_p_excursionists_saturation_level]))

    ens_weights = scenarios.ensemble_weights[0]
    ens_assignments = scenarios.assignments()
    scenario_keys = list(ens_assignments.keys())
    sample_tourists, sample_excursionists = [], []
    for i, w in enumerate(ens_weights):
        cvs_i = {k: ens_assignments[k][i] for k in scenario_keys}
        nr = max(1, round(w * target_presence_samples))
        for s in model.pv_tourists.sample(cvs=cvs_i, nr=nr):
            sample_tourists.append(presence_transformation(s, rf_t, sl_t))
        for s in model.pv_excursionists.sample(cvs=cvs_i, nr=nr):
            sample_excursionists.append(presence_transformation(s, rf_e, sl_e))

    axes_dict = {model.pv_tourists: tt, model.pv_excursionists: ee}

    area = _compute_sustainable_area(field, axes_dict)
    (i, c_ci) = _compute_sustainability_index_with_ci(
        field, axes_dict, list(zip(sample_tourists, sample_excursionists)), confidence=0.8
    )
    sust_indexes = _compute_sustainability_index_with_ci_per_constraint(
        field_elements, axes_dict, list(zip(sample_tourists, sample_excursionists)), confidence=0.8
    )
    critical = min(sust_indexes, key=lambda x: sust_indexes[x][0])
    modals = _compute_modal_line_per_constraint(field_elements, axes_dict)

    # field has shape (N_t, N_e); pcolormesh expects (N_e, N_t) for meshgrid(tt, ee).
    xx, yy = np.meshgrid(tt, ee)
    ax.pcolormesh(xx, yy, field.T, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    for modal in modals.values():
        ax.plot(*modal, color="black", linewidth=2)
    ax.scatter(sample_excursionists, sample_tourists, color="gainsboro", edgecolors="black")
    ax.set_title(
        f"{title}\n"
        + f"area = {area / 10e6:.2f} kp$^2$ - "
        + f"Sustainability = {i * 100:.2f}% +/- {c_ci * 100:.2f}%\n"
        + f"Critical = {critical.capacity.name}"
        + f"({sust_indexes[critical][0] * 100:.2f}% +/- {sust_indexes[critical][1] * 100:.2f}%)",
        fontsize=12,
    )
    ax.set_xlim(left=0, right=t_max)
    ax.set_ylim(bottom=0, top=e_max)

    return fig


if __name__ == "__main__":
    start_time = time.time()

    _out = Path(__file__).parent / "output"
    _out.mkdir(exist_ok=True)

    result, scenarios = evaluate_scenario(model, S_Base)
    fig_base = plot_scenario(model, result, scenarios, "Base")
    fig_base.savefig(_out / "base.png", dpi=150)
    plt.close(fig_base)

    result, scenarios = evaluate_scenario(model, S_Good_Weather)
    fig_good_weather = plot_scenario(model, result, scenarios, "Good weather")
    fig_good_weather.savefig(_out / "good_weather.png", dpi=150)
    plt.close(fig_good_weather)

    result, scenarios = evaluate_scenario(model, S_Bad_Weather)
    fig_bad_weather = plot_scenario(model, result, scenarios, "Bad weather")
    fig_bad_weather.savefig(_out / "bad_weather.png", dpi=150)
    plt.close(fig_bad_weather)

    print("--- %s seconds ---" % (time.time() - start_time))
