"""Example to interactively explore the frozen Molveno model.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import time
from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy import interpolate, ndimage, stats

from civic_digital_twins.dt_model import Evaluation
from civic_digital_twins.dt_model.model.index import Distribution
from civic_digital_twins.dt_model.simulation.ensemble import WeightedScenario

try:
    from overtourism_molveno.overtourism_metamodel import OvertourismEnsemble
    from overtourism_molveno.molveno_model import (
        CV_weather,
        I_P_excursionists_reduction_factor,
        I_P_excursionists_saturation_level,
        I_P_tourists_reduction_factor,
        I_P_tourists_saturation_level,
        M_Base,
        PV_excursionists,
        PV_tourists,
    )
except ImportError:
    from overtourism_metamodel import OvertourismEnsemble
    from molveno_model import (
        CV_weather,
        I_P_excursionists_reduction_factor,
        I_P_excursionists_saturation_level,
        I_P_tourists_reduction_factor,
        I_P_tourists_saturation_level,
        M_Base,
        PV_excursionists,
        PV_tourists,
    )

# Base situation
S_Base = {}

# Good weather situation
S_Good_Weather = {CV_weather: ["good", "unsettled"]}

# Bad weather situation
S_Bad_Weather = {CV_weather: ["bad"]}

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
    """Compute the modal line for each constraint."""
    axes_list = list(axes.values())  # [tt, ee]
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

        horizontal_regr = None
        vertical_regr = None
        try:
            horizontal_regr = stats.linregress(axes_list[0][yi], axes_list[1][xi])
        except ValueError:
            pass
        try:
            vertical_regr = stats.linregress(axes_list[1][xi], axes_list[0][yi])
        except ValueError:
            pass

        def _vertical(regr) -> tuple[tuple[float, float], tuple[float, float]]:
            if regr.slope < 0.0:
                return ((regr.intercept, 0.0), (0.0, -regr.intercept / regr.slope))
            else:
                return ((regr.intercept, regr.intercept), (0.0, 10000.0))

        def _horizontal(regr) -> tuple[tuple[float, float], tuple[float, float]]:
            if regr.slope < 0.0:
                return ((0.0, -regr.intercept / regr.slope), (regr.intercept, 0.0))
            else:
                return ((0.0, 10000.0), (regr.intercept, regr.intercept))

        if horizontal_regr and vertical_regr:
            if vertical_regr.rvalue >= horizontal_regr.rvalue:
                modal_lines[c] = _vertical(vertical_regr)
            else:
                modal_lines[c] = _horizontal(horizontal_regr)
        elif horizontal_regr:
            modal_lines[c] = _horizontal(horizontal_regr)
        elif vertical_regr:
            modal_lines[c] = _vertical(vertical_regr)

    return modal_lines


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scale(p, v):
    """Scale a probability by a value."""
    return p * v


def threshold(p, t):
    """Threshold a probability by a value."""
    return min(p, t) + 0.05 / (1 + np.exp(-(p - t)))


def evaluate_scenario(model, situation) -> tuple:
    """Evaluate the sustainability field for *situation*.

    Returns ``(result, scenarios)`` where *result* is an
    :class:`~dt_model.simulation.evaluation.EvaluationResult` and *scenarios*
    is the list of :data:`~dt_model.simulation.evaluation.WeightedScenario`
    used (needed downstream for presence-sample generation).
    """
    scenarios: list[WeightedScenario] = list(
        OvertourismEnsemble(model, situation, cv_ensemble_size=ensemble_size)
    )
    tt = np.linspace(0, t_max, t_sample + 1)
    ee = np.linspace(0, e_max, e_sample + 1)
    result = Evaluation(model).evaluate(scenarios, axes={PV_tourists: tt, PV_excursionists: ee})
    return result, scenarios


def plot_scenario(ax, model, result, scenarios, title):
    """Render the sustainability field and KPIs onto *ax*.

    Parameters
    ----------
    ax:
        Matplotlib axes to draw on.
    model:
        The :class:`~overtourism_molveno.model.OvertourismModel` being plotted.
    result:
        :class:`~dt_model.simulation.evaluation.EvaluationResult` from
        :func:`evaluate_scenario`.
    scenarios:
        The list of weighted scenarios used in the evaluation (needed for
        presence-sample generation).
    title:
        Plot title prefix.
    """
    tt = result.axes[PV_tourists]
    ee = result.axes[PV_excursionists]

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

    rf_t = float(np.mean(result[I_P_tourists_reduction_factor]))
    sl_t = float(np.mean(result[I_P_tourists_saturation_level]))
    rf_e = float(np.mean(result[I_P_excursionists_reduction_factor]))
    sl_e = float(np.mean(result[I_P_excursionists_saturation_level]))

    sample_tourists = [
        presence_transformation(sample, rf_t, sl_t)
        for w, assignments in scenarios
        for sample in PV_tourists.sample(cvs=assignments, nr=max(1, round(w * target_presence_samples)))
    ]
    sample_excursionists = [
        presence_transformation(sample, rf_e, sl_e)
        for w, assignments in scenarios
        for sample in PV_excursionists.sample(cvs=assignments, nr=max(1, round(w * target_presence_samples)))
    ]

    axes_dict = {PV_tourists: tt, PV_excursionists: ee}

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


start_time = time.time()

fig, axs = plt.subplots(1, 3, figsize=(18, 10), layout="constrained")
result, scenarios = evaluate_scenario(M_Base, S_Base)
plot_scenario(axs[0], M_Base, result, scenarios, "Base")
result, scenarios = evaluate_scenario(M_Base, S_Good_Weather)
plot_scenario(axs[1], M_Base, result, scenarios, "Good weather")
result, scenarios = evaluate_scenario(M_Base, S_Bad_Weather)
plot_scenario(axs[2], M_Base, result, scenarios, "Bad weather")
fig.colorbar(mappable=ScalarMappable(Normalize(0, 1), cmap="coolwarm_r"), ax=axs)
fig.supxlabel("Tourists", fontsize=18)
fig.supylabel("Excursionists", fontsize=18)


print("--- %s seconds ---" % (time.time() - start_time))

plt.show()
