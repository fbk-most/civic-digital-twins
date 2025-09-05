"""Example to interactively explore the frozen Molveno model.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from civic_digital_twins.dt_model import Ensemble, Evaluation, InstantiatedModel
from civic_digital_twins.dt_model.internal.sympyke import Symbol
from civic_digital_twins.dt_model.reference_models.molveno.overtourism import (
    CV_weather,
    I_P_excursionists_reduction_factor,
    I_P_excursionists_saturation_level,
    I_P_tourists_reduction_factor,
    I_P_tourists_saturation_level,
    M_Base,
    PV_excursionists,
    PV_tourists,
)

# Instantiate the model
IM_Base = InstantiatedModel(M_Base)

# Base situation
S_Base = {}

# Good weather situation
S_Good_Weather = {CV_weather: [Symbol("good"), Symbol("unsettled")]}

# Bad weather situation
S_Bad_Weather = {CV_weather: [Symbol("bad")]}

# PLOTTING

(t_max, e_max) = (10000, 10000)
(t_sample, e_sample) = (100, 100)
target_presence_samples = 200
ensemble_size = 20  # TODO: make configurable; may it be a CV parameter?


def scale(p, v):
    """Scale a probability by a value."""
    return p * v


def threshold(p, t):
    """Threshold a probability by a value."""
    return min(p, t) + 0.05 / (1 + np.exp(-(p - t)))


def plot_scenario(ax, model, situation, title):
    """Plot a scenario."""
    ensemble = Ensemble(model, situation, cv_ensemble_size=ensemble_size)
    tt = np.linspace(0, t_max, t_sample + 1)
    ee = np.linspace(0, e_max, e_sample + 1)
    xx, yy = np.meshgrid(tt, ee)
    evaluation = Evaluation(model, ensemble)
    zz = evaluation.evaluate_grid({PV_tourists: tt, PV_excursionists: ee})

    sample_tourists = [
        sample
        for c in ensemble
        for sample in PV_tourists.sample(cvs=c[1], nr=max(1, round(c[0] * target_presence_samples)))
    ]
    sample_excursionists = [
        sample
        for c in ensemble
        for sample in PV_excursionists.sample(cvs=c[1], nr=max(1, round(c[0] * target_presence_samples)))
    ]

    # Presence Transformation function
    # TODO: manage differently!
    def presence_transformation(presence, reduction_factor, saturation_level, sharpness=3):
        tmp = presence * reduction_factor
        return tmp * saturation_level / ((tmp**sharpness + saturation_level**sharpness) ** (1 / sharpness))

    sample_tourists = [
        presence_transformation(
            presence,
            evaluation.get_index_mean_value(I_P_tourists_reduction_factor),
            evaluation.get_index_mean_value(I_P_tourists_saturation_level),
        )
        for presence in sample_tourists
    ]
    sample_excursionists = [
        presence_transformation(
            presence,
            evaluation.get_index_mean_value(I_P_excursionists_reduction_factor),
            evaluation.get_index_mean_value(I_P_excursionists_saturation_level),
        )
        for presence in sample_excursionists
    ]

    # TODO: move elsewhere, it cannot be computed this way...
    area = evaluation.compute_sustainable_area()
    (i,c) = evaluation.compute_sustainability_index_with_ci(list(zip(sample_tourists, sample_excursionists)),
                                                            confidence=0.8)
    sust_indexes = evaluation.compute_sustainability_index_with_ci_per_constraint(
        list(zip(sample_tourists, sample_excursionists)), confidence=0.8)
    critical = min(sust_indexes, key=lambda i: sust_indexes.get(i)[0])
    modals = evaluation.compute_modal_line_per_constraint()

    ax.pcolormesh(xx, yy, zz, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    for modal in modals.values():
        ax.plot(*modal, color="black", linewidth=2)
    ax.scatter(sample_excursionists, sample_tourists, color="gainsboro", edgecolors="black")
    ax.set_title(
        f"{title}\n"
        + f"area = {area / 10e6:.2f} kp$^2$ - "
        + f"Sustainability = {i * 100:.2f}% +/- {c * 100:.2f}%\n"
        + f"Critical = {critical.capacity.name}"
        + f"({sust_indexes[critical][0] * 100:.2f}% +/- {sust_indexes[critical][1] * 100:.2f}%)",
        fontsize=12,
    )
    ax.set_xlim(left=0, right=t_max)
    ax.set_ylim(bottom=0, top=e_max)


start_time = time.time()

fig, axs = plt.subplots(1, 3, figsize=(18, 10), layout="constrained")
plot_scenario(axs[0], IM_Base, S_Base, "Base")
plot_scenario(axs[1], IM_Base, S_Good_Weather, "Good weather")
plot_scenario(axs[2], IM_Base, S_Bad_Weather, "Bad weather")
fig.colorbar(mappable=ScalarMappable(Normalize(0, 1), cmap="coolwarm_r"), ax=axs)
fig.supxlabel("Tourists", fontsize=18)
fig.supylabel("Excursionists", fontsize=18)



print("--- %s seconds ---" % (time.time() - start_time))

plt.show()
