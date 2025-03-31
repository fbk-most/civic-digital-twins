"""This example allows to interactively explore the frozen Molveno model
included into the source tree as an illustrative example."""

# SPDX-License-Identifier: Apache-2.0

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from dt_model import Ensemble
from dt_model.examples.molveno.overtourism import (
    CV_weather,
    I_P_excursionists_reduction_factor,
    I_P_excursionists_saturation_level,
    I_P_tourists_reduction_factor,
    I_P_tourists_saturation_level,
    M_Base,
    #M_MoreParking,
    PV_excursionists,
    PV_tourists,
)
from dt_model.internal.sympyke import Symbol

# Base situation
S_Base = {}

# Good weather situation
S_Good_Weather = {CV_weather: [Symbol("good")]}

# Bad weather situation
S_Bad_Weather = {CV_weather: [Symbol("bad")]}

# PLOTTING

(t_max, e_max) = (10000, 10000)
(t_sample, e_sample) = (100, 100)
target_presence_samples = 200
ensemble_size = 20  # TODO: make configurable; may it be a CV parameter?


def scale(p, v):
    return p * v


def threshold(p, t):
    return min(p, t) + 0.05 / (1 + np.exp(-(p - t)))


def plot_scenario(ax, model, situation, title):
    ensemble = Ensemble(model, situation, cv_ensemble_size=ensemble_size)
    tt = np.linspace(0, t_max, t_sample + 1)
    ee = np.linspace(0, e_max, e_sample + 1)
    xx, yy = np.meshgrid(tt, ee)
    zz = model.evaluate({PV_tourists: tt, PV_excursionists: ee}, ensemble)

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
            model.get_index_mean_value(I_P_tourists_reduction_factor),
            model.get_index_mean_value(I_P_tourists_saturation_level),
        )
        for presence in sample_tourists
    ]
    sample_excursionists = [
        presence_transformation(
            presence,
            model.get_index_mean_value(I_P_excursionists_reduction_factor),
            model.get_index_mean_value(I_P_excursionists_saturation_level),
        )
        for presence in sample_excursionists
    ]

    # TODO: move elsewhere, it cannot be computed this way...
    area = model.compute_sustainable_area()
    index = model.compute_sustainability_index(list(zip(sample_tourists, sample_excursionists)))
    indexes = model.compute_sustainability_index_per_constraint(list(zip(sample_tourists, sample_excursionists)))
    critical = min(indexes, key=indexes.get)
    modals = model.compute_modal_line_per_constraint()

    ax.pcolormesh(xx, yy, zz, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    for modal in modals.values():
        ax.plot(*modal, color="black", linewidth=2)
    ax.scatter(sample_excursionists, sample_tourists, color="gainsboro", edgecolors="black")
    ax.set_title(
        f"{title}\n"
        + f"area = {area / 10e6:.2f} kp$^2$ - "
        + f"Sustainability = {index * 100:.2f}%\n"
        + f"Critical = {critical.capacity.name} ({indexes[critical] * 100:.2f}%)",
        fontsize=12,
    )
    ax.set_xlim(left=0, right=t_max)
    ax.set_ylim(bottom=0, top=e_max)

    model.reset()


start_time = time.time()

fig, axs = plt.subplots(2, 3, figsize=(18, 10), layout="constrained")
plot_scenario(axs[0, 0], M_Base, S_Base, "Base")
plot_scenario(axs[0, 1], M_Base, S_Good_Weather, "Good weather")
plot_scenario(axs[0, 2], M_Base, S_Bad_Weather, "Bad weather")
#plot_scenario(axs[1, 0], M_MoreParking, S_Base, "More parking ")
#plot_scenario(axs[1, 1], M_MoreParking, S_Good_Weather, "More parking - Good weather")
#plot_scenario(axs[1, 2], M_MoreParking, S_Bad_Weather, "More parking - Bad weather")
fig.colorbar(mappable=ScalarMappable(Normalize(0, 1), cmap="coolwarm_r"), ax=axs)
fig.supxlabel("Tourists", fontsize=18)
fig.supylabel("Excursionists", fontsize=18)

print("--- %s seconds ---" % (time.time() - start_time))

plt.show()
