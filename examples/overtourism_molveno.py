import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from presence_stats import excursionist_presences_stats, season, tourist_presences_stats, weather, weekday
from scipy import stats
from sympy import Eq, Piecewise, Symbol

from dt_model import (
    CategoricalContextVariable,
    Constraint,
    Ensemble,
    Index,
    Model,
    PresenceVariable,
    UniformCategoricalContextVariable,
)

# MODEL DEFINITION

# Context variables

CV_weekday = UniformCategoricalContextVariable("weekday", [Symbol(v) for v in weekday])
CV_season = CategoricalContextVariable("season", {Symbol(v): season[v] for v in season.keys()})
CV_weather = CategoricalContextVariable("weather", {Symbol(v): weather[v] for v in weather.keys()})

# Presence variables

PV_tourists = PresenceVariable("tourists", [CV_weekday, CV_season, CV_weather], tourist_presences_stats)
PV_excursionists = PresenceVariable("excursionists", [CV_weekday, CV_season, CV_weather], excursionist_presences_stats)

# Capacity indexes

I_C_parking = Index("parking capacity", stats.uniform(loc=350.0, scale=100.0))
I_C_beach = Index("beach capacity", stats.uniform(loc=6000.0, scale=1000.0))
I_C_accommodation = Index("accommodation capacity", stats.lognorm(s=0.125, loc=0.0, scale=5000.0))
I_C_food = Index("food service capacity", stats.triang(loc=3000.0, scale=1000.0, c=0.5))

# Usage indexes

I_U_tourists_parking = Index("tourist parking usage factor", 0.02)
I_U_excursionists_parking = Index(
    "excursionist parking usage factor",
    Piecewise((0.55, Eq(CV_weather, Symbol("bad"))), (0.80, True)),
    cvs=[CV_weather],
)

I_U_tourists_beach = Index(
    "tourist beach usage factor", Piecewise((0.25, Eq(CV_weather, Symbol("bad"))), (0.50, True)), cvs=[CV_weather]
)
I_U_excursionists_beach = Index(
    "excursionist beach usage factor", Piecewise((0.35, Eq(CV_weather, Symbol("bad"))), (0.80, True)), cvs=[CV_weather]
)

I_U_tourists_accommodation = Index("tourist accommodation usage factor", 0.90)

I_U_tourists_food = Index("tourist food service usage factor", 0.20)
I_U_excursionists_food = Index(
    "excursionist food service usage factor",
    Piecewise((0.80, Eq(CV_weather, Symbol("bad"))), (0.40, True)),
    cvs=[CV_weather, CV_weekday],
)

# Conversion indexes

I_Xa_tourists_per_vehicle = Index("tourists per vehicle allocation factor", 2.5)
I_Xa_excursionists_per_vehicle = Index("excursionists per vehicle allocation factor", 2.5)
I_Xo_tourists_parking = Index("tourists in parking rotation factor", 1.02)
I_Xo_excursionists_parking = Index("excursionists in parking rotation factor", 3.5)

I_Xo_tourists_beach = Index("tourists on beach rotation factor", stats.uniform(loc=1.0, scale=2.0))
I_Xo_excursionists_beach = Index("excursionists on beach rotation factor", 1.02)

I_Xa_tourists_accommodation = Index("tourists per accommodation allocation factor", 1.05)

I_Xa_visitors_food = Index("visitors in food service allocation factor", 0.9)
I_Xo_visitors_food = Index("visitors in food service rotation factor", 2.0)

# Presence indexes

I_P_tourists_reduction_factor = Index("tourists reduction factor", 1.0)
I_P_excursionists_reduction_factor = Index("excursionists reduction factor", 1.0)

I_P_tourists_saturation_level = Index("tourists saturation level", 10000)
I_P_excursionists_saturation_level = Index("excursionists saturation level", 10000)


# Constraints
# TODO: add names to constraints?

C_parking = Constraint(
    usage=PV_tourists * I_U_tourists_parking / (I_Xa_tourists_per_vehicle * I_Xo_tourists_parking)
    + PV_excursionists * I_U_excursionists_parking / (I_Xa_excursionists_per_vehicle * I_Xo_excursionists_parking),
    capacity=I_C_parking,
)

C_beach = Constraint(
    usage=PV_tourists * I_U_tourists_beach / I_Xo_tourists_beach
    + PV_excursionists * I_U_excursionists_beach / I_Xo_excursionists_beach,
    capacity=I_C_beach,
)

# TODO: also capacity should be a formula
# C_accommodation = Constraint(usage=PV_tourists * I_U_tourists_accommodation,
#                              capacity=I_C_accommodation *  I_Xa_tourists_accommodation)

C_accommodation = Constraint(
    usage=PV_tourists * I_U_tourists_accommodation / I_Xa_tourists_accommodation, capacity=I_C_accommodation
)

# TODO: also capacity should be a formula
# C_food = Constraint(usage=PV_tourists * I_U_tourists_food +
#                              PV_excursionists * I_U_excursionists_food,
#                     capacity=I_C_food * I_Xa_visitors_food * I_Xo_visitors_food)
C_food = Constraint(
    usage=(PV_tourists * I_U_tourists_food + PV_excursionists * I_U_excursionists_food)
    / (I_Xa_visitors_food * I_Xo_visitors_food),
    capacity=I_C_food,
)


# Models
# TODO: what is the better process to create a model? (e.g., adding elements incrementally)

# Base model
M_Base = Model(
    "base model",
    [CV_weekday, CV_season, CV_weather],
    [PV_tourists, PV_excursionists],
    [
        I_U_tourists_parking,
        I_U_excursionists_parking,
        I_U_tourists_beach,
        I_U_excursionists_beach,
        I_U_tourists_accommodation,
        I_U_tourists_food,
        I_U_excursionists_food,
        I_Xa_tourists_per_vehicle,
        I_Xa_excursionists_per_vehicle,
        I_Xa_tourists_accommodation,
        I_Xo_tourists_parking,
        I_Xo_excursionists_parking,
        I_Xo_tourists_beach,
        I_Xo_excursionists_beach,
        I_Xa_visitors_food,
        I_Xo_visitors_food,
        I_P_tourists_reduction_factor,
        I_P_excursionists_reduction_factor,
        I_P_tourists_saturation_level,
        I_P_excursionists_saturation_level,
    ],
    [I_C_parking, I_C_beach, I_C_accommodation, I_C_food],
    [C_parking, C_beach, C_accommodation, C_food],
)

# Larger park capacity model
I_C_parking_larger = Index("larger parking capacity", stats.uniform(loc=550.0, scale=100.0))

M_MoreParking = M_Base.variation("larger parking model", change_capacities={I_C_parking: I_C_parking_larger})

# ANALYSIS SITUATIONS

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
plot_scenario(axs[1, 0], M_MoreParking, S_Base, "More parking ")
plot_scenario(axs[1, 1], M_MoreParking, S_Good_Weather, "More parking - Good weather")
plot_scenario(axs[1, 2], M_MoreParking, S_Bad_Weather, "More parking - Bad weather")
fig.colorbar(mappable=ScalarMappable(Normalize(0, 1), cmap="coolwarm_r"), ax=axs)
fig.supxlabel("Tourists", fontsize=18)
fig.supylabel("Excursionists", fontsize=18)
fig.show()

print("--- %s seconds ---" % (time.time() - start_time))

plt.show()
