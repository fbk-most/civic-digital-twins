"""Molveno overtourism model definition."""

# SPDX-License-Identifier: Apache-2.0

from civic_digital_twins.dt_model import piecewise
from civic_digital_twins.dt_model.model.index import Index, LognormDistIndex, TriangDistIndex, UniformDistIndex

try:
    from .overtourism_metamodel import (
        CategoricalContextVariable,
        Constraint,
        OvertourismModel,
        PresenceVariable,
        UniformCategoricalContextVariable,
    )
    from .molveno_presence_stats import (
        excursionist_presences_stats,
        season,
        tourist_presences_stats,
        weather,
        weekday,
    )
except ImportError:
    from overtourism_metamodel import (
        CategoricalContextVariable,
        Constraint,
        OvertourismModel,
        PresenceVariable,
        UniformCategoricalContextVariable,
    )
    from molveno_presence_stats import (
        excursionist_presences_stats,
        season,
        tourist_presences_stats,
        weather,
        weekday,
    )

# Context variables

CV_weekday = UniformCategoricalContextVariable("weekday", list(weekday))
CV_season = CategoricalContextVariable("season", {v: season[v] for v in season.keys()})
CV_weather = CategoricalContextVariable("weather", {v: weather[v] for v in weather.keys()})

# Presence variables

PV_tourists = PresenceVariable("tourists", [CV_weekday, CV_season, CV_weather], tourist_presences_stats)
PV_excursionists = PresenceVariable("excursionists", [CV_weekday, CV_season, CV_weather], excursionist_presences_stats)

# Capacity indexes

I_C_parking = UniformDistIndex("parking capacity", loc=350.0, scale=100.0)
I_C_beach = UniformDistIndex("beach capacity", loc=6000.0, scale=1000.0)
I_C_accommodation = LognormDistIndex("accommodation capacity", s=0.125, loc=0.0, scale=5000.0)
I_C_food = TriangDistIndex("food service capacity", loc=3000.0, scale=1000.0, c=0.5)

# Usage indexes

I_U_tourists_parking = Index("tourist parking usage factor", 0.02)
I_U_excursionists_parking = Index(
    "excursionist parking usage factor",
    piecewise((0.55, CV_weather == "bad"), (0.80, True)),
)

I_U_tourists_beach = Index(
    "tourist beach usage factor", piecewise((0.25, CV_weather == "bad"), (0.50, True))
)
I_U_excursionists_beach = Index(
    "excursionist beach usage factor",
    piecewise((0.35, CV_weather == "bad"), (0.80, True)),
)

I_U_tourists_accommodation = Index("tourist accommodation usage factor", 0.90)

I_U_tourists_food = Index("tourist food service usage factor", 0.20)
I_U_excursionists_food = Index(
    "excursionist food service usage factor",
    piecewise((0.80, CV_weather == "bad"), (0.40, True)),
)

# Conversion indexes

I_Xa_tourists_per_vehicle = Index("tourists per vehicle allocation factor", 2.5)
I_Xa_excursionists_per_vehicle = Index("excursionists per vehicle allocation factor", 2.5)
I_Xo_tourists_parking = Index("tourists in parking rotation factor", 1.02)
I_Xo_excursionists_parking = Index("excursionists in parking rotation factor", 3.5)

I_Xo_tourists_beach = UniformDistIndex("tourists on beach rotation factor", loc=1.0, scale=2.0)
I_Xo_excursionists_beach = Index("excursionists on beach rotation factor", 1.02)

I_Xa_tourists_accommodation = Index("tourists per accommodation allocation factor", 1.05)

I_Xa_visitors_food = Index("visitors in food service allocation factor", 0.9)
I_Xo_visitors_food = Index("visitors in food service rotation factor", 2.0)

# Presence indexes

I_P_tourists_reduction_factor = Index("tourists reduction factor", 1.0)
I_P_excursionists_reduction_factor = Index("excursionists reduction factor", 1.0)

I_P_tourists_saturation_level = Index("tourists saturation level", 10000)
I_P_excursionists_saturation_level = Index("excursionists saturation level", 10000)

# Usage indexes (formula-mode Index objects wrapping the usage expressions)

I_U_parking = Index(
    "parking usage",
    PV_tourists.node * I_U_tourists_parking.node / (I_Xa_tourists_per_vehicle.node * I_Xo_tourists_parking.node)
    + PV_excursionists.node
    * I_U_excursionists_parking.node
    / (I_Xa_excursionists_per_vehicle.node * I_Xo_excursionists_parking.node),
)

I_U_beach = Index(
    "beach usage",
    PV_tourists.node * I_U_tourists_beach.node / I_Xo_tourists_beach.node
    + PV_excursionists.node * I_U_excursionists_beach.node / I_Xo_excursionists_beach.node,
)

I_U_accommodation = Index(
    "accommodation usage",
    PV_tourists.node * I_U_tourists_accommodation.node / I_Xa_tourists_accommodation.node,
)

I_U_food = Index(
    "food usage",
    (PV_tourists.node * I_U_tourists_food.node + PV_excursionists.node * I_U_excursionists_food.node)
    / (I_Xa_visitors_food.node * I_Xo_visitors_food.node),
)

# Constraints

C_parking = Constraint(name="parking", usage=I_U_parking, capacity=I_C_parking)
C_beach = Constraint(name="beach", usage=I_U_beach, capacity=I_C_beach)
C_accommodation = Constraint(name="accommodation", usage=I_U_accommodation, capacity=I_C_accommodation)
C_food = Constraint(name="food", usage=I_U_food, capacity=I_C_food)

# Model

M_Base = OvertourismModel(
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
