import math
import random
from scipy import stats
import numpy as np
from functools import reduce

from dt_model import UniformCategoricalContextVariable, ContextVariable, PresenceVariable, CategoricalContextVariable, Constraint, Ensemble, Model, Index, ConstIndex,SymIndex, DistributionIndex
#UniformDistIndex, LognormDistIndex, TriangDistIndex, 
from sympy import Symbol, Eq, Piecewise

from viz_model import VizSituation, VizApp

# MODEL DEFINITION

days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
weather = {'molto bassa': 0.65, 'bassa': 0.2, 'media': 0.075, 'alta': 0.075}

CV_weekday = UniformCategoricalContextVariable('weekday', [Symbol(v) for v in days])
CV_weather = CategoricalContextVariable('weather', {Symbol(v): weather[v] for v in weather.keys()})

# Presence variables

from presence_stats import tourist_presences_stats, excursionist_presences_stats

PV_tourists = PresenceVariable('tourists', [CV_weekday, CV_weather], tourist_presences_stats)
PV_excursionists = PresenceVariable('excursionists', [CV_weekday, CV_weather], excursionist_presences_stats)

# Capacity indexes

I_C_parking = DistributionIndex('parking capacity', distribution='uniform', loc=350.0, scale=100.0, group="Parcheggi")
I_C_beach = DistributionIndex('beach capacity', distribution='uniform', loc=6000.0, scale=1000.0, group="Spiaggia",)
I_C_accommodation = DistributionIndex('accommodation capacity', distribution='lognorm', loc=0.0, scale=5000.0, s=0.125, group="Alberghi")
I_C_food = DistributionIndex('food service capacity', distribution='triang', loc=3000.0, scale=1000.0, c=0.5, group="Ristoranti")

# Usage indexes

I_U_tourists_parking = ConstIndex('tourist parking usage factor', 0.02, group="Parcheggi")
I_U_excursionists_parking = SymIndex('excursionist parking usage factor',
                                  Piecewise((0.55, Eq(CV_weather, Symbol('alta'))| Eq(CV_weather, Symbol('media'))), (0.80, True)),
                                  cvs=[CV_weather], group="Parcheggi")

I_U_tourists_beach = SymIndex('tourist beach usage factor',
                           Piecewise((0.15, Eq(CV_weather, Symbol('alta')) | Eq(CV_weather, Symbol('media'))),
                                     (0.50, True)),
                           cvs=[CV_weather], group="Spiaggia")
I_U_excursionists_beach = SymIndex('excursionist beach usage factor',
                                Piecewise((0.35, Eq(CV_weather, Symbol('alta')) | Eq(CV_weather, Symbol('media'))),
                                          (0.80, True)),
                                cvs=[CV_weather], group="Spiaggia")

I_U_tourists_accommodation = ConstIndex('tourist accommodation usage factor', 0.90, group="Alberghi")

I_U_tourists_food = ConstIndex('tourist food service usage factor', 0.20, group="Ristoranti")
I_U_excursionists_food = SymIndex('excursionist food service usage factor',
                               Piecewise((0.80, Eq(CV_weather, Symbol('alta')) | Eq(CV_weather, Symbol('media'))),
                                         (0.40, True)),
                               cvs=[CV_weather], group="Ristoranti")

# Conversion indexes

I_Xa_tourists_per_vehicle = ConstIndex('tourists per vehicle allocation factor', 2.5, group="Parcheggi")
I_Xa_excursionists_per_vehicle = ConstIndex('excursionists per vehicle allocation factor', 2.5, group="Parcheggi")
I_Xo_tourists_parking  = ConstIndex('tourists in parking rotation factor', 1.02, group="Parcheggi")
I_Xo_excursionists_parking = ConstIndex('excursionists in parking rotation factor', 3.5, group="Parcheggi")

I_Xo_tourists_beach = DistributionIndex('tourists on beach rotation factor', distribution='uniform', loc=1.0, scale=2.0, group="Spiaggia")
I_Xo_excursionists_beach = ConstIndex('excursionists on beach rotation factor', 1.02, group="Spiaggia")

I_Xa_tourists_accommodation = ConstIndex('tourists per accommodation allocation factor', 1.05, group="Alberghi")

I_Xa_visitors_food = ConstIndex('visitors in food service allocation factor', 0.9, group="Ristoranti")
I_Xo_visitors_food = ConstIndex('visitors in food service rotation factor', 2.0, group="Ristoranti")

# Constraints
# TODO: add names to constraints?
C_parking = Constraint(usage=PV_tourists * I_U_tourists_parking / (I_Xa_tourists_per_vehicle * I_Xo_tourists_parking) +
                             PV_excursionists * I_U_excursionists_parking / (
                                         I_Xa_excursionists_per_vehicle * I_Xo_excursionists_parking),
                       capacity=I_C_parking, group="Parcheggi")

C_beach = Constraint(usage=PV_tourists * I_U_tourists_beach / I_Xo_tourists_beach +
                             PV_excursionists * I_U_excursionists_beach / I_Xo_excursionists_beach,
                    capacity=I_C_beach, group="Spiaggia")

# TODO: also capacity should be a formula
# C_accommodation = Constraint(usage=PV_tourists * I_U_tourists_accommodation,
#                              capacity=I_C_accommodation *  I_Xa_tourists_accommodation)

C_accommodation = Constraint(usage=PV_tourists * I_U_tourists_accommodation / I_Xa_tourists_accommodation,
                             capacity=I_C_accommodation, group="Alberghi")

# TODO: also capacity should be a formula
# C_food = Constraint(usage=PV_tourists * I_U_tourists_food +
#                              PV_excursionists * I_U_excursionists_food,
#                     capacity=I_C_food * I_Xa_visitors_food * I_Xo_visitors_food)
C_food = Constraint(usage=(PV_tourists * I_U_tourists_food + PV_excursionists * I_U_excursionists_food) /
                          (I_Xa_visitors_food * I_Xo_visitors_food),
                    capacity=I_C_food, group="Ristoranti")

# Base model
M_Base = Model('base model', [CV_weekday, CV_weather], [PV_tourists, PV_excursionists],
               [I_U_tourists_parking, I_U_excursionists_parking, I_U_tourists_beach, I_U_excursionists_beach,
                I_U_tourists_accommodation, I_U_tourists_food, I_U_excursionists_food,
                I_Xa_tourists_per_vehicle, I_Xa_excursionists_per_vehicle, I_Xa_tourists_accommodation,
                I_Xo_tourists_parking, I_Xo_excursionists_parking, I_Xo_tourists_beach, I_Xo_excursionists_beach,
                I_Xa_visitors_food, I_Xo_visitors_food],
               [I_C_parking, I_C_beach, I_C_accommodation, I_C_food],
               [C_parking, C_beach, C_accommodation, C_food])


# Base situation
S_Base = {}

# Good weather situation
S_Good_Weather = {CV_weather: [Symbol('molto bassa'), Symbol('bassa')]}

# Bad weather situation
S_Bad_Weather = {CV_weather: [Symbol('alta')]}

(x_max, y_max) = (10000, 10000)

def app():
    return VizApp(M_Base, 
                  {"Base": S_Base, "Good Weather": S_Good_Weather, "Bad Weather": S_Bad_Weather},
                  PV_tourists, x_max,
                  PV_excursionists, y_max)

