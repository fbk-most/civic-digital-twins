import math
import random
from scipy import stats
import numpy as np

from dt_model import ContextVariable, PresenceVariable, Index, Constraint, Ensemble, Model
from sympy import Symbol, Eq, Piecewise

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# MODEL DEFINITION

# Context variables
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
weather = {'molto bassa': 0.65, 'bassa': 0.2, 'media': 0.075, 'alta': 0.075}

CV_weekday = ContextVariable('weekday', [Symbol(v) for v in days])
CV_weather = ContextVariable('weather', [Symbol(v) for v in weather.keys()],
                             {Symbol(v): weather[v] for v in weather.keys()})

# Presence variables

from presence_stats import tourist_presences_stats, excursionist_presences_stats

PV_tourists = PresenceVariable('tourists', [CV_weekday, CV_weather], tourist_presences_stats)
PV_excursionists = PresenceVariable('excursionists', [CV_weekday, CV_weather], excursionist_presences_stats)

# Capacity indexes

I_C_parking = Index('parking capacity', stats.uniform(loc=350.0, scale=100.0))
I_C_beach = Index('beach capacity', stats.uniform(loc=6000.0, scale=1000.0))
I_C_accommodation = Index('accommodation capacity', stats.lognorm(s=0.125, loc=0.0, scale=5000.0))
I_C_food = Index('food service capacity', stats.triang(loc=3000.0, scale=1000.0, c=0.5))

# Usage indexes

I_U_tourists_parking = Index('tourist parking usage factor', 0.02)
I_U_excursionists_parking = Index('excursionist parking usage factor',
                                  Piecewise((0.55, Eq(CV_weather, Symbol('alta'))| Eq(CV_weather, Symbol('media'))), (0.80, True)),
                                  cvs=[CV_weather])

I_U_tourists_beach = Index('tourist beach usage factor',
                           Piecewise((0.15, Eq(CV_weather, Symbol('alta')) | Eq(CV_weather, Symbol('media'))),
                                     (0.50, True)),
                           cvs=[CV_weather])
I_U_excursionists_beach = Index('excursionist beach usage factor',
                                Piecewise((0.35, Eq(CV_weather, Symbol('alta')) | Eq(CV_weather, Symbol('media'))),
                                          (0.80, True)),
                                cvs=[CV_weather])

I_U_tourists_accommodation = Index('tourist accommodation usage factor', 0.90)

I_U_tourists_food = Index('tourist food service usage factor', 0.20)
I_U_excursionists_food = Index('excursionist food service usage factor',
                               Piecewise((0.80, Eq(CV_weather, Symbol('alta')) | Eq(CV_weather, Symbol('media'))),
                                         (0.40, True)),
                               cvs=[CV_weather])

# Conversion indexes

I_Xa_tourists_per_vehicle = Index('tourists per vehicle allocation factor', 2.5)
I_Xa_excursionists_per_vehicle = Index('excursionists per vehicle allocation factor', 2.5)
I_Xo_tourists_parking  = Index('tourists in parking rotation factor', 1.02)
I_Xo_excursionists_parking = Index('excursionists in parking rotation factor', 3.5)

I_Xo_tourists_beach = Index('tourists on beach rotation factor', stats.uniform(loc=1.0, scale=2.0))
I_Xo_excursionists_beach = Index('excursionists on beach rotation factor', 1.02)

I_Xa_tourists_accommodation = Index('tourists per accommodation allocation factor', 1.05)

I_Xa_visitors_food = Index('visitors in food service allocation factor', 0.9)
I_Xo_visitors_food = Index('visitors in food service rotation factor', 2.0)

# Constraints
# TODO: add names to constraints?

C_parking = Constraint(usage=PV_tourists * I_U_tourists_parking / (I_Xa_tourists_per_vehicle * I_Xo_tourists_parking) +
                             PV_excursionists * I_U_excursionists_parking / (
                                         I_Xa_excursionists_per_vehicle * I_Xo_excursionists_parking),
                       capacity=I_C_parking)

C_beach = Constraint(usage=PV_tourists * I_U_tourists_beach / I_Xo_tourists_beach +
                             PV_excursionists * I_U_excursionists_beach / I_Xo_excursionists_beach,
                    capacity=I_C_beach)

# TODO: also capacity should be a formula
# C_accommodation = Constraint(usage=PV_tourists * I_U_tourists_accommodation,
#                              capacity=I_C_accommodation *  I_Xa_tourists_accommodation)

C_accommodation = Constraint(usage=PV_tourists * I_U_tourists_accommodation / I_Xa_tourists_accommodation,
                             capacity=I_C_accommodation)

# TODO: also capacity should be a formula
# C_food = Constraint(usage=PV_tourists * I_U_tourists_food +
#                              PV_excursionists * I_U_excursionists_food,
#                     capacity=I_C_food * I_Xa_visitors_food * I_Xo_visitors_food)
C_food = Constraint(usage=(PV_tourists * I_U_tourists_food + PV_excursionists * I_U_excursionists_food) /
                          (I_Xa_visitors_food * I_Xo_visitors_food),
                    capacity=I_C_food)

# Models
# TODO: what is the better process to create a model? (e.g., adding elements incrementally)

# Base model
M_Base = Model('base model', [CV_weekday, CV_weather], [PV_tourists, PV_excursionists],
               [I_U_tourists_parking, I_U_excursionists_parking, I_U_tourists_beach, I_U_excursionists_beach,
                I_U_tourists_accommodation, I_U_tourists_food, I_U_excursionists_food,
                I_Xa_tourists_per_vehicle, I_Xa_excursionists_per_vehicle, I_Xa_tourists_accommodation,
                I_Xo_tourists_parking, I_Xo_excursionists_parking, I_Xo_tourists_beach, I_Xo_excursionists_beach,
                I_Xa_visitors_food, I_Xo_visitors_food],
               [I_C_parking, I_C_beach, I_C_accommodation, I_C_food],
               [C_parking, C_beach, C_accommodation, C_food])

# Larger park capacity model
I_C_parking_larger = Index('larger parking capacity', stats.uniform(loc=550.0, scale=100.0))

M_MoreParking = M_Base.variation('larger parking model', change_capacities={I_C_parking: I_C_parking_larger})

# ANALYSIS SITUATIONS

# Base situation
S_Base = {}

# Good weather situation
S_Good_Weather = {CV_weather: [Symbol('molto bassa'), Symbol('bassa')]}

# Bad weather situation
S_Bad_Weather = {CV_weather: [Symbol('alta')]}

# PLOTTING

(x_max, y_max) = (10000, 10000)
(x_sample, y_sample) = (100, 100)
target_presence_samples = 200
ensemble_size = 20  # TODO: make configurable; may it be a CV parameter?


def plot_scenario(ax, model, situation, title):
    ensemble = Ensemble(model, situation, cv_ensemble_size=ensemble_size)
    xx = np.linspace(0, x_max, x_sample + 1)
    yy = np.linspace(0, y_max, y_sample + 1)
    xx, yy = np.meshgrid(xx, yy)
    zz = model.evaluate({PV_tourists: xx, PV_excursionists: yy}, ensemble)
    #zz = sum([model.evaluate_single_case({PV_tourists: xx, PV_excursionists: yy}, case) for case in ensemble])/ensemble.size()

    case_number = ensemble.size
    samples_per_case = math.ceil(target_presence_samples/case_number)

    sample_tourists = [sample for case in ensemble for sample in PV_tourists.sample(cvs=case, nr=samples_per_case)]
    sample_excursionists = [sample for case in ensemble for sample in PV_excursionists.sample(cvs=case, nr=samples_per_case)]

    if case_number*samples_per_case > target_presence_samples:
        sample_tourists = random.sample(sample_tourists, target_presence_samples)
        sample_excursionists = random.sample(sample_excursionists, target_presence_samples)

    # TODO: move elsewhere, it cannot be computed this way...
    area = zz.sum() * (x_max / x_sample / 1000) * (y_max / y_sample / 1000)

    # TODO: re-enable median
    # C_accommodation.median(cvs=scenario).plot(ax, color='red')
    # C_parking.median(cvs=scenario).plot(ax, color='red')
    ax.contourf(xx, yy, zz, levels=100, cmap='coolwarm_r')
    ax.scatter(sample_excursionists, sample_tourists, color='gainsboro', edgecolors='black')
    ax.set_title(f'{title}\nArea = {area:.2f} kp$^2$', fontsize=12)
    ax.set_xlim(left=0, right=x_max)
    ax.set_ylim(bottom=0, top=y_max)

import time
start_time = time.time()

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
fig.subplots_adjust(hspace=0.3)
plot_scenario(axs[0, 0], M_Base, S_Base, 'Base')
plot_scenario(axs[0, 1], M_Base, S_Good_Weather, 'Good weather')
plot_scenario(axs[0, 2], M_Base, S_Bad_Weather, 'Bad weather')
plot_scenario(axs[1, 0], M_MoreParking, S_Base, 'More parking ')
plot_scenario(axs[1, 1], M_MoreParking, S_Good_Weather, 'More parking - Good weather')
plot_scenario(axs[1, 2], M_MoreParking, S_Bad_Weather, 'More parking - Bad weather')
fig.colorbar(mappable=ScalarMappable(Normalize(0, 1), cmap='coolwarm_r'), ax=axs)
fig.show()

print("--- %s seconds ---" % (time.time() - start_time))
