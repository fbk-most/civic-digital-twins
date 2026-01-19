"""Example to interactively explore the frozen Molveno model.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import json
import requests
import datetime

from civic_digital_twins.dt_model import Ensemble, Evaluation, InstantiatedModel
from civic_digital_twins.dt_model.internal.sympyke import Symbol
from civic_digital_twins.dt_model.reference_models.molveno.overtourism import (
    CV_weather,
    I_P_excursionists_reduction_factor,
    I_P_excursionists_saturation_level,
    I_P_tourists_reduction_factor,
    I_P_tourists_saturation_level,
    M_Base,
    M_Parking,
    PV_excursionists,
    PV_tourists,
)

from civic_digital_twins.dt_model.symbols.index import UniformDistIndex

def scale(p, v):
    """Scale a probability by a value."""
    return p * v


def threshold(p, t):
    """Threshold a probability by a value."""
    return min(p, t) + 0.05 / (1 + np.exp(-(p - t)))


def compute_scenario(model, situation):

    (t_max, e_max) = (10000, 10000)
    (t_sample, e_sample) = (100, 100)
    target_presence_samples = 200
    ensemble_size = 20  # TODO: make configurable; may it be a CV parameter?

    """Compute a scenario."""
    ensemble = Ensemble(model, situation, cv_ensemble_size=ensemble_size)
    tt = np.linspace(0, t_max, t_sample + 1)
    ee = np.linspace(0, e_max, e_sample + 1)
    evaluation = Evaluation(model, ensemble)
    evaluation.evaluate_grid({PV_tourists: tt, PV_excursionists: ee})

    sample_tourists = [
        sample
        for c in ensemble
        for sample in PV_tourists.sample(
            cvs=c[1], nr=max(1, round(c[0] * target_presence_samples))
        )
    ]
    sample_excursionists = [
        sample
        for c in ensemble
        for sample in PV_excursionists.sample(
            cvs=c[1], nr=max(1, round(c[0] * target_presence_samples))
        )
    ]

    # Presence Transformation function
    # TODO: manage differently!
    def presence_transformation(
        presence, reduction_factor, saturation_level, sharpness=3
    ):
        tmp = presence * reduction_factor
        return (
            tmp
            * saturation_level
            / ((tmp**sharpness + saturation_level**sharpness) ** (1 / sharpness))
        )

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
    (i, c) = evaluation.compute_sustainability_index_with_ci(
        list(zip(sample_tourists, sample_excursionists)), confidence=0.8
    )
    sust_indexes = evaluation.compute_sustainability_index_with_ci_per_constraint(
        list(zip(sample_tourists, sample_excursionists)), confidence=0.8
    )
    critical = min(sust_indexes, key=lambda i: sust_indexes.get(i)[0])
    modals = evaluation.compute_modal_line_per_constraint()

    return evaluation, sample_tourists, sample_excursionists


def constraint_name(c):
    return getattr(c, "name", None) or str(c)


def summarize_simulation(evaluation, sample_tourists, sample_excursionists):
    summary = {}

    # Initial conditions
    summary["ensemble_size"] = 20
    summary["tourist_samples"] = sample_tourists[:10]  # show first 10 for brevity
    summary["excursionist_samples"] = sample_excursionists[:10]
    summary["grid_shape"] = evaluation.grid[list(evaluation.grid.keys())[0]].shape
    summary["sustainable_area"] = evaluation.compute_sustainable_area()

    # Overall sustainability index
    si_mean, si_ci = evaluation.compute_sustainability_index_with_ci(
        list(zip(sample_tourists, sample_excursionists)), confidence=0.8
    )
    summary["sustainability_index_mean"] = si_mean
    summary["sustainability_index_ci"] = si_ci

    # Per-constraint sustainability
    per_constraint = evaluation.compute_sustainability_index_with_ci_per_constraint(
        list(zip(sample_tourists, sample_excursionists)), confidence=0.8
    )

    summary["per_constraint_index"] = {
        constraint_name(c): {"mean": v[0], "ci": v[1]}
        for c, v in per_constraint.items()
    }

    # # Critical constraint
    critical = min(per_constraint, key=lambda c: per_constraint[c][0])
    summary["critical_constraint"] = constraint_name(critical)

    # Modal lines
    #modal_lines = evaluation.compute_modal_line_per_constraint()
    #summary["modal_lines"] = {str(c): ml for c, ml in modal_lines.items()}

    return summary

def get_scenarios():
    model1 = InstantiatedModel(M_Base)
    model2 = InstantiatedModel(M_Parking)

    S_Good_Weather = {CV_weather: [Symbol("good"), Symbol("unsettled")]}

    evaluation1, sample_tourists1, sample_excursionists1 = compute_scenario(
        model1, S_Good_Weather
    )

    evaluation2, sample_tourists2, sample_excursionists2 = compute_scenario(
        model2, S_Good_Weather
    )

    summary1 = summarize_simulation(evaluation1, sample_tourists1, sample_excursionists1)
    summary2 = summarize_simulation(evaluation2, sample_tourists2, sample_excursionists2)

    return summary1, summary2

summary1, summary2 = get_scenarios()

# QUERY SINGLE SIMULATION
URL = "http://127.0.0.1:9000/agent"

state = {
    "user_message": "What is the most critical index?",
    "sim1_id": "sim1",
    "sim2_id": None,
    "simulation_data": {
        "sim1": summary1
    }, 
    "simulation_data_loaded_at": datetime.datetime.now().isoformat(),
    "pending_inputs": {},
    "missing_inputs": [],
    "awaiting_confirmation": False,
    "running_simulation": False,
    "next_action": None,
    "response": "",
}

# print("\n=== SENDING STATE TO AGENT ===")
# print(json.dumps(state, indent=2))

response = requests.post(
    URL,
    json=state,
    timeout=2000
)

response.raise_for_status()

result = response.json()

print("\n=== AGENT RESPONSE SINGLE SIMULATION ===")
print(result["response"])


state = {
    "user_message": "What is the main difference between the two scenarios?",
    "sim1_id": "sim1",
    "sim2_id": "sim2",
    "simulation_data": {
        "sim1": summary1,
        "sim2": summary2
    }, 
    "simulation_data_loaded_at": datetime.datetime.now().isoformat(),
    "pending_inputs": {},
    "missing_inputs": [],
    "awaiting_confirmation": False,
    "running_simulation": False,
    "next_action": None,
    "response": "",
}

# print("\n=== SENDING STATE TO AGENT ===")
# print(json.dumps(state, indent=2))

response = requests.post(
    URL,
    json=state,
    timeout=2000
)

response.raise_for_status()

result = response.json()

print("\n=== AGENT RESPONSE SINGLE SIMULATION ===")
print(result["response"])