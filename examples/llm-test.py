"""Example to interactively explore the frozen Molveno model.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np
import json

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

# Good weather situation
S_Good_Weather = {CV_weather: [Symbol("good"), Symbol("unsettled")]}

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


def compute_scenario(model, situation):
    """Plot a scenario."""
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
    modal_lines = evaluation.compute_modal_line_per_constraint()
    summary["modal_lines"] = {str(c): ml for c, ml in modal_lines.items()}

    return summary


evaluation, sample_tourists, sample_excursionists = compute_scenario(
    IM_Base, S_Good_Weather
)

summary = summarize_simulation(evaluation, sample_tourists, sample_excursionists)


prompt = f"""
Sei un esperto di simulazioni di sostenibilità che deve parlare sia ad altri esperti che normali cittadini.

Queste sono le condizioni iniziali e i risultati della simulazione corrente:

{json.dumps(summary, indent=2)}

Scrivi una descrizione testuale chiara che includa:
- La configurazione iniziale e le condizioni dell'insieme
- Risultati chiave: area sostenibile, indice di sostenibilità complessivo
- Evidenzia i vincoli critici
- Spiega le linee modali in termini semplici
- Fornisci un'interpretazione generale dei risultati
"""

# prompt = f"""
# You are a sustainability simulation expert.

# Here are the initial conditions and results of a simulation:

# {json.dumps(summary, indent=2)}

# Write a clear textual description that includes:
# - The initial setup and ensemble conditions
# - Key results: sustainable area, overall sustainability index
# - Highlight the critical constraint
# - Explain the modal lines in simple terms
# - Provide a general interpretation of the results
# """

from time import time
from ollama import Client

t1 = time()
client = Client(host="http://localhost:11434")  # adjust if using different port

# Example
messages = [{"role": "user", "content": prompt}]

response = client.chat(model="mistral", messages=messages)
print(response["message"]["content"])

print("Elapsed: ", time() - t1)
