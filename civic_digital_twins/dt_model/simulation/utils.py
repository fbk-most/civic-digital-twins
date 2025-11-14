from pathlib import Path
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from civic_digital_twins.dt_model import InstantiatedModel
from civic_digital_twins.dt_model import Ensemble, Evaluation
from civic_digital_twins.dt_model.reference_models.molveno.overtourism import (
    I_P_excursionists_reduction_factor,
    I_P_excursionists_saturation_level,
    I_P_tourists_reduction_factor,
    I_P_tourists_saturation_level,
    PV_excursionists,
    PV_tourists,
    M_Base,
)


def presence_transformation(presence, reduction_factor, saturation_level, sharpness=3):
    tmp = presence * reduction_factor
    return (
        tmp
        * saturation_level
        / ((tmp**sharpness + saturation_level**sharpness) ** (1 / sharpness))
    )


def compute_scenario(model, scenario_config, early_stopping):
    """Compute all data for a given scenario"""

    # static data, might parameters later
    (t_max, e_max) = (10000, 10000)
    (t_sample, e_sample) = (100, 100)
    target_presence_samples = 200
    ensemble_size = 20

    ensemble = Ensemble(model, scenario_config, cv_ensemble_size=ensemble_size)
    scenario_hash = ensemble.compute_hash(
        [t_max, e_max, t_sample, e_sample, target_presence_samples]
    )
    evaluation = Evaluation(model, ensemble)

    tt = np.linspace(0, t_max, t_sample + 1)
    ee = np.linspace(0, e_max, e_sample + 1)

    if early_stopping:
        zz = evaluation.evaluate_grid_incremental(
            {PV_tourists: tt, PV_excursionists: ee}, early_stopping
        )
    else:
        zz = evaluation.evaluate_grid({PV_tourists: tt, PV_excursionists: ee})

    sample_tourists = [
        presence_transformation(
            presence,
            evaluation.get_index_mean_value(I_P_tourists_reduction_factor),
            evaluation.get_index_mean_value(I_P_tourists_saturation_level),
        )
        for c in ensemble
        for presence in PV_tourists.sample(
            cvs=c[1], nr=max(1, round(c[0] * target_presence_samples))
        )
    ]

    sample_excursionists = [
        presence_transformation(
            presence,
            evaluation.get_index_mean_value(I_P_excursionists_reduction_factor),
            evaluation.get_index_mean_value(I_P_excursionists_saturation_level),
        )
        for c in ensemble
        for presence in PV_excursionists.sample(
            cvs=c[1], nr=max(1, round(c[0] * target_presence_samples))
        )
    ]

    return {
        "evaluation": evaluation,
        "zz": zz,
        "sample_tourists": sample_tourists,
        "sample_excursionists": sample_excursionists,
        "t_max": t_max,
        "t_sample": t_sample,
        "e_max": e_max,
        "e_sample": e_sample,
    }, scenario_hash


def compute_scenario_worker(scenario_config: dict, early_stopping):
    """
    Compute one scenario and save the results to disk.

    Args:
        scenario_name: name of the scenario ("Base", "GoodWeather", etc.)
        params: optional dict with custom parameters (future-proof)
    Returns:
        Path to the saved results file.
    """

    # Each worker builds its own model
    IM_Base = InstantiatedModel(M_Base, values=scenario_config)

    # Compute scenario data
    result, scenario_name = compute_scenario(IM_Base, scenario_config, early_stopping)

    return scenario_name, result


def plot_scenario(data, filename: str | Path = None):
    """Plot a single scenario using precomputed data."""

    # Compute relevant parts
    tt = np.linspace(0, data["t_max"], data["t_sample"] + 1)
    ee = np.linspace(0, data["e_max"], data["e_sample"] + 1)
    xx, yy = np.meshgrid(tt, ee)
    evaluation = data["evaluation"]

    area = evaluation.compute_sustainable_area()
    i, c = evaluation.compute_sustainability_index_with_ci(
        list(zip(data["sample_tourists"], data["sample_excursionists"])), confidence=0.8
    )
    sust_indexes = evaluation.compute_sustainability_index_with_ci_per_constraint(
        list(zip(data["sample_tourists"], data["sample_excursionists"])), confidence=0.8
    )
    critical = min(sust_indexes, key=lambda k: sust_indexes[k][0])
    modals = evaluation.compute_modal_line_per_constraint()

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(xx, yy, data["zz"], cmap="coolwarm_r", vmin=0.0, vmax=1.0)

    for modal in modals.values():
        ax.plot(*modal, color="black", linewidth=2)

    ax.scatter(
        data["sample_excursionists"],
        data["sample_tourists"],
        color="gainsboro",
        edgecolors="black",
    )

    critical_mean, critical_ci = sust_indexes[critical]
    ax.set_title(
        f"Area = {area / 10e6:.2f} kp$^2$ - "
        f"Sustainability = {i * 100:.2f}% ± {c * 100:.2f}%\n"
        f"Critical = {critical.capacity.name}"
        f" ({critical_mean * 100:.2f}% ± {critical_ci * 100:.2f}%)",
        fontsize=12,
    )
    ax.set_xlim(0, data["t_max"])
    ax.set_ylim(0, data["e_max"])
    fig.colorbar(ScalarMappable(Normalize(0, 1), cmap="coolwarm_r"), ax=ax)
    ax.set_xlabel("Tourists")
    ax.set_ylabel("Excursionists")

    if filename:
        plt.savefig(str(filename), bbox_inches="tight")
    else:
        plt.show()
