# SPDX-License-Identifier: Apache-2.0
"""Test script for policy scenarios in the Bologna Mobility simulation."""

import argparse
import logging
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from civic_digital_twins.dt_model import DistributionIndex, Index
from examples.scenario_analysis.bologna_mobility_simulation import BolognaMobilityModel, evaluate
from examples.scenario_analysis.config_policy_scenarios import (
    BASE_PARAMS,
    BEHAVIORAL_PARAMS,
    POLICY_PARAMS,
    group_scenarios,
    scenarios,
)

logging_level = logging.INFO


def compute_global_kpis(m, evals, params):
    """Compute global Key Performance Indicators from model evaluations.

    Parameters
    ----------
    m : BolognaMobilityModel
        The model instance.
    evals : dict
        Evaluation results.
    params : dict
        Parameters including policy_params_init.

    Returns
    -------
    dict
        Dictionary of computed KPIs.
    """
    t1, t2 = _traffic_within_policy_h(
        m, evals, start=params["policy_params_init"]["i_p_start_time"], end=params["policy_params_init"]["i_p_end_time"]
    )
    avg_cost_mean = float(evals[m.outputs.modified_avg_cost].mean())
    return {
        "Inflow difference abs [veh/day]": round(
            int((evals[m.outputs.total_modified_inflow] - evals[m.outputs.total_base_inflow]).mean()) / 1000, 1
        ),
        "Inflow difference rel [%/day]": round(
            float(
                (
                    (evals[m.outputs.total_modified_inflow] - evals[m.outputs.total_base_inflow])
                    / evals[m.outputs.total_base_inflow]
                ).mean()
            )
            * 100,
            1,
        ),
        "Traffic difference abs [veh/policy-hours]": round(t1 / 1000, 1),
        "Traffic difference rel [%/policy-hours]": round(t2 * 100, 1),
        "Traffic difference abs [veh/day]": round(
            int((evals[m.outputs.total_modified_traffic] - evals[m.outputs.total_traffic]).mean()) / 1000, 1
        ),
        "Traffic difference rel [%/day]": round(
            float(
                (
                    (evals[m.outputs.total_modified_traffic] - evals[m.outputs.total_traffic])
                    / evals[m.outputs.total_traffic]
                ).mean()
            )
            * 100,
            1,
        ),
        "Emissions difference abs [NOx/day]": round(
            int((evals[m.outputs.total_modified_emissions].mean() - evals[m.outputs.total_emissions].mean())) / 1000, 1
        ),
        "Emissions difference rel [%]": round(
            float(
                (evals[m.outputs.total_modified_emissions].mean() - evals[m.outputs.total_emissions].mean())
                / evals[m.outputs.total_emissions].mean()
            )
            * 100,
            1,
        ),
        "Paying inflow [veh/day]": round(int(evals[m.outputs.total_paying].mean()) / 1000, 1)
        if avg_cost_mean > 0
        else 0,
        "Collected fees [€/day]": int(evals[m.outputs.total_paid].mean()),
        "Time-shifted abs [veh/day]": round(
            int((evals[m.outputs.total_time_shifted] + evals[m.expose.total_time_shifted_inside]).mean()) / 1000, 1
        ),
        "Mode-shifted abs [veh/day]": round(
            int((evals[m.outputs.total_mode_shifted] + evals[m.expose.total_mode_shifted_inside]).mean()) / 1000, 1
        ),
        "Lost abs [veh/day]": round(
            int((evals[m.outputs.total_lost] + evals[m.expose.total_lost_inside]).mean()) / 1000, 1
        ),
    }


def _traffic_within_policy_h(m, evals, start, end):
    times_seconds = np.array(
        [
            (t - pd.Timestamp("00:00:00")).total_seconds()
            for t in pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")
        ]
    )
    start_sec = pd.Timedelta(start).total_seconds()  # 7*3600 = 25200
    end_sec = pd.Timedelta(end).total_seconds()  # 9*3600 = 32400
    mask = (times_seconds >= start_sec) & (times_seconds < end_sec)

    t = evals[m.expose.traffic][:, mask]
    mod_t = evals[m.expose.modified_traffic][:, mask]

    maxt = np.max(t, axis=1)
    maxmod_t = np.max(mod_t, axis=1)

    return (maxmod_t - maxt).mean(), ((maxmod_t - maxt) / maxt).mean()


def update_ts_kips(
    m: BolognaMobilityModel,
    eval: dict,
    kpi_plot: dict,
    scenario: str,
):
    """Update time-series KPIs for a given scenario.

    Parameters
    ----------
    m : BolognaMobilityModel
        The model instance.
    eval : dict
        Evaluation results.
    kpi_plot : dict
        Dictionary to store KPI data for plotting.
    scenario : str
        Name of the scenario.

    Returns
    -------
    dict
        Updated kpi_plot dictionary.
    """
    kpi_plot["Inflow"][scenario] = eval[m.expose.modified_inflow]
    kpi_plot["Traffic"][scenario] = eval[m.expose.modified_traffic]
    kpi_plot["Emissions"][scenario] = eval[m.expose.modified_emissions]

    return kpi_plot


def plot_ts_kpis(
    kpi_plot: dict[str, dict[str, list]],
    nameplot: str,
    which_scenarios: list,
    path_save: Path,
):
    """Plot time-series KPIs for multiple scenarios.

    Parameters
    ----------
    kpi_plot : dict
        Dictionary containing KPI data for each scenario.
    nameplot : str
        Base name for saved plot files.
    which_scenarios : list
        List of scenario names to include in the plot.
    path_save : Path
        Directory where plot files will be saved.
    """
    time_minutes = np.arange(0, 24 * 60, 5)  # 288 elementi
    palette = plt.get_cmap("tab10")
    for quantity, kpi_scenarios in kpi_plot.items():
        plt.figure()

        for idx, name in enumerate(which_scenarios):
            kpi_scenario = np.mean(kpi_scenarios[name], axis=0)
            if name != "Base":
                plt.plot(time_minutes, kpi_scenario, label=name, color=palette(idx), linewidth=2)
            else:
                plt.plot(time_minutes, kpi_scenario, label=name, color="black", linestyle="--", linewidth=2)

        plt.xlabel("Time [minutes from midnight]", fontsize=16)
        plt.ylabel(quantity, fontsize=16)
        plt.title(f"{quantity.capitalize()}", fontsize=18)
        plt.legend(fontsize=12, loc="upper left")
        plt.grid(True)
        plt.tick_params(axis="both", which="major", labelsize=14)

        plt.tight_layout()
        plt.savefig(path_save / f"plot_{nameplot}_{quantity}.png", dpi=300)
        plt.close()


def distribution(field, size=10000, num=100):
    """Compute Poisson distribution CDF for uncertainty visualization.

    Parameters
    ----------
    field : numpy.ndarray
        Input data field.
    size : int, optional
        Maximum value for grid, by default 10000.
    num : int, optional
        Number of grid points, by default 100.

    Returns
    -------
    numpy.ndarray
        Mean of the CDF values.
    """
    xx, yy = np.meshgrid(np.linspace(0, size, num + 1), range(field.shape[1]))
    zz = stats.poisson(mu=np.expand_dims(field, axis=2)).cdf(np.expand_dims(xx, axis=0))
    return zz.mean(axis=0)


def get_custom_cmap():
    """Create a custom colormap from RGBA color strings.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Custom colormap for visualization.
    """
    field_color = [
        "rgba(255,255,255,0.0)",
        "rgba(195,95,100,0.8)",
        "rgba(165,15,21,1.0)",
        "rgba(195,95,100,0.8)",
        "rgba(255,255,255,0.0)",
    ]

    def plotly_to_mpl_color(rgba_str):
        """Parse 'rgba(r,g,b,a)' into a standard (R, G, B, A) tuple for Matplotlib."""
        numbers = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", rgba_str)]
        # Normalize R, G, B values from 0-255 down to 0-1 range
        return (numbers[0] / 255.0, numbers[1] / 255.0, numbers[2] / 255.0, numbers[3])

    mpl_colors = [plotly_to_mpl_color(c) for c in field_color]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("plotly_field", mpl_colors)
    return custom_cmap


def plot_ts_uncertinty(
    kpi_plot: dict[str, dict[str, list]],
    nameplot: str,
    path_save: Path,
):
    """Plot uncertainty visualization for KPIs across scenarios.

    Parameters
    ----------
    kpi_plot : dict
        Dictionary containing KPI data for each scenario.
    nameplot : str
        Base name for saved plot files.
    which_scenarios : list
        List of scenario names to include in the plot.
    path_save : Path
        Directory where plot files will be saved.
    """
    time_minutes = np.arange(0, 24 * 60, 5)  # 288 elementi
    for quantity, kpi_scenarios in kpi_plot.items():
        plt.figure()
        kpi_scenario = np.mean(kpi_scenarios[nameplot], axis=0)
        plt.plot(
            time_minutes,
            kpi_scenario,
            label=nameplot,
            color="black",
        )
        y_max = plt.gca().get_ylim()[1]
        dist = distribution(kpi_scenarios[nameplot], size=y_max, num=24 * 60).T
        plt.imshow(
            np.flip(dist, axis=0),
            cmap=get_custom_cmap(),
            extent=[0, 24 * 60, 0, y_max],
            interpolation="nearest",
            aspect="auto",
        )

        plt.xlabel("Time [minutes from midnight]", fontsize=16)
        plt.ylabel(quantity, fontsize=16)
        plt.title(f"{quantity.capitalize()} - {nameplot}", fontsize=18)
        plt.grid(True)
        plt.tick_params(axis="both", which="major", labelsize=14)

        plt.tight_layout()
        plt.savefig(path_save / f"plot_u_{nameplot}_{quantity}.png", dpi=300)
        plt.close()


def update_base_params(scenario: dict) -> dict:
    """Update base parameters with scenario-specific overrides.

    Parameters
    ----------
    scenario : dict
        Scenario dictionary containing optional parameter overrides.

    Returns
    -------
    dict
        Dictionary with updated policy, behavioral, and base parameters.
    """
    # Update the base params with the scenario-specific params
    policy_params_init = POLICY_PARAMS.copy()
    behavioral_params_init = BEHAVIORAL_PARAMS.copy()
    base_params_init = BASE_PARAMS.copy()
    if "policy_params" in scenario:
        policy_params_init.update(scenario["policy_params"])
    if "behavioral_params" in scenario:
        behavioral_params_init.update(scenario["behavioral_params"])
    if "base_params" in scenario:
        base_params_init.update(scenario["base_params"])
    return {
        "policy_params_init": policy_params_init,
        "behavioral_params_init": behavioral_params_init,
        "base_params_init": base_params_init,
    }


def main(scenarios: dict, group_scenarios: dict[str, list[str]], plot: bool = False):
    """Run the main simulation for all scenarios and compute KPIs.

    Parameters
    ----------
    scenarios : dict
        Dictionary of scenario definitions.
    group_scenarios : dict[str, list[str]]
        Dictionary mapping group names to scenario names.
    plot : bool, optional
        If True, generate and save plots, by default False.
    """
    logging.basicConfig(level=logging_level)

    ts_kpi_plot = {"Inflow": {}, "Traffic": {}, "Emissions": {}}
    global_kpi_save = {}

    logging.info("Starting main for %s scenarios", group_scenarios)
    i = 0
    for name, scenario in scenarios.items():
        i = i + 1
        logging.info("SCENARIO %s ------------------------------", name)
        params = update_base_params(scenario)

        # initialize the model
        logging.info("Initializing BolognaMobilityModel for scenario %s", name)
        np.random.seed(42 * 11 - i)

        pp = params["policy_params_init"]
        bp = params["behavioral_params_init"]
        inputs = BolognaMobilityModel.Inputs(
            modal_shift_option=Index("modal shift option", params["base_params_init"]["MODAL_SHIFT_OPTION"]),
            induced_demand_strategy=Index(
                "induced demand strategy", params["base_params_init"]["INDUCED_DEMAND_STRATEGY"]
            ),
            i_p_start_time=Index(
                "start time", (pd.Timestamp(pp["i_p_start_time"]) - pd.Timestamp("00:00:00")).total_seconds()
            ),
            i_p_end_time=Index(
                "end time", (pd.Timestamp(pp["i_p_end_time"]) - pd.Timestamp("00:00:00")).total_seconds()
            ),
            i_p_cost=[Index(f"cost euro {e}", pp["i_p_cost"][e]) for e in range(7)],
            i_p_fraction_exempted=Index("exempted vehicles %", pp["i_p_fraction_exempted"]),
            i_p_pt_frequency_modification=Index(
                "modification of the actual frequency of the PT", pp["i_p_pt_frequency_modification"]
            ),
            i_p_pt_capillarity_modification=Index(
                "modification of the actual capillarity of the PT", pp["i_p_pt_capillarity_modification"]
            ),
            i_p_pt_cost_modification=Index("modification of the actual cost of the PT", pp["i_p_pt_cost_modification"]),
            i_p_pt_time_modification=Index(
                "modification of the actual time difference of the PT", pp["i_p_pt_time_modification"]
            ),
            i_b_p50_cost=DistributionIndex("cost 50% threshold", stats.uniform, bp["i_b_p50_cost"]),
            i_b_p50_anticipating=Index("anticipation 50% likelihood", bp["i_b_p50_anticipating"]),
            i_b_p50_postponing=Index("postponement 50% likelihood", bp["i_b_p50_postponing"]),
            i_b_p50_anticipation=Index("anticipation distribution 50% threshold", bp["i_b_p50_anticipation"]),
            i_b_p50_postponement=Index("postponement distribution 50% threshold", bp["i_b_p50_postponement"]),
            i_b_pt_capillarity=Index("importance level of capillarity of the pt", bp["i_b_pt_capillarity"]),
            i_b_pt_frequency=Index("importance level of frequency per stop of the pt", bp["i_b_pt_frequency"]),
            i_b_pt_cost=Index("importance level of the cost of the pt", bp["i_b_pt_cost"]),
            i_b_pt_time=Index("importance level of the time difference of the car and pt", bp["i_b_pt_time"]),
            i_b_share_induced_demand=Index(
                "maximum share of additional traffic induced by congestion relief", bp["i_b_share_induced_demand"]
            ),
            i_b_p50_induced_demand=Index("share of induced demand", bp["i_b_p50_induced_demand"]),
        )
        m = BolognaMobilityModel(inputs=inputs)
        # evaluate
        logging.info("Evaluating model (size=20) for scenario %s", name)
        eval = evaluate(model=m, size=20)

        # compute and print kpis
        logging.info("Computing global KPIs for scenario %s", name)
        kpis = compute_global_kpis(m=m, evals=eval, params=params)
        global_kpi_save[name] = kpis
        logging.debug("Global KPIs for %s: %s", name, kpis)

        # save np.array with scenario temporal kpi
        logging.info("Computing time-series KPIs for scenario %s", name)
        ts_kpi_plot = update_ts_kips(m=m, eval=eval, kpi_plot=ts_kpi_plot, scenario=name)

    output_kpi_path = Path(__file__).parent.resolve() / "output" / "kpi"
    output_kpi_path.mkdir(parents=True, exist_ok=True)
    global_kpi_save_df = pd.DataFrame(global_kpi_save)
    global_kpi_save_df.to_csv(output_kpi_path / "global_kpis_scenarios.csv")

    logging.info("Saved global KPIs to output/kpi/global_kpis_scenarios.csv")

    if plot:
        # Add the base scenario to the plot data
        ts_kpi_plot["Inflow"]["Base"] = eval[m.expose.ts_inflow]
        ts_kpi_plot["Traffic"]["Base"] = eval[m.expose.traffic]
        ts_kpi_plot["Emissions"]["Base"] = eval[m.expose.emissions]

        output_img_path = Path(__file__).parent.resolve() / "output" / "img"
        output_img_path.mkdir(parents=True, exist_ok=True)
        for group_name, group_elem in group_scenarios.items():
            logging.info("Plotting time-series KPIs for scenarios %s: %s", group_name, group_elem)
            plot_ts_kpis(
                kpi_plot=ts_kpi_plot, nameplot=group_name, which_scenarios=group_elem, path_save=output_img_path
            )
        for name, _ in scenarios.items():
            logging.info("Plotting time-series uncertainty for scenario %s", name)
            plot_ts_uncertinty(kpi_plot=ts_kpi_plot, nameplot=name, path_save=output_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run policy scenarios for Bologna Mobility simulation.")
    parser.add_argument("--plot", action="store_true", help="Generate and save plots")
    args = parser.parse_args()
    main(scenarios, group_scenarios, plot=args.plot)
