"""Example to interactively explore the frozen Molveno model.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")  # must be called before any other matplotlib sub-imports

from civic_digital_twins.dt_model import Scenario
from civic_digital_twins.dt_model.simulation.runner import EvaluationConfig

try:
    from overtourism_molveno.molveno_model import (
        MolvenoEvaluator,
        MolvenoModel,
        MolvenoOutput,
    )
except ImportError:
    from molveno_model import MolvenoEvaluator, MolvenoModel, MolvenoOutput

model = MolvenoModel()

# PLOTTING

(t_max, e_max) = (10000, 10000)
(t_sample, e_sample) = (100, 100)
target_presence_samples = 200
ensemble_size = 20  # TODO: make configurable; may it be a CV parameter?


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_scenario(model: MolvenoModel, output: MolvenoOutput, title: str) -> Figure:
    """Render the sustainability field and KPIs onto a figure.

    Parameters
    ----------
    model : MolvenoModel
        The :class:`~overtourism_molveno.molveno_model.MolvenoModel` being plotted.
    output : MolvenoOutput
        Evaluated output carrying the field, axes, and presence samples.
    title : str
        Plot title prefix.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 10), layout="constrained")

    area = output.sustainable_area
    (i, c_ci) = output.sustainability_index
    sust_indexes = output.sustainability_by_constraint
    critical_name = min(sust_indexes, key=lambda k: sust_indexes[k][0])
    critical = next(c for c in model.constraints if c.name == critical_name)
    modals = output.modal_lines

    # field has shape (N_t, N_e); pcolormesh expects (N_e, N_t) for meshgrid(tt, ee).
    xx, yy = np.meshgrid(output.tt, output.ee)
    ax.pcolormesh(xx, yy, output.field.T, cmap="coolwarm_r", vmin=0.0, vmax=1.0)
    for modal in modals.values():
        ax.plot(*modal, color="black", linewidth=2)
    ax.scatter(output.sample_excursionists, output.sample_tourists, color="gainsboro", edgecolors="black")
    ax.set_title(
        f"{title}\n"
        + f"area = {area / 10e6:.2f} kp$^2$ - "
        + f"Sustainability = {i * 100:.2f}% +/- {c_ci * 100:.2f}%\n"
        + f"Critical = {critical.capacity.name}"
        + f"({sust_indexes[critical_name][0] * 100:.2f}% +/- {sust_indexes[critical_name][1] * 100:.2f}%)",
        fontsize=12,
    )
    ax.set_xlim(left=0, right=output.tt.max())
    ax.set_ylim(bottom=0, top=output.ee.max())

    return fig


if __name__ == "__main__":
    start_time = time.time()

    _out = Path(__file__).parent / "output"
    _out.mkdir(exist_ok=True)

    evaluator = MolvenoEvaluator(
        model,
        t_max=t_max,
        e_max=e_max,
        t_sample=t_sample,
        e_sample=e_sample,
        target_presence_samples=target_presence_samples,
    )
    config = EvaluationConfig(ensemble_size=ensemble_size)

    output_base = evaluator.evaluate(Scenario(model), config)
    fig_base = plot_scenario(model, output_base, "Base")
    fig_base.savefig(_out / "base.png", dpi=150)
    plt.close(fig_base)

    output_good = evaluator.evaluate(
        Scenario(model, overrides={model.cv_weather: {"good": 0.5, "unsettled": 0.5}}), config
    )
    fig_good_weather = plot_scenario(model, output_good, "Good weather")
    fig_good_weather.savefig(_out / "good_weather.png", dpi=150)
    plt.close(fig_good_weather)

    output_bad = evaluator.evaluate(Scenario(model, overrides={model.cv_weather: "bad"}), config)
    fig_bad_weather = plot_scenario(model, output_bad, "Bad weather")
    fig_bad_weather.savefig(_out / "bad_weather.png", dpi=150)
    plt.close(fig_bad_weather)

    print("--- %s seconds ---" % (time.time() - start_time))
