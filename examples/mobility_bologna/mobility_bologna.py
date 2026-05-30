# SPDX-License-Identifier: Apache-2.0
"""Bologna mobility scenario — visualization helpers and entry point."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be called before any other matplotlib sub-imports

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from civic_digital_twins.dt_model import Index, Scenario
from civic_digital_twins.dt_model.simulation.runner import EvaluationConfig

try:
    from .bologna_model import BolognaEvaluator, BolognaModel, BolognaOutput
except ImportError:
    from bologna_model import BolognaEvaluator, BolognaModel, BolognaOutput


def distribution(field, size=10000, num=100):
    """Compute the field distribution for graphical display."""
    xx, yy = np.meshgrid(np.linspace(0, size, num + 1), range(field.shape[1]))
    zz = stats.poisson(mu=np.expand_dims(field, axis=2)).cdf(np.expand_dims(xx, axis=0))
    return zz.mean(axis=0)


field_color = (165 / 256, 15 / 256, 21 / 256)
delta = 0.5
field_light_color = (
    (field_color[0] + delta) / (1 + delta),
    (field_color[1] + delta) / (1 + delta),
    (field_color[2] + delta) / (1 + delta),
)

field_colormap = LinearSegmentedColormap.from_list(
    "mid_red_bar", colors=["white", field_light_color, field_color, field_light_color, "white"], N=100
)


def plot_field_graph(
    field, horizontal_label, vertical_label, vertical_size=None, vertical_formatter=None, reference_line=None
):
    """Generate plot figure."""
    if vertical_size is None:
        vertical_size = roundup(np.max(field))
    dist = distribution(field, vertical_size, 100)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    pcm = ax.pcolormesh(
        pd.date_range(start="00:00:00", periods=12 * 24, freq="5min"),
        np.linspace(0, vertical_size, 100 + 1),
        dist.T,
        cmap=field_colormap,
        vmin=0.0,
        vmax=1.0,
    )
    if reference_line is not None:
        ax.plot(
            pd.date_range(start="00:00:00", periods=12 * 24, freq="5min"),
            reference_line,
            "--",
            linewidth=1,
            color="black",
            label="Riferimento",
        )
    ax.plot(
        pd.date_range(start="00:00:00", periods=12 * 24, freq="5min"),
        field.mean(axis=0),
        linewidth=1,
        color="black",
        label="Modificato (mediana)",
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_ticks([0.00, 0.25, 0.50, 0.75, 1.00])
    cbar.set_ticklabels([f"{x}%" for x in [0, 25, 50, 75, 100]])
    ax.set_ylim((0, vertical_size))
    if vertical_formatter is not None:
        ax.yaxis.set_major_formatter(vertical_formatter)
    ax.set_ylabel(vertical_label)
    fig.tight_layout()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    ax.set_xlabel(horizontal_label)
    ax.legend(loc="upper right")
    return fig


def roundup(val):
    """Compute a rounded-up approximation of `val`."""
    v = val * 1.4
    s = math.floor(math.log10(v * 1.3))
    return round(v / 10**s) * 10**s


def _save_scenario_plots(label: str, m: BolognaModel, output: BolognaOutput, out: Path) -> None:
    """Save inflow, traffic and emissions field graphs for one scenario."""
    fig = plot_field_graph(
        output.fields["modified_inflow"],
        horizontal_label="Time",
        vertical_label="Flow (vehicles/hour)",
        vertical_size=1600,
        vertical_formatter=mticker.FuncFormatter(lambda x, _: f"{int(x * 12)}"),
        reference_line=output.timeseries["ts_inflow"],
    )
    fig.savefig(out / f"{label}_inflow.png", dpi=150)
    plt.close(fig)

    fig = plot_field_graph(
        output.fields["modified_traffic"],
        horizontal_label="Time",
        vertical_label="Traffic (circulating vehicles)",
        vertical_size=15000,
        reference_line=output.timeseries["traffic"],
    )
    fig.savefig(out / f"{label}_traffic.png", dpi=150)
    plt.close(fig)

    fig = plot_field_graph(
        output.fields["modified_emissions"],
        horizontal_label="Time",
        vertical_label="Emissions (NOx gr/h)",
        vertical_size=4000,
        vertical_formatter=mticker.FuncFormatter(lambda x, _: f"{int(x * 12)}"),
        reference_line=output.timeseries["emissions"],
    )
    fig.savefig(out / f"{label}_emissions.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    _out = Path(__file__).parent / "output"
    _out.mkdir(exist_ok=True)

    _config = EvaluationConfig(ensemble_size=20)

    # ── Reference scenario (default parameters) ──────────────────────────────
    _m = BolognaModel(inputs=BolognaModel.default_inputs(), fns=BolognaModel.default_fns())
    _evaluator = BolognaEvaluator(_m)
    _output = _evaluator.evaluate(Scenario(_m), _config)
    _save_scenario_plots("reference", _m, _output, _out)

    print("Reference scenario:")
    for k, v in _output.kpis.items():
        print(f"  {k} - {v:,}")

    # ── Stricter pricing scenario ─────────────────────────────────────────────
    # Higher fees with a steeper Euro-class gradient: older/more polluting
    # vehicles pay substantially more, incentivising fleet-mix shifts.
    _m_strict = BolognaModel(
        inputs=dataclasses.replace(
            BolognaModel.default_inputs(),
            i_p_cost=[Index(f"cost euro {e}", 8.00 - e * 0.50) for e in range(7)],
        ),
        fns=BolognaModel.default_fns(),
    )
    _evaluator_strict = BolognaEvaluator(_m_strict)
    _output_strict = _evaluator_strict.evaluate(Scenario(_m_strict), _config)
    _save_scenario_plots("strict", _m_strict, _output_strict, _out)

    print("\nStricter pricing scenario (euro_0: 8.00 €, euro_6: 5.00 €):")
    for k, v in _output_strict.kpis.items():
        print(f"  {k} - {v:,}")
