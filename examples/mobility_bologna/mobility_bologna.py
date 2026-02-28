"""Example of Bologna mobility scenario.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from scipy import stats

from civic_digital_twins.dt_model import DistributionEnsemble, Index, Model, TimeseriesIndex, UniformDistIndex
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
# from civic_digital_twins.dt_model.engine import compileflags

try:
    from .mobility_bologna_data import vehicle_inflow, vehicle_starting, euro_class_split, euro_class_emission
except ImportError:
    from mobility_bologna_data import vehicle_inflow, vehicle_starting, euro_class_split, euro_class_emission

def _ts_solve(ts: np.ndarray) -> np.ndarray:
    """Solve traffic with iterative method.

    Computes steady-state circulating traffic from an inflow time series by
    iterating a simple feedback loop until convergence (50 iterations).

    Parameters
    ----------
    ts:
        Inflow timeseries.  Shape ``(T,)`` for a single scenario or
        ``(S, T)`` for an ensemble of *S* samples.

    Returns
    -------
    np.ndarray
        Circulating traffic, same shape as *ts*.
    """
    tot_traffic = 2_200_368.245_435_709  # TODO: should not be a constant value
    series = ts
    for _ in range(50):  # TODO: decide when to finish based on convergence
        mu = 1.0 + 3.0 * series.sum(axis=-1, keepdims=True) / tot_traffic
        alfa = (mu - 1.0) / mu
        series = ts + np.roll(series, 1, axis=-1) * alfa
    return series


# MODEL DEFINITION

class BolognaModel(Model):
    """Model for the Bologna mobility example."""

    def __init__(self):
        self.TS = TimeseriesIndex(
            "time range",
            np.array(
                [
                    (t - pd.Timestamp("00:00:00")).total_seconds()
                    for t in pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")
                ]
            ),
        )

        # Flow variables
        self.TS_inflow = TimeseriesIndex("inflow", vehicle_inflow)
        self.TS_starting = TimeseriesIndex("staring", vehicle_starting)

        # Indexes
        self.I_P_start_time = Index("start time", (pd.Timestamp("07:30:00") - pd.Timestamp("00:00:00")).total_seconds())
        self.I_P_end_time = Index("end time", (pd.Timestamp("19:30:00") - pd.Timestamp("00:00:00")).total_seconds())

        self.I_P_cost = [Index(f"cost euro {e}", 5.00 - e * 0.25) for e in range(7)]

        self.I_P_fraction_exempted = Index("exempted vehicles %", 0.15)

        self.I_B_p50_cost = UniformDistIndex("cost 50% threshold", loc=4.00, scale=7.00)
        self.I_B_p50_anticipating = Index("anticipation 50% likelihood", 0.5)
        self.I_B_p50_anticipation = Index("anticipation distribution 50% threshold", 0.25)
        self.I_B_p50_postponing = Index("postponement 50% likelihood", 0.8)
        self.I_B_p50_postponement = Index("postponement distribution 50% threshold", 0.50)
        self.I_B_starting_modified_factor = Index("starting modified factor", 1.00)

        self.I_avg_cost = Index(
            "average cost",
            self.I_P_cost[0] * euro_class_split["euro_0"]
            + self.I_P_cost[1] * euro_class_split["euro_1"]
            + self.I_P_cost[2] * euro_class_split["euro_2"]
            + self.I_P_cost[3] * euro_class_split["euro_3"]
            + self.I_P_cost[4] * euro_class_split["euro_4"]
            + self.I_P_cost[5] * euro_class_split["euro_5"]
            + self.I_P_cost[6] * euro_class_split["euro_6"],
        )

        self.I_fraction_rigid_euro = [
            Index(
                f"rigid vehicles euro_{e} %",
                (1 - self.I_P_fraction_exempted) * (np.e ** (self.I_P_cost[e] / self.I_B_p50_cost * np.log(0.5))),
            )
            for e in range(7)
        ]

        self.I_fraction_rigid = Index(
            "rigid vehicles %",
            self.I_fraction_rigid_euro[0] * euro_class_split["euro_0"]
            + self.I_fraction_rigid_euro[1] * euro_class_split["euro_1"]
            + self.I_fraction_rigid_euro[2] * euro_class_split["euro_2"]
            + self.I_fraction_rigid_euro[3] * euro_class_split["euro_3"]
            + self.I_fraction_rigid_euro[4] * euro_class_split["euro_4"]
            + self.I_fraction_rigid_euro[5] * euro_class_split["euro_5"]
            + self.I_fraction_rigid_euro[6] * euro_class_split["euro_6"],
        )

        self.I_modified_euro_class_split = [
            Index(
                f"modified split euro_{e} %",
                euro_class_split[f"euro_{e}"]
                * (self.I_P_fraction_exempted + self.I_fraction_rigid_euro[e])
                / (self.I_P_fraction_exempted + self.I_fraction_rigid),
            )
            for e in range(7)
        ]

        self.I_delta_from_start = TimeseriesIndex(
            "delta time from start",
            graph.piecewise(
                ((self.TS - self.I_P_start_time) / pd.Timedelta("1h").total_seconds(), self.TS >= self.I_P_start_time),
                (np.inf, True),
            ),
        )

        self.I_fraction_anticipating = TimeseriesIndex(
            "anticipating vehicles %",
            np.e ** (self.I_delta_from_start / self.I_B_p50_anticipating * np.log(0.5))
            * (1 - self.I_P_fraction_exempted - self.I_fraction_rigid),
        )

        self.I_number_anticipating = TimeseriesIndex(
            "anticipating vehicles", self.I_fraction_anticipating * self.TS_inflow
        )

        self.I_delta_to_end = TimeseriesIndex(
            "delta time to end",
            graph.piecewise(
                ((self.I_P_end_time - self.TS) / pd.Timedelta("1h").total_seconds(), self.TS <= self.I_P_end_time),
                (np.inf, True),
            ),
        )

        self.I_fraction_postponing = TimeseriesIndex(
            "postponing vehicles %",
            np.e ** (self.I_delta_to_end / self.I_B_p50_postponing * np.log(0.5))
            * (1 - self.I_P_fraction_exempted - self.I_fraction_rigid),
        )

        self.I_number_postponing = TimeseriesIndex("postponing vehicles", self.I_fraction_postponing * self.TS_inflow)

        self.I_total_anticipating = Index("total anticipating vehicles", self.I_number_anticipating.sum())

        self.I_total_postponing = Index("total postponing vehicles", self.I_number_postponing.sum())

        self.I_delta_before_start = TimeseriesIndex(
            "delta time before start",
            graph.piecewise(
                ((self.I_P_start_time - self.TS) / pd.Timedelta("1h").total_seconds(), self.TS < self.I_P_start_time),
                (np.inf, True),
            ),
        )
        self.I_number_anticipated = TimeseriesIndex(
            "anticipated vehicles",
            np.e ** (self.I_delta_before_start / self.I_B_p50_anticipation * np.log(0.5))
            / self.I_B_p50_anticipation
            * np.log(2)
            / 12
            * self.I_total_anticipating,
        )

        self.I_delta_after_end = TimeseriesIndex(
            "delta time after end",
            graph.piecewise(
                ((self.TS - self.I_P_end_time) / pd.Timedelta("1h").total_seconds(), self.TS > self.I_P_end_time),
                (np.inf, True),
            ),
        )
        self.I_number_postponed = TimeseriesIndex(
            "postponed vehicles",
            np.e ** (self.I_delta_after_end / self.I_B_p50_postponement * np.log(0.5))
            / self.I_B_p50_postponement
            * np.log(2)
            / 12
            * self.I_total_postponing,
        )

        self.I_number_shifted = TimeseriesIndex("shifted vehicles", self.I_number_anticipated + self.I_number_postponed)

        self.I_modified_inflow = Index(
            "modified vehicle inflow",
            graph.piecewise(
                (
                    (self.I_P_fraction_exempted + self.I_fraction_rigid) * self.TS_inflow,
                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time),
                ),
                (self.TS_inflow + self.I_number_shifted, True),
            ),
        )

        self.I_total_base_inflow = Index("total base vehicle flow", self.TS_inflow.sum())
        self.I_total_modified_inflow = Index("total modified vehicle inflow", self.I_modified_inflow.sum())
        self.I_number_paying = Index(
            "paying vehicles",
            graph.piecewise(
                (
                    self.I_fraction_rigid * self.TS_inflow,
                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time),
                ),
                (0, True),
            ),
        )

        self.I_total_paying = Index("total vehicles paying", self.I_number_paying.sum())

        self.I_modified_starting = Index(
            "modified starting", self.TS_starting + self.I_modified_inflow * (self.I_B_starting_modified_factor - 1)
        )

        # TODO: flat rates for all AV
        self.I_inflow_ratio = Index(
            "ratio between modified flow and base flow", self.TS_inflow / self.I_modified_inflow
        )

        self.I_starting_ratio = Index(
            "ratio between modified starting and base starting", self.TS_starting / self.I_modified_starting
        )

        # TODO: fix, compute real value!
        self.I_total_payed = Index("total payed fees", self.I_total_paying * self.I_avg_cost)

        self.I_total_anticipated = Index("total vehicles anticipated", self.I_number_anticipated.sum())

        self.I_total_postponed = Index("total vehicles postponed", self.I_number_postponed.sum())

        self.I_total_shifted = Index("total vehicles shifted", self.I_total_anticipated + self.I_total_postponed)

        self.I_traffic = TimeseriesIndex(
            "reference traffic", graph.function_call("ts_solve", self.TS_inflow + self.TS_starting)
        )

        self.I_modified_traffic = TimeseriesIndex(
            "modified traffic", graph.function_call("ts_solve", self.I_modified_inflow + self.I_modified_starting)
        )

        self.I_traffic_ratio = Index(
            "ratio between modified traffic and base traffic", self.I_traffic / self.I_modified_traffic
        )

        self.I_average_emissions = Index(
            "average emissions (per vehicle, per km)",
            euro_class_emission["euro_0"] * euro_class_split["euro_0"]
            + euro_class_emission["euro_1"] * euro_class_split["euro_1"]
            + euro_class_emission["euro_2"] * euro_class_split["euro_2"]
            + euro_class_emission["euro_3"] * euro_class_split["euro_3"]
            + euro_class_emission["euro_4"] * euro_class_split["euro_4"]
            + euro_class_emission["euro_5"] * euro_class_split["euro_5"]
            + euro_class_emission["euro_6"] * euro_class_split["euro_6"],
        )

        self.I_modified_average_emissions = Index(
            "modified average emissions (per vehicle, per km)",
            euro_class_emission["euro_0"] * self.I_modified_euro_class_split[0]
            + euro_class_emission["euro_1"] * self.I_modified_euro_class_split[1]
            + euro_class_emission["euro_2"] * self.I_modified_euro_class_split[2]
            + euro_class_emission["euro_3"] * self.I_modified_euro_class_split[3]
            + euro_class_emission["euro_4"] * self.I_modified_euro_class_split[4]
            + euro_class_emission["euro_5"] * self.I_modified_euro_class_split[5]
            + euro_class_emission["euro_6"] * self.I_modified_euro_class_split[6],
        )

        # TODO: improve - at the moment, the conversion factor is 2,5 km per 5 minutes
        self.I_emissions = TimeseriesIndex("emissions", 2.5 * self.I_average_emissions * self.I_traffic)

        # I_modified_emissions = Index('modified emissions', 2.5 * I_modified_average_emissions * I_modified_traffic)
        #
        # TODO: The average emissions is probably different outside regulated hours
        #  (shifted cars' emissions are probably proportional to shifted cars' euro level mix)
        #
        self.I_modified_emissions = Index(
            "modified emissions",
            graph.piecewise(
                (
                    2.5 * self.I_modified_average_emissions * self.I_modified_traffic,
                    (self.TS >= self.I_P_start_time) & (self.TS <= self.I_P_end_time),
                ),
                (2.5 * self.I_average_emissions * self.I_modified_traffic, True),
            ),
        )

        self.I_total_emissions = Index("total emissions", self.I_emissions.sum())

        self.I_total_modified_emissions = Index("total modified emissions", self.I_modified_emissions.sum())

        self.indexes = [
            self.TS,
            self.TS_inflow,
            self.TS_starting,
            self.I_P_start_time,
            self.I_P_end_time,
            *self.I_P_cost,
            self.I_P_fraction_exempted,
            self.I_B_p50_cost,
            self.I_B_p50_anticipating,
            self.I_B_p50_anticipation,
            self.I_B_p50_postponing,
            self.I_B_p50_postponement,
            self.I_B_starting_modified_factor,
            self.I_avg_cost,
            *self.I_fraction_rigid_euro,
            self.I_fraction_rigid,
            *self.I_modified_euro_class_split,
            self.I_delta_from_start,
            self.I_fraction_anticipating,
            self.I_number_anticipating,
            self.I_delta_to_end,
            self.I_fraction_postponing,
            self.I_number_postponing,
            self.I_total_anticipating,
            self.I_total_postponing,
            self.I_delta_before_start,
            self.I_number_anticipated,
            self.I_delta_after_end,
            self.I_number_postponed,
            self.I_number_shifted,
            self.I_modified_inflow,
            self.I_modified_starting,
            self.I_inflow_ratio,
            self.I_starting_ratio,
            self.I_total_base_inflow,
            self.I_total_modified_inflow,
            self.I_number_paying,
            self.I_total_paying,
            self.I_total_payed,
            self.I_total_anticipated,
            self.I_total_postponed,
            self.I_total_shifted,
            self.I_traffic,
            self.I_modified_traffic,
            self.I_traffic_ratio,
            self.I_average_emissions,
            self.I_modified_average_emissions,
            self.I_emissions,
            self.I_modified_emissions,
            self.I_total_emissions,
            self.I_total_modified_emissions,
        ]
        super().__init__("Bologna mobility", self.indexes)


def evaluate(model: BolognaModel, size: int = 1) -> dict:
    """Evaluate *model* over an ensemble of *size* samples.

    Draws *size* samples from each distribution-backed abstract index via
    :class:`~dt_model.simulation.ensemble.DistributionEnsemble`, runs the
    standard :class:`~dt_model.simulation.evaluation.Evaluation` engine, and
    returns a ``subs`` dict mapping each index to a 2-D array so that ``[0]``
    indexing and ``.mean(axis=0)`` work uniformly in the helpers below.
    """
    ensemble = DistributionEnsemble(model, size)

    result = Evaluation(model).evaluate(
        ensemble,
        functions={"ts_solve": executor.LambdaAdapter(_ts_solve)},
    )

    subs = {}
    for idx in model.indexes:
        val = result[idx]
        if val.ndim == 0:
            subs[idx] = np.full((size, 1), float(val))
        elif val.ndim == 1:
            subs[idx] = np.expand_dims(val, axis=0)
        else:
            subs[idx] = val
    return subs


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
    ax.set_ylim([0, vertical_size])
    if vertical_formatter is not None:
        ax.yaxis.set_major_formatter(vertical_formatter)
    ax.set_ylabel(vertical_label)
    fig.tight_layout()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    ax.set_xlabel(horizontal_label)
    ax.legend(loc="upper right")
    return fig


def compute_kpis(m, evals):
    """Compute the KPIs for the mobility example."""
    return {
        "Base inflow [veh/day]": int(evals[m.I_total_base_inflow].mean()),
        "Modified inflow [veh/day]": int(evals[m.I_total_modified_inflow].mean()),
        "Shifted inflow [veh/day]": int(evals[m.I_total_shifted].mean()),
        "Paying inflow [veh/day]": int(evals[m.I_total_paying].mean()) if evals[m.I_avg_cost].mean() > 0 else 0,
        "Collected fees [€/day]": int(evals[m.I_total_payed].mean()),
        "Emissions [NOx gr/day]": int(evals[m.I_total_modified_emissions].mean()),
        "Modified emissions [NOx gr/day]": int(evals[m.I_total_emissions].mean())
        - int(evals[m.I_total_modified_emissions].mean()),
    }


def roundup(val):
    """Compute a rounded-up approximation of `val`."""
    v = val * 1.4
    s = math.floor(math.log10(v * 1.3))
    return round(v / 10**s) * 10**s


if __name__ == "__main__":
    m = BolognaModel()
    subs = evaluate(m, 20)

    plot_field_graph(
        subs[m.I_modified_inflow],
        horizontal_label="Time",
        vertical_label="Flow (vehicles/hour)",
        vertical_size=1600,
        vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
        reference_line=subs[m.TS_inflow][0],
    )
    plt.show()
    plot_field_graph(
        subs[m.I_modified_traffic],
        horizontal_label="Time",
        vertical_label="Traffic (circulating vehicles)",
        vertical_size=15000,
        reference_line=subs[m.I_traffic][0],
    )
    plt.show()
    plot_field_graph(
        subs[m.I_modified_emissions],
        horizontal_label="Time",
        vertical_label="Emissions (NOx gr/h)",
        vertical_size=4000,
        vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
        reference_line=subs[m.I_emissions][0],
    )
    plt.show()
    for k, v in compute_kpis(m, subs).items():
        print(f"{k} - {v:,}")
