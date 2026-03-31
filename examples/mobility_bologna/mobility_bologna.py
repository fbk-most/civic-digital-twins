"""Example of Bologna mobility scenario.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # must be called before any other matplotlib sub-imports

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from civic_digital_twins.dt_model import DistributionEnsemble, DistributionIndex, Index, Model, TimeseriesIndex
from civic_digital_twins.dt_model.engine.frontend import graph
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation, EvaluationResult

try:
    from .mobility_bologna_data import euro_class_emission, euro_class_split, vehicle_inflow, vehicle_starting
except ImportError:
    from mobility_bologna_data import euro_class_emission, euro_class_split, vehicle_inflow, vehicle_starting


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


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class InflowModel(Model):
    """Sub-model that computes modified vehicle inflow under a pricing policy.

    Computes per-euro-class rigidity fractions, anticipating/postponing
    behaviour, the modified inflow and starting timeseries, and payment
    statistics.
    """

    @dataclass
    class Inputs:
        """Inputs of :class:`InflowModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        ts: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_cost: list[Index]
        i_p_fraction_exempted: Index
        i_b_p50_cost: DistributionIndex
        i_b_p50_anticipating: Index
        i_b_p50_anticipation: Index
        i_b_p50_postponing: Index
        i_b_p50_postponement: Index
        i_b_starting_modified_factor: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`InflowModel`."""

        modified_inflow: Index
        modified_starting: Index
        total_base_inflow: Index
        total_modified_inflow: Index
        fraction_rigid: Index
        modified_euro_class_split: list[Index]
        number_paying: Index
        total_paying: Index
        avg_cost: Index
        total_paid: Index
        total_shifted: Index

    @dataclass
    class Expose:
        """Inspectable intermediate indexes of :class:`InflowModel`."""

        i_fraction_rigid_euro: list[Index]
        i_delta_from_start: TimeseriesIndex
        i_fraction_anticipating: TimeseriesIndex
        i_number_anticipating: TimeseriesIndex
        i_delta_to_end: TimeseriesIndex
        i_fraction_postponing: TimeseriesIndex
        i_number_postponing: TimeseriesIndex
        i_total_anticipating: Index
        i_total_postponing: Index
        i_delta_before_start: TimeseriesIndex
        i_number_anticipated: TimeseriesIndex
        i_delta_after_end: TimeseriesIndex
        i_number_postponed: TimeseriesIndex
        i_number_shifted: TimeseriesIndex
        i_total_anticipated: Index
        i_total_postponed: Index

    def __init__(
        self,
        ts_inflow: TimeseriesIndex,
        ts_starting: TimeseriesIndex,
        ts: TimeseriesIndex,
        i_p_start_time: Index,
        i_p_end_time: Index,
        i_p_cost: list[Index],
        i_p_fraction_exempted: Index,
        i_b_p50_cost: DistributionIndex,
        i_b_p50_anticipating: Index,
        i_b_p50_anticipation: Index,
        i_b_p50_postponing: Index,
        i_b_p50_postponement: Index,
        i_b_starting_modified_factor: Index,
    ) -> None:
        Inputs = InflowModel.Inputs
        Outputs = InflowModel.Outputs
        Expose = InflowModel.Expose

        inputs = Inputs(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            ts=ts,
            i_p_start_time=i_p_start_time,
            i_p_end_time=i_p_end_time,
            i_p_cost=i_p_cost,
            i_p_fraction_exempted=i_p_fraction_exempted,
            i_b_p50_cost=i_b_p50_cost,
            i_b_p50_anticipating=i_b_p50_anticipating,
            i_b_p50_anticipation=i_b_p50_anticipation,
            i_b_p50_postponing=i_b_p50_postponing,
            i_b_p50_postponement=i_b_p50_postponement,
            i_b_starting_modified_factor=i_b_starting_modified_factor,
        )

        avg_cost = Index(
            "average cost",
            inputs.i_p_cost[0] * euro_class_split["euro_0"]
            + inputs.i_p_cost[1] * euro_class_split["euro_1"]
            + inputs.i_p_cost[2] * euro_class_split["euro_2"]
            + inputs.i_p_cost[3] * euro_class_split["euro_3"]
            + inputs.i_p_cost[4] * euro_class_split["euro_4"]
            + inputs.i_p_cost[5] * euro_class_split["euro_5"]
            + inputs.i_p_cost[6] * euro_class_split["euro_6"],
        )

        i_fraction_rigid_euro = [
            Index(
                f"rigid vehicles euro_{e} %",
                (1 - inputs.i_p_fraction_exempted) * (np.e ** (inputs.i_p_cost[e] / inputs.i_b_p50_cost * np.log(0.5))),
            )
            for e in range(7)
        ]

        fraction_rigid = Index(
            "rigid vehicles %",
            i_fraction_rigid_euro[0] * euro_class_split["euro_0"]
            + i_fraction_rigid_euro[1] * euro_class_split["euro_1"]
            + i_fraction_rigid_euro[2] * euro_class_split["euro_2"]
            + i_fraction_rigid_euro[3] * euro_class_split["euro_3"]
            + i_fraction_rigid_euro[4] * euro_class_split["euro_4"]
            + i_fraction_rigid_euro[5] * euro_class_split["euro_5"]
            + i_fraction_rigid_euro[6] * euro_class_split["euro_6"],
        )

        modified_euro_class_split = [
            Index(
                f"modified split euro_{e} %",
                euro_class_split[f"euro_{e}"]
                * (inputs.i_p_fraction_exempted + i_fraction_rigid_euro[e])
                / (inputs.i_p_fraction_exempted + fraction_rigid),
            )
            for e in range(7)
        ]

        i_delta_from_start = TimeseriesIndex(
            "delta time from start",
            graph.piecewise(
                (
                    (inputs.ts - inputs.i_p_start_time) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts >= inputs.i_p_start_time,
                ),
                (np.inf, True),
            ),
        )

        i_fraction_anticipating = TimeseriesIndex(
            "anticipating vehicles %",
            np.e ** (i_delta_from_start / inputs.i_b_p50_anticipating * np.log(0.5))
            * (1 - inputs.i_p_fraction_exempted - fraction_rigid),
        )

        i_number_anticipating = TimeseriesIndex("anticipating vehicles", i_fraction_anticipating * inputs.ts_inflow)

        i_delta_to_end = TimeseriesIndex(
            "delta time to end",
            graph.piecewise(
                (
                    (inputs.i_p_end_time - inputs.ts) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts <= inputs.i_p_end_time,
                ),
                (np.inf, True),
            ),
        )

        i_fraction_postponing = TimeseriesIndex(
            "postponing vehicles %",
            np.e ** (i_delta_to_end / inputs.i_b_p50_postponing * np.log(0.5))
            * (1 - inputs.i_p_fraction_exempted - fraction_rigid),
        )

        i_number_postponing = TimeseriesIndex("postponing vehicles", i_fraction_postponing * inputs.ts_inflow)

        i_total_anticipating = Index("total anticipating vehicles", i_number_anticipating.sum())
        i_total_postponing = Index("total postponing vehicles", i_number_postponing.sum())

        i_delta_before_start = TimeseriesIndex(
            "delta time before start",
            graph.piecewise(
                (
                    (inputs.i_p_start_time - inputs.ts) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts < inputs.i_p_start_time,
                ),
                (np.inf, True),
            ),
        )

        i_number_anticipated = TimeseriesIndex(
            "anticipated vehicles",
            np.e ** (i_delta_before_start / inputs.i_b_p50_anticipation * np.log(0.5))
            / inputs.i_b_p50_anticipation
            * np.log(2)
            / 12
            * i_total_anticipating,
        )

        i_delta_after_end = TimeseriesIndex(
            "delta time after end",
            graph.piecewise(
                (
                    (inputs.ts - inputs.i_p_end_time) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts > inputs.i_p_end_time,
                ),
                (np.inf, True),
            ),
        )

        i_number_postponed = TimeseriesIndex(
            "postponed vehicles",
            np.e ** (i_delta_after_end / inputs.i_b_p50_postponement * np.log(0.5))
            / inputs.i_b_p50_postponement
            * np.log(2)
            / 12
            * i_total_postponing,
        )

        i_number_shifted = TimeseriesIndex("shifted vehicles", i_number_anticipated + i_number_postponed)

        modified_inflow = Index(
            "modified vehicle inflow",
            graph.piecewise(
                (
                    (inputs.i_p_fraction_exempted + fraction_rigid) * inputs.ts_inflow,
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (inputs.ts_inflow + i_number_shifted, True),
            ),
        )

        total_base_inflow = Index("total base vehicle flow", inputs.ts_inflow.sum())
        total_modified_inflow = Index("total modified vehicle inflow", modified_inflow.sum())

        number_paying = Index(
            "paying vehicles",
            graph.piecewise(
                (
                    fraction_rigid * inputs.ts_inflow,
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0, True),
            ),
        )

        total_paying = Index("total vehicles paying", number_paying.sum())

        modified_starting = Index(
            "modified starting",
            inputs.ts_starting + modified_inflow * (inputs.i_b_starting_modified_factor - 1),
        )

        i_total_anticipated = Index("total vehicles anticipated", i_number_anticipated.sum())
        i_total_postponed = Index("total vehicles postponed", i_number_postponed.sum())
        total_shifted = Index("total vehicles shifted", i_total_anticipated + i_total_postponed)

        # TODO: fix, compute real value!
        total_paid = Index("total paid fees", total_paying * avg_cost)

        super().__init__(
            "Inflow",
            inputs=inputs,
            outputs=Outputs(
                modified_inflow=modified_inflow,
                modified_starting=modified_starting,
                total_base_inflow=total_base_inflow,
                total_modified_inflow=total_modified_inflow,
                fraction_rigid=fraction_rigid,
                modified_euro_class_split=modified_euro_class_split,
                number_paying=number_paying,
                total_paying=total_paying,
                avg_cost=avg_cost,
                total_paid=total_paid,
                total_shifted=total_shifted,
            ),
            expose=Expose(
                i_fraction_rigid_euro=i_fraction_rigid_euro,
                i_delta_from_start=i_delta_from_start,
                i_fraction_anticipating=i_fraction_anticipating,
                i_number_anticipating=i_number_anticipating,
                i_delta_to_end=i_delta_to_end,
                i_fraction_postponing=i_fraction_postponing,
                i_number_postponing=i_number_postponing,
                i_total_anticipating=i_total_anticipating,
                i_total_postponing=i_total_postponing,
                i_delta_before_start=i_delta_before_start,
                i_number_anticipated=i_number_anticipated,
                i_delta_after_end=i_delta_after_end,
                i_number_postponed=i_number_postponed,
                i_number_shifted=i_number_shifted,
                i_total_anticipated=i_total_anticipated,
                i_total_postponed=i_total_postponed,
            ),
        )


class TrafficModel(Model):
    """Sub-model that computes both baseline and modified circulating traffic.

    Receives raw inflow/starting timeseries and the policy-modified versions
    from :class:`InflowModel`; produces the steady-state traffic for both
    scenarios together with ratio indexes.
    """

    @dataclass
    class Inputs:
        """Inputs of :class:`TrafficModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: Index
        modified_starting: Index

    @dataclass
    class Outputs:
        """Outputs of :class:`TrafficModel`."""

        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        total_modified_traffic: Index
        inflow_ratio: Index
        starting_ratio: Index
        traffic_ratio: Index

    def __init__(
        self,
        ts_inflow: TimeseriesIndex,
        ts_starting: TimeseriesIndex,
        modified_inflow: Index,
        modified_starting: Index,
    ) -> None:
        Inputs = TrafficModel.Inputs
        Outputs = TrafficModel.Outputs

        inputs = Inputs(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            modified_inflow=modified_inflow,
            modified_starting=modified_starting,
        )

        traffic = TimeseriesIndex(
            "reference traffic",
            graph.function_call("ts_solve", inputs.ts_inflow + inputs.ts_starting),
        )
        modified_traffic = TimeseriesIndex(
            "modified traffic",
            graph.function_call("ts_solve", inputs.modified_inflow + inputs.modified_starting),
        )
        total_modified_traffic = Index("total modified traffic", modified_traffic.sum())
        inflow_ratio = Index("ratio between modified flow and base flow", inputs.ts_inflow / inputs.modified_inflow)
        starting_ratio = Index(
            "ratio between modified starting and base starting",
            inputs.ts_starting / inputs.modified_starting,
        )
        traffic_ratio = Index(
            "ratio between modified traffic and base traffic",
            traffic / modified_traffic,
        )

        super().__init__(
            "Traffic",
            inputs=inputs,
            outputs=Outputs(
                traffic=traffic,
                modified_traffic=modified_traffic,
                total_modified_traffic=total_modified_traffic,
                inflow_ratio=inflow_ratio,
                starting_ratio=starting_ratio,
                traffic_ratio=traffic_ratio,
            ),
        )


class EmissionsModel(Model):
    """Sub-model that computes both baseline and modified emissions.

    Computes the fleet-average emission factor internally from the fixed
    euro-class data; uses the modified euro-class split from :class:`InflowModel`
    to derive the modified average emissions.
    """

    @dataclass
    class Inputs:
        """Inputs of :class:`EmissionsModel`."""

        ts: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index
        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        modified_euro_class_split: list[Index]

    @dataclass
    class Outputs:
        """Outputs of :class:`EmissionsModel`."""

        average_emissions: Index
        emissions: TimeseriesIndex
        modified_emissions: Index
        total_emissions: Index
        total_modified_emissions: Index

    def __init__(
        self,
        ts: TimeseriesIndex,
        i_p_start_time: Index,
        i_p_end_time: Index,
        traffic: TimeseriesIndex,
        modified_traffic: TimeseriesIndex,
        modified_euro_class_split: list[Index],
    ) -> None:
        Inputs = EmissionsModel.Inputs
        Outputs = EmissionsModel.Outputs

        inputs = Inputs(
            ts=ts,
            i_p_start_time=i_p_start_time,
            i_p_end_time=i_p_end_time,
            traffic=traffic,
            modified_traffic=modified_traffic,
            modified_euro_class_split=modified_euro_class_split,
        )

        average_emissions = Index(
            "average emissions (per vehicle, per km)",
            euro_class_emission["euro_0"] * euro_class_split["euro_0"]
            + euro_class_emission["euro_1"] * euro_class_split["euro_1"]
            + euro_class_emission["euro_2"] * euro_class_split["euro_2"]
            + euro_class_emission["euro_3"] * euro_class_split["euro_3"]
            + euro_class_emission["euro_4"] * euro_class_split["euro_4"]
            + euro_class_emission["euro_5"] * euro_class_split["euro_5"]
            + euro_class_emission["euro_6"] * euro_class_split["euro_6"],
        )

        i_modified_average_emissions = Index(
            "modified average emissions (per vehicle, per km)",
            euro_class_emission["euro_0"] * inputs.modified_euro_class_split[0]
            + euro_class_emission["euro_1"] * inputs.modified_euro_class_split[1]
            + euro_class_emission["euro_2"] * inputs.modified_euro_class_split[2]
            + euro_class_emission["euro_3"] * inputs.modified_euro_class_split[3]
            + euro_class_emission["euro_4"] * inputs.modified_euro_class_split[4]
            + euro_class_emission["euro_5"] * inputs.modified_euro_class_split[5]
            + euro_class_emission["euro_6"] * inputs.modified_euro_class_split[6],
        )

        # TODO: improve - at the moment, the conversion factor is 2.5 km per 5 minutes
        emissions = TimeseriesIndex(
            "emissions",
            2.5 * average_emissions * inputs.traffic,
        )

        # TODO: The average emissions is probably different outside regulated hours
        #  (shifted cars' emissions are proportional to shifted cars' euro level mix)
        modified_emissions = Index(
            "modified emissions",
            graph.piecewise(
                (
                    2.5 * i_modified_average_emissions * inputs.modified_traffic,
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (2.5 * average_emissions * inputs.modified_traffic, True),
            ),
        )

        total_emissions = Index("total emissions", emissions.sum())
        total_modified_emissions = Index("total modified emissions", modified_emissions.sum())

        super().__init__(
            "Emissions",
            inputs=inputs,
            outputs=Outputs(
                average_emissions=average_emissions,
                emissions=emissions,
                modified_emissions=modified_emissions,
                total_emissions=total_emissions,
                total_modified_emissions=total_modified_emissions,
            ),
        )


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class BolognaModel(Model):
    """Root model for the Bologna mobility example.

    Composes three sub-models:

    * :class:`InflowModel` — policy-modified inflow and payment statistics.
    * :class:`TrafficModel` — baseline and modified circulating traffic.
    * :class:`EmissionsModel` — baseline and modified emissions.

    KPI outputs are declared on ``outputs``; timeseries used by plotting
    helpers are surfaced via ``expose``.
    """

    @dataclass
    class Outputs:
        """KPI outputs of :class:`BolognaModel`."""

        total_base_inflow: Index
        total_modified_inflow: Index
        total_shifted: Index
        total_paying: Index
        avg_cost: Index
        total_payed: Index
        total_emissions: Index
        total_modified_emissions: Index

    @dataclass
    class Expose:
        """Inspectable timeseries and behavioral parameters.

        The timeseries fields are used by plotting helpers; ``i_b_p50_cost``
        is surfaced here so that :class:`~dt_model.simulation.ensemble.DistributionEnsemble`
        can discover and sample it.
        """

        ts_inflow: TimeseriesIndex
        modified_inflow: Index
        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        emissions: TimeseriesIndex
        modified_emissions: Index
        i_b_p50_cost: DistributionIndex

    def __init__(self) -> None:
        Outputs = BolognaModel.Outputs
        Expose = BolognaModel.Expose

        ts = TimeseriesIndex(
            "time range",
            np.array(
                [
                    (t - pd.Timestamp("00:00:00")).total_seconds()
                    for t in pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")
                ]
            ),
        )
        ts_inflow = TimeseriesIndex("inflow", vehicle_inflow)
        ts_starting = TimeseriesIndex("staring", vehicle_starting)

        i_p_start_time = Index("start time", (pd.Timestamp("07:30:00") - pd.Timestamp("00:00:00")).total_seconds())
        i_p_end_time = Index("end time", (pd.Timestamp("19:30:00") - pd.Timestamp("00:00:00")).total_seconds())
        i_p_cost = [Index(f"cost euro {e}", 5.00 - e * 0.25) for e in range(7)]
        i_p_fraction_exempted = Index("exempted vehicles %", 0.15)

        i_b_p50_cost = DistributionIndex("cost 50% threshold", stats.uniform, {"loc": 4.00, "scale": 7.00})
        i_b_p50_anticipating = Index("anticipation 50% likelihood", 0.5)
        i_b_p50_anticipation = Index("anticipation distribution 50% threshold", 0.25)
        i_b_p50_postponing = Index("postponement 50% likelihood", 0.8)
        i_b_p50_postponement = Index("postponement distribution 50% threshold", 0.50)
        i_b_starting_modified_factor = Index("starting modified factor", 1.00)

        _inflow = InflowModel(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            ts=ts,
            i_p_start_time=i_p_start_time,
            i_p_end_time=i_p_end_time,
            i_p_cost=i_p_cost,
            i_p_fraction_exempted=i_p_fraction_exempted,
            i_b_p50_cost=i_b_p50_cost,
            i_b_p50_anticipating=i_b_p50_anticipating,
            i_b_p50_anticipation=i_b_p50_anticipation,
            i_b_p50_postponing=i_b_p50_postponing,
            i_b_p50_postponement=i_b_p50_postponement,
            i_b_starting_modified_factor=i_b_starting_modified_factor,
        )

        _traffic = TrafficModel(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            modified_inflow=_inflow.outputs.modified_inflow,
            modified_starting=_inflow.outputs.modified_starting,
        )

        _emissions = EmissionsModel(
            ts=ts,
            i_p_start_time=i_p_start_time,
            i_p_end_time=i_p_end_time,
            traffic=_traffic.outputs.traffic,
            modified_traffic=_traffic.outputs.modified_traffic,
            modified_euro_class_split=_inflow.outputs.modified_euro_class_split,
        )

        super().__init__(
            "Bologna mobility",
            outputs=Outputs(
                total_base_inflow=_inflow.outputs.total_base_inflow,
                total_modified_inflow=_inflow.outputs.total_modified_inflow,
                total_shifted=_inflow.outputs.total_shifted,
                total_paying=_inflow.outputs.total_paying,
                avg_cost=_inflow.outputs.avg_cost,
                total_payed=_inflow.outputs.total_paid,
                total_emissions=_emissions.outputs.total_emissions,
                total_modified_emissions=_emissions.outputs.total_modified_emissions,
            ),
            expose=Expose(
                ts_inflow=ts_inflow,
                modified_inflow=_inflow.outputs.modified_inflow,
                traffic=_traffic.outputs.traffic,
                modified_traffic=_traffic.outputs.modified_traffic,
                emissions=_emissions.outputs.emissions,
                modified_emissions=_emissions.outputs.modified_emissions,
                i_b_p50_cost=i_b_p50_cost,
            ),
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def evaluate(model: BolognaModel, size: int = 1) -> EvaluationResult:
    """Evaluate *model* over an ensemble of *size* samples.

    Draws *size* samples from each distribution-backed abstract index via
    :class:`~dt_model.simulation.ensemble.DistributionEnsemble` and runs the
    standard :class:`~dt_model.simulation.evaluation.Evaluation` engine.

    Parameters
    ----------
    model:
        A :class:`BolognaModel` instance.
    size:
        Number of ensemble samples to draw.

    Returns
    -------
    EvaluationResult
        Use ``result[idx]`` for the raw ``(S, 1)`` or ``(S, T)`` array and
        ``result.marginalize(idx)`` for the weighted mean.
    """
    ensemble = DistributionEnsemble(model, size)
    return Evaluation(model).evaluate(
        ensemble,
        functions={"ts_solve": executor.LambdaAdapter(_ts_solve)},
    )


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


def compute_kpis(m: BolognaModel, evals: EvaluationResult) -> dict:
    """Compute the KPIs for the mobility example.

    Parameters
    ----------
    m:
        A :class:`BolognaModel` instance.
    evals:
        The subs dict returned by :func:`evaluate`.

    Returns
    -------
    dict
        Mapping of KPI label strings to integer values.
    """
    return {
        "Base inflow [veh/day]": int(evals[m.outputs.total_base_inflow].mean()),
        "Modified inflow [veh/day]": int(evals[m.outputs.total_modified_inflow].mean()),
        "Shifted inflow [veh/day]": int(evals[m.outputs.total_shifted].mean()),
        "Paying inflow [veh/day]": (
            int(evals[m.outputs.total_paying].mean()) if evals[m.outputs.avg_cost].mean() > 0 else 0
        ),
        "Collected fees [€/day]": int(evals[m.outputs.total_payed].mean()),
        "Emissions [NOx gr/day]": int(evals[m.outputs.total_modified_emissions].mean()),
        "Modified emissions [NOx gr/day]": int(evals[m.outputs.total_emissions].mean())
        - int(evals[m.outputs.total_modified_emissions].mean()),
    }


def roundup(val):
    """Compute a rounded-up approximation of `val`."""
    v = val * 1.4
    s = math.floor(math.log10(v * 1.3))
    return round(v / 10**s) * 10**s


if __name__ == "__main__":
    m = BolognaModel()
    subs = evaluate(m, 20)

    fig = plot_field_graph(
        subs[m.expose.modified_inflow],
        horizontal_label="Time",
        vertical_label="Flow (vehicles/hour)",
        vertical_size=1600,
        vertical_formatter=mticker.FuncFormatter(lambda x, _: f"{int(x * 12)}"),
        reference_line=subs[m.expose.ts_inflow][0],
    )
    fig.savefig("bologna_inflow.png", dpi=150)
    plt.close(fig)

    fig = plot_field_graph(
        subs[m.expose.modified_traffic],
        horizontal_label="Time",
        vertical_label="Traffic (circulating vehicles)",
        vertical_size=15000,
        reference_line=subs[m.expose.traffic][0],
    )
    fig.savefig("bologna_traffic.png", dpi=150)
    plt.close(fig)

    fig = plot_field_graph(
        subs[m.expose.modified_emissions],
        horizontal_label="Time",
        vertical_label="Emissions (NOx gr/h)",
        vertical_size=4000,
        vertical_formatter=mticker.FuncFormatter(lambda x, _: f"{int(x * 12)}"),
        reference_line=subs[m.expose.emissions][0],
    )
    fig.savefig("bologna_emissions.png", dpi=150)
    plt.close(fig)

    for k, v in compute_kpis(m, subs).items():
        print(f"{k} - {v:,}")
