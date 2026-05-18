"""Example of Bologna mobility scenario.

We include this model into the source tree as an illustrative example.
"""

# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import matplotlib

matplotlib.use("Agg")  # must be called before any other matplotlib sub-imports

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from civic_digital_twins.dt_model import (
    ConstIndex,
    ConstTimeseriesIndex,
    DistributionEnsemble,
    DistributionIndex,
    Evaluation,
    EvaluationResult,
    Index,
    Model,
    NumpyBackend,
    Scenario,
    TimeseriesIndex,
    graph,
)
from civic_digital_twins.dt_model.simulation.runner import (
    EvaluationConfig,
    ModelEvaluator,
    ModelOutput,
    ModelRunHandle,
    ResumeState,
    _decode_array,
    _decode_result,
    _encode_array,
    _encode_result,
    _get_dt_model_version,
)

_LN2: float = math.log(2)
"""Natural logarithm of 2 (≈ 0.6931). Normalisation constant in half-life decay formulas."""

_LN_HALF: float = -_LN2
"""Natural logarithm of 0.5 (= −ln 2). Exponent base for half-life decay: exp(Δt / p50 · ln½) = ½^(Δt/p50)."""

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
                (1 - inputs.i_p_fraction_exempted) * graph.exp(inputs.i_p_cost[e] / inputs.i_b_p50_cost * _LN_HALF),
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
            graph.exp(i_delta_from_start / inputs.i_b_p50_anticipating * _LN_HALF)
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
            graph.exp(i_delta_to_end / inputs.i_b_p50_postponing * _LN_HALF)
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
            graph.exp(i_delta_before_start / inputs.i_b_p50_anticipation * _LN_HALF)
            / inputs.i_b_p50_anticipation
            * _LN2
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
            graph.exp(i_delta_after_end / inputs.i_b_p50_postponement * _LN_HALF)
            / inputs.i_b_p50_postponement
            * _LN2
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

        average_emissions = ConstIndex(
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

    All policy parameters (``i_p_*``) and behavioural parameters (``i_b_*``)
    are declared in ``Inputs`` and can be overridden at construction time.
    KPI outputs are declared on ``outputs``; timeseries used by plotting
    helpers are surfaced via ``expose``.
    """

    @dataclass
    class Inputs:
        """Policy and behavioural parameters of :class:`BolognaModel`."""

        # Policy parameters
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_cost: list[Index]
        i_p_fraction_exempted: Index
        # Behavioural parameters
        i_b_p50_cost: DistributionIndex
        i_b_p50_anticipating: Index
        i_b_p50_anticipation: Index
        i_b_p50_postponing: Index
        i_b_p50_postponement: Index
        i_b_starting_modified_factor: Index

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
        """Inspectable timeseries used by plotting helpers."""

        ts_inflow: TimeseriesIndex
        modified_inflow: Index
        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        emissions: TimeseriesIndex
        modified_emissions: Index

    @classmethod
    def default_inputs(cls) -> dict:
        """Return the reference-scenario input parameters as a keyword-argument dict.

        Pass directly to :class:`BolognaModel` or override individual entries::

            m = BolognaModel(**BolognaModel.default_inputs())
            m_alt = BolognaModel(**{**BolognaModel.default_inputs(), "i_p_cost": [...]})
        """
        return {
            "i_p_start_time": Index(
                "start time", (pd.Timestamp("07:30:00") - pd.Timestamp("00:00:00")).total_seconds()
            ),
            "i_p_end_time": Index("end time", (pd.Timestamp("19:30:00") - pd.Timestamp("00:00:00")).total_seconds()),
            "i_p_cost": [Index(f"cost euro {e}", 5.00 - e * 0.25) for e in range(7)],
            "i_p_fraction_exempted": Index("exempted vehicles %", 0.15),
            "i_b_p50_cost": DistributionIndex("cost 50% threshold", stats.uniform, {"loc": 4.00, "scale": 7.00}),
            "i_b_p50_anticipating": Index("anticipation 50% likelihood", 0.5),
            "i_b_p50_anticipation": Index("anticipation distribution 50% threshold", 0.25),
            "i_b_p50_postponing": Index("postponement 50% likelihood", 0.8),
            "i_b_p50_postponement": Index("postponement distribution 50% threshold", 0.50),
            "i_b_starting_modified_factor": Index("starting modified factor", 1.00),
        }

    def __init__(
        self,
        *,
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
        Inputs = BolognaModel.Inputs
        Outputs = BolognaModel.Outputs
        Expose = BolognaModel.Expose

        inputs = Inputs(
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

        ts = ConstTimeseriesIndex(
            "time range",
            np.array(
                [
                    (t - pd.Timestamp("00:00:00")).total_seconds()
                    for t in pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")
                ]
            ),
        )
        ts_inflow = ConstTimeseriesIndex("inflow", vehicle_inflow)
        ts_starting = ConstTimeseriesIndex("staring", vehicle_starting)

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
            inputs=inputs,
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
        ``result.expected_value(idx)`` for the weighted mean.
    """
    ensemble = DistributionEnsemble(model, size)
    return Evaluation(model).evaluate(
        ensemble=ensemble,
        functions={"ts_solve": NumpyBackend.adapt(_ts_solve)},
        backend=NumpyBackend,
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


def compute_kpis(m: BolognaModel, result: EvaluationResult) -> dict:
    """Compute the KPIs for the mobility example.

    Parameters
    ----------
    m:
        A :class:`BolognaModel` instance.
    result:
        The :class:`~dt_model.simulation.evaluation.EvaluationResult` returned
        by :func:`evaluate`.

    Returns
    -------
    dict
        Mapping of KPI label strings to integer values.
    """
    return {
        "Base inflow [veh/day]": int(result.expected_value(m.outputs.total_base_inflow)),
        "Modified inflow [veh/day]": int(result.expected_value(m.outputs.total_modified_inflow)),
        "Shifted inflow [veh/day]": int(result.expected_value(m.outputs.total_shifted)),
        "Paying inflow [veh/day]": (
            int(result.expected_value(m.outputs.total_paying)) if result.expected_value(m.outputs.avg_cost) > 0 else 0
        ),
        "Collected fees [€/day]": int(result.expected_value(m.outputs.total_payed)),
        "Emissions [NOx gr/day]": int(result.expected_value(m.outputs.total_modified_emissions)),
        "Modified emissions [NOx gr/day]": int(result.expected_value(m.outputs.total_emissions))
        - int(result.expected_value(m.outputs.total_modified_emissions)),
    }


def roundup(val):
    """Compute a rounded-up approximation of `val`."""
    v = val * 1.4
    s = math.floor(math.log10(v * 1.3))
    return round(v / 10**s) * 10**s


def _save_scenario_plots(label: str, m: BolognaModel, output: "BolognaOutput", out: Path) -> None:
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


class BolognaOutput(ModelOutput):
    """Evaluation output for the Bologna mobility model.

    Carries the post-processed KPIs, the expected-value 1-D timeseries for
    the six ``expose`` indexes, the raw ensemble field arrays for the three
    modified-quantity indexes, and an optional resume payload that allows
    :meth:`BolognaEvaluator.resume` to extend the ensemble in a later session.

    Parameters
    ----------
    kpis : dict[str, int]
        Scalar KPI values keyed by human-readable label, produced by
        :func:`~mobility_bologna.mobility_bologna.compute_kpis`.
    timeseries : dict[str, numpy.ndarray]
        Expected-value 1-D arrays for the six ``expose`` indexes:
        ``"ts_inflow"``, ``"modified_inflow"``, ``"traffic"``,
        ``"modified_traffic"``, ``"emissions"``, ``"modified_emissions"``.
    fields : dict[str, numpy.ndarray]
        Raw ensemble field arrays of shape ``(S, T)`` for the three
        modified-quantity indexes: ``"modified_inflow"``,
        ``"modified_traffic"``, ``"modified_emissions"``.
        These are used by :func:`plot_field_graph` to render the full
        stochastic distribution rather than only the expected value.
    serialized_resume : dict or None, optional
        Pre-encoded resume payload from :func:`_encode_result`.  When
        provided the output is immediately marked resumable
        (``is_resumable == True``).
    """

    def __init__(
        self,
        kpis: dict[str, int],
        timeseries: dict[str, np.ndarray],
        fields: dict[str, np.ndarray],
        *,
        serialized_resume: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.kpis = kpis
        self.timeseries = timeseries
        self.fields = fields
        self._serialized_resume = serialized_resume
        if serialized_resume is not None:
            self._is_resumable = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise the output to a JSON-compatible dict.

        Always includes ``"dt_model_version"``, ``"kpis"``, and
        ``"timeseries"`` (arrays encoded via :func:`_encode_array`).
        Also includes ``"_resume"`` when a resume payload is present.

        Returns
        -------
        dict[str, Any]
            Serialised output dict.
        """
        data: dict[str, Any] = {
            "dt_model_version": _get_dt_model_version(),
            "kpis": self.kpis,
            "timeseries": {name: _encode_array(arr) for name, arr in self.timeseries.items()},
            "fields": {name: _encode_array(arr) for name, arr in self.fields.items()},
        }
        if self._serialized_resume is not None:
            data["_resume"] = self._serialized_resume
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct a :class:`BolognaOutput` from a serialised dict.

        Always reconstructs ``kpis`` and ``timeseries``.  If ``"_resume"``
        is present the resume payload is stored and the output is marked
        resumable.

        Parameters
        ----------
        data : dict[str, Any]
            Dict previously produced by :meth:`to_dict`.

        Returns
        -------
        BolognaOutput
            Reconstructed instance.
        """
        obj: BolognaOutput = cls.__new__(cls)
        ModelOutput.__init__(obj)
        obj.kpis = dict(data["kpis"])
        obj.timeseries = {name: _decode_array(encoded) for name, encoded in data["timeseries"].items()}
        obj.fields = {name: _decode_array(encoded) for name, encoded in data.get("fields", {}).items()}
        obj._serialized_resume = None
        if "_resume" in data:
            obj._serialized_resume = data["_resume"]
            obj._is_resumable = True
        return obj


class BolognaEvaluator(ModelEvaluator[BolognaOutput]):
    """Scenario evaluator for the Bologna mobility model.

    Implements the :class:`~dt_model.simulation.runner.ModelEvaluator` protocol
    for :class:`~mobility_bologna.mobility_bologna.BolognaModel`, covering
    blocking evaluation, engine-level async evaluation, resumability, and
    model structure introspection.

    Parameters
    ----------
    model : BolognaModel
        A :class:`~mobility_bologna.mobility_bologna.BolognaModel` instance.
    """

    def __init__(self, model: BolognaModel) -> None:
        super().__init__(model)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _post_process(self, scenario: Scenario, result: Any) -> BolognaOutput:
        """Build a :class:`BolognaOutput` from a raw :class:`~simulation.evaluation.EvaluationResult`.

        Parameters
        ----------
        scenario : Scenario
            The scenario that was evaluated.
        result : EvaluationResult
            The raw evaluation result.

        Returns
        -------
        BolognaOutput
            Fully populated output including resume payload.
        """
        m = cast(BolognaModel, scenario.model)
        kpis = compute_kpis(m, result)
        expose = m.expose
        timeseries = {
            "ts_inflow": result.expected_value(expose.ts_inflow),
            "modified_inflow": result.expected_value(expose.modified_inflow),
            "traffic": result.expected_value(expose.traffic),
            "modified_traffic": result.expected_value(expose.modified_traffic),
            "emissions": result.expected_value(expose.emissions),
            "modified_emissions": result.expected_value(expose.modified_emissions),
        }
        fields = {
            "modified_inflow": result[expose.modified_inflow],
            "modified_traffic": result[expose.modified_traffic],
            "modified_emissions": result[expose.modified_emissions],
        }
        encoded = _encode_result(result, scenario.model.indexes)
        return BolognaOutput(kpis=kpis, timeseries=timeseries, fields=fields, serialized_resume=encoded)

    # ------------------------------------------------------------------
    # ModelEvaluator abstract interface
    # ------------------------------------------------------------------

    def evaluate(self, scenario: Scenario, config: EvaluationConfig) -> BolognaOutput:
        """Run a blocking evaluation and return a :class:`BolognaOutput`.

        Parameters
        ----------
        scenario : Scenario
            The scenario to evaluate.
        config : EvaluationConfig
            Evaluation parameters; ``config.ensemble_size`` controls the
            number of Monte Carlo samples.

        Returns
        -------
        BolognaOutput
            Post-processed output with KPIs, timeseries, and resume payload.
        """
        ensemble = DistributionEnsemble(scenario, config.ensemble_size)
        result = Evaluation(scenario).evaluate(
            ensemble=ensemble,
            functions={"ts_solve": NumpyBackend.adapt(_ts_solve)},
            backend=NumpyBackend,
        )
        return self._post_process(scenario, result)

    def run_async(self, scenario: Scenario, config: EvaluationConfig) -> ModelRunHandle[BolognaOutput]:
        """Submit an engine-level async evaluation and return a handle immediately.

        Uses :meth:`~simulation.evaluation.Evaluation.submit_evaluate` to run
        the evaluation on a background thread.  The returned
        :class:`~simulation.runner.ModelRunHandle` wraps the future and a
        post-processor lambda.

        Parameters
        ----------
        scenario : Scenario
            The scenario to evaluate.
        config : EvaluationConfig
            Evaluation parameters.

        Returns
        -------
        ModelRunHandle[BolognaOutput]
            Handle whose :meth:`~simulation.runner.ModelRunHandle.get` returns
            the :class:`BolognaOutput` once the evaluation completes.
        """
        async_handle = Evaluation(scenario).submit_evaluate(
            config.ensemble_size,
            functions={"ts_solve": NumpyBackend.adapt(_ts_solve)},
            backend=NumpyBackend,
        )
        return ModelRunHandle(
            future=async_handle.future,
            post_process=lambda result: self._post_process(scenario, result),
        )

    def structure(self) -> dict[str, dict[str, Any]]:
        """Return a schema dict for the tunable policy and behavioural indexes.

        Covers all fields declared on :class:`~mobility_bologna.mobility_bologna.BolognaModel.Inputs`.
        Each scalar :class:`~dt_model.Index` maps to ``{"type": "scalar"}``; each
        :class:`~dt_model.DistributionIndex` maps to ``{"type": "distribution"}``.
        List-valued fields (``i_p_cost``) produce one entry per element.

        Returns
        -------
        dict[str, dict[str, Any]]
            Index name to metadata dict.
        """
        m = cast(BolognaModel, self._model)
        inputs = m.inputs
        result: dict[str, dict[str, Any]] = {}

        def _add(idx: Index | DistributionIndex) -> None:
            entry_type = "distribution" if isinstance(idx, DistributionIndex) else "scalar"
            result[idx.name] = {"type": entry_type}

        _add(inputs.i_p_start_time)
        _add(inputs.i_p_end_time)
        for cost_idx in inputs.i_p_cost:
            _add(cost_idx)
        _add(inputs.i_p_fraction_exempted)
        _add(inputs.i_b_p50_cost)
        _add(inputs.i_b_p50_anticipating)
        _add(inputs.i_b_p50_anticipation)
        _add(inputs.i_b_p50_postponing)
        _add(inputs.i_b_p50_postponement)
        _add(inputs.i_b_starting_modified_factor)
        return result

    def _extract_resume_state(self, output: BolognaOutput) -> ResumeState:
        """Extract the resume payload from a previously saved :class:`BolognaOutput`.

        Parameters
        ----------
        output : BolognaOutput
            A :class:`BolognaOutput` for which ``is_resumable`` is ``True``.

        Returns
        -------
        ResumeState
            All state needed to reconstruct an
            :class:`~simulation.handle.EvaluationHandle`.

        Raises
        ------
        AssertionError
            If ``output._serialized_resume`` is ``None``.
        """
        assert output._serialized_resume is not None, "resume payload must not be None"
        result = _decode_result(output._serialized_resume, self._model.indexes)
        return ResumeState(
            result=result,
            parameters={},
            functions={"ts_solve": NumpyBackend.adapt(_ts_solve)},
            backend=NumpyBackend,
        )


if __name__ == "__main__":
    _out = Path(__file__).parent / "output"
    _out.mkdir(exist_ok=True)

    _config = EvaluationConfig(ensemble_size=20)

    # ── Reference scenario (default parameters) ──────────────────────────────
    _m = BolognaModel(**BolognaModel.default_inputs())
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
        **{
            **BolognaModel.default_inputs(),
            "i_p_cost": [Index(f"cost euro {e}", 8.00 - e * 0.50) for e in range(7)],
        }
    )
    _evaluator_strict = BolognaEvaluator(_m_strict)
    _output_strict = _evaluator_strict.evaluate(Scenario(_m_strict), _config)
    _save_scenario_plots("strict", _m_strict, _output_strict, _out)

    print("\nStricter pricing scenario (euro_0: 8.00 €, euro_6: 5.00 €):")
    for k, v in _output_strict.kpis.items():
        print(f"  {k} - {v:,}")
