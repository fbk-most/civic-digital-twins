# SPDX-License-Identifier: Apache-2.0
"""Bologna mobility model: sub-models, BolognaOutput, and BolognaEvaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from civic_digital_twins.dt_model import (
    ConstIndex,
    ConstTimeseriesIndex,
    DistributionIndex,
    EvaluationResult,
    Functor,
    Index,
    Model,
    NumpyBackend,
    Scenario,
    TimeseriesIndex,
    define,
    expose,
    functions,
    graph,
    inputs,
    outputs,
)
from civic_digital_twins.dt_model.simulation.runner import (
    ModelEvaluator,
    ModelOutput,
)

_LN2: float = math.log(2)
"""Natural logarithm of 2 (≈ 0.6931). Normalisation constant in half-life decay formulas."""

_LN_HALF: float = -_LN2
"""Natural logarithm of 0.5 (= −ln 2). Exponent base for half-life decay: exp(Δt / p50 · ln½) = ½^(Δt/p50)."""

try:
    from .bologna_data import euro_class_emission, euro_class_split, vehicle_inflow, vehicle_starting
except ImportError:
    from bologna_data import euro_class_emission, euro_class_split, vehicle_inflow, vehicle_starting


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


@define("Inflow")
class InflowModel(Model):
    """Sub-model that computes modified vehicle inflow under a pricing policy.

    Computes per-euro-class rigidity fractions, anticipating/postponing
    behaviour, the modified inflow and starting timeseries, and payment
    statistics.
    """

    @inputs
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

    @outputs
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

    @expose
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

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        """Compute modified inflow, payment stats, and exposed intermediates."""
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

        return (
            InflowModel.Outputs(
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
            InflowModel.Expose(
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


@define("Traffic")
class TrafficModel(Model):
    """Sub-model that computes both baseline and modified circulating traffic.

    Receives raw inflow/starting timeseries and the policy-modified versions
    from :class:`InflowModel`; produces the steady-state traffic for both
    scenarios together with ratio indexes.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`TrafficModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: Index
        modified_starting: Index

    @outputs
    class Outputs:
        """Outputs of :class:`TrafficModel`."""

        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        total_modified_traffic: Index
        inflow_ratio: Index
        starting_ratio: Index
        traffic_ratio: Index

    @functions
    class Functions:
        """Functions required by :class:`TrafficModel`."""

        ts_solve: Functor

    def compute(self, inputs: Inputs, *, fns: Functions) -> Outputs:
        """Compute steady-state traffic for baseline and modified scenarios."""
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
        return TrafficModel.Outputs(
            traffic=traffic,
            modified_traffic=modified_traffic,
            total_modified_traffic=total_modified_traffic,
            inflow_ratio=inflow_ratio,
            starting_ratio=starting_ratio,
            traffic_ratio=traffic_ratio,
        )


@define("Emissions")
class EmissionsModel(Model):
    """Sub-model that computes both baseline and modified emissions.

    Computes the fleet-average emission factor internally from the fixed
    euro-class data; uses the modified euro-class split from :class:`InflowModel`
    to derive the modified average emissions.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`EmissionsModel`."""

        ts: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index
        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        modified_euro_class_split: list[Index]

    @outputs
    class Outputs:
        """Outputs of :class:`EmissionsModel`."""

        average_emissions: Index
        emissions: TimeseriesIndex
        modified_emissions: Index
        total_emissions: Index
        total_modified_emissions: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute baseline and modified vehicle emissions."""
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

        return EmissionsModel.Outputs(
            average_emissions=average_emissions,
            emissions=emissions,
            modified_emissions=modified_emissions,
            total_emissions=total_emissions,
            total_modified_emissions=total_modified_emissions,
        )


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------


class BolognaModel(Model, legacy=True):
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

    @inputs
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

    @outputs
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

    @expose
    class Expose:
        """Inspectable timeseries used by plotting helpers."""

        ts_inflow: TimeseriesIndex
        modified_inflow: Index
        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        emissions: TimeseriesIndex
        modified_emissions: Index

    @functions
    class Functions:
        """Functions required by :class:`BolognaModel`."""

        ts_solve: Functor

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
        functions: BolognaModel.Functions | None = None,
    ) -> None:
        fns = functions or BolognaModel.Functions(ts_solve=NumpyBackend.adapt(_ts_solve))
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

        _inflow = InflowModel(inputs=InflowModel.Inputs(  # type: ignore[call-arg]
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
        ))

        _traffic = TrafficModel(  # type: ignore[call-arg]
            inputs=TrafficModel.Inputs(
                ts_inflow=ts_inflow,
                ts_starting=ts_starting,
                modified_inflow=_inflow.outputs.modified_inflow,
                modified_starting=_inflow.outputs.modified_starting,
            ),
            fns=TrafficModel.Functions(ts_solve=fns.ts_solve),
        )

        _emissions = EmissionsModel(inputs=EmissionsModel.Inputs(  # type: ignore[call-arg]
            ts=ts,
            i_p_start_time=i_p_start_time,
            i_p_end_time=i_p_end_time,
            traffic=_traffic.outputs.traffic,
            modified_traffic=_traffic.outputs.modified_traffic,
            modified_euro_class_split=_inflow.outputs.modified_euro_class_split,
        ))

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
            functions=fns,
        )


# ---------------------------------------------------------------------------
# KPI helper
# ---------------------------------------------------------------------------


def compute_kpis(m: BolognaModel, result: EvaluationResult) -> dict:
    """Compute the KPIs for the mobility example.

    Parameters
    ----------
    m:
        A :class:`BolognaModel` instance.
    result:
        The :class:`~dt_model.simulation.evaluation.EvaluationResult` returned
        by :meth:`BolognaEvaluator.evaluate`.

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


# ---------------------------------------------------------------------------
# Runner protocol: BolognaOutput and BolognaEvaluator
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class BolognaOutput(ModelOutput):
    """Evaluation output for the Bologna mobility model.

    Carries the post-processed KPIs, the expected-value 1-D timeseries for
    the six ``expose`` indexes, the raw ensemble field arrays for the three
    modified-quantity indexes, and an optional resume payload that allows
    :meth:`BolognaEvaluator.resume` to extend the ensemble in a later session.

    Attributes
    ----------
    kpis : dict[str, int]
        Scalar KPI values keyed by human-readable label.
    timeseries : dict[str, numpy.ndarray]
        Expected-value 1-D arrays for the six ``expose`` indexes.
    fields : dict[str, numpy.ndarray]
        Raw ensemble field arrays of shape ``(S, T)`` for the three
        modified-quantity indexes.
    """

    kpis: dict[str, int]
    timeseries: dict[str, np.ndarray]
    fields: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        """Initialise the :class:`ModelOutput` base after dataclass field assignment."""
        super().__init__()


class BolognaEvaluator(ModelEvaluator[BolognaModel, BolognaOutput]):
    """Scenario evaluator for the Bologna mobility model.

    Implements the :class:`~dt_model.simulation.runner.ModelEvaluator` protocol
    for :class:`~mobility_bologna.bologna_model.BolognaModel`, covering
    blocking evaluation, engine-level async evaluation, resumability, and
    model structure introspection.

    Parameters
    ----------
    model : BolognaModel
        A :class:`~mobility_bologna.bologna_model.BolognaModel` instance.
    """

    def __init__(self, model: BolognaModel) -> None:
        super().__init__(model)

    def post_process(self, scenario: Scenario, result: EvaluationResult) -> BolognaOutput:
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
            Populated summary output; the resume payload is attached by the
            base :meth:`evaluate` template after this method returns.
        """
        m = self._model
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
        return BolognaOutput(kpis=kpis, timeseries=timeseries, fields=fields)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def input_schema(self) -> dict[str, dict[str, Any]]:
        """Return a schema dict for the tunable policy and behavioural indexes.

        Covers all fields declared on :class:`~mobility_bologna.bologna_model.BolognaModel.Inputs`.
        Each scalar :class:`~dt_model.Index` maps to ``{"type": "scalar"}``; each
        :class:`~dt_model.DistributionIndex` maps to ``{"type": "distribution"}``.
        List-valued fields (``i_p_cost``) produce one entry per element.

        Returns
        -------
        dict[str, dict[str, Any]]
            Index name to metadata dict.
        """
        m = self._model
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
