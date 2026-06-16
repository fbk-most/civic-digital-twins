"""Bologna Mobility model — ``@define``/``compute`` pattern.

Hierarchy::

    BolognaMobilityModel
    ├── BaseFlowsModel              — baseline inflow/starting timeseries
    ├── PolicyWindowModel           — policy-window time deltas
    ├── ParallelBehaviorModel       — behaviour fractions (parallel decision)
    │   └── ModelVariant("ModalShift")
    │       ├── TpmModalShiftModel  — logit PT modal-shift
    │       └── ZeroModalShiftModel — no modal shift
    ├── TimeShiftModel              — anticipators/postponers/shifted/lost counts
    │   └── FlexibleShiftFormula    — closed-form exponential redistribution
    ├── AreaModifiedFlowsModel      — modified inflow & starting
    │   └── FlexibleShiftFormula    — inside-area redistribution
    ├── AreaRevenueModel            — paying vehicles & collected fees
    ├── TrafficModel                — steady-state traffic (ts_solve)
    ├── InducedDemandModel          — congestion-relief induced demand
    │   └── ModelVariant("InducedDemandFormula")
    │       ├── NoInducedDemand     — no induced demand
    │       └── ElemReliefInducedDemand — element-relief formula
    └── EmissionsModel              — NOx emissions (fleet-average EF)
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from civic_digital_twins.dt_model import (
    DistributionEnsemble,
    DistributionIndex,
    Index,
    Model,
    ModelVariant,
    TimeseriesIndex,
    define,
    expose,
    graph,
    inputs,
    outputs,
)
from civic_digital_twins.dt_model.engine.numpybackend import executor
from civic_digital_twins.dt_model.simulation.evaluation import Evaluation
from examples.scenario_analysis.bologna_mobility_data import (
    euro_class_emission,
    euro_class_split,
    vehicle_inflow,
    vehicle_starting,
)

# Constants
I_B_PT_BASELINE = -1.28  # A-priori chosen (knowledge-based)
P_PROB_THRESHOLD = 0.005  # A-priori chosen (knowledge-based)
P_RECORD_FREQUENCY = 12
P_RECORD_HEADWAY = 1 / P_RECORD_FREQUENCY
P_DWELL_TIME_BM = 1 / 3  # A-priori chosen (knowledge-based)

# PT metrics (scalar constants)
PT_CAPILLARITY = 0.396115386983386
PT_FREQUENCY = 0.9450667478611485
PT_AVG_COST = 3.0539574852378704
PT_AVG_TIME_DIFF = -4.887494531270383

# Dict policy init values:
policy_params_init = {
    "i_p_start_time": "07:30:00",
    "i_p_end_time": "19:30:00",
    "i_p_cost": [5.00 - e * 0.25 for e in range(7)],
    "i_p_fraction_exempted": 0.15,
    "i_p_pt_frequency_modification": 0.0,
    "i_p_pt_capillarity_modification": 0.0,
    "i_p_pt_cost_modification": 0.0,
    "i_p_pt_time_modification": 0.0,
}

# Dict behaviors init values:
behaviors_params_init = {
    "i_b_p50_cost": {"loc": 4.00, "scale": 7.00},
    "i_b_p50_anticipating": 0.25,
    "i_b_p50_postponing": 0.5,
    "i_b_p50_anticipation": 0.50,
    "i_b_p50_postponement": 0.50,
    "i_b_pt_capillarity": 4.5,
    "i_b_pt_frequency": -1.45,
    "i_b_pt_cost": -0.30,
    "i_b_pt_time": -0.034,
}

# ---------------------------------------------------------------------------
# Shared graph functions (registered with the evaluation engine at evaluate())
# ---------------------------------------------------------------------------


def ts_sum(ts: np.ndarray) -> np.ndarray:
    """Sum a timeseries over the time axis."""
    return ts.sum(axis=-1, keepdims=True)


def ts_max(ts: np.ndarray) -> np.ndarray:
    """Maximum of a timeseries over the time axis."""
    return ts.max(axis=-1, keepdims=True)


def ts_solve(
    ts: np.ndarray, dwell_time: float = P_DWELL_TIME_BM, max_iter: int = 100, min_diff: float = 1e-5
) -> np.ndarray:
    """Solve steady-state traffic iteratively."""
    tau = dwell_time * P_RECORD_FREQUENCY
    series = ts.copy()
    for _ in range(max_iter):
        mu = 1 + (tau - 1)
        alfa = (mu - 1) / mu
        next_series = ts + np.roll(series, 1, axis=-1) * alfa
        if np.max(np.abs(series - next_series)) < min_diff:
            break
        series = next_series
    return series


def ts_b_choose(w_a: np.ndarray, *list_w_b: np.ndarray) -> np.ndarray:
    """Competitive probability allocation across behaviour fractions."""
    if np.all(w_a == 1):
        return np.ones_like(w_a)
    if len(list_w_b) == 0:
        return np.ones_like(w_a)
    for w_b in list_w_b:
        if np.all(w_b == 1):
            return np.zeros_like(w_a)
    p = np.zeros_like(w_a)
    for r in range(len(list_w_b) + 1):
        for indices in combinations(range(len(list_w_b)), r):
            p_tmp = w_a.copy()
            for i, w_b in enumerate(list_w_b):
                if i in indices:
                    p_tmp = p_tmp * w_b
                else:
                    p_tmp = p_tmp * (1 - w_b)
            if r > 0:
                denominator = w_a.copy()
                for idx in indices:
                    denominator = denominator + list_w_b[idx]
                weight = np.divide(w_a, denominator, out=np.zeros_like(denominator), where=denominator != 0)
                p_tmp = p_tmp * weight
            p = p + p_tmp
    return p


def ts_anticipate(
    number_anticipating: np.ndarray,
    delta_from_start: np.ndarray,
    p50_anticipating: float,  # need to be scalar
) -> np.ndarray:
    """Redistribute anticipating vehicles before the policy window."""
    # First, check the shapes of the ndarrays
    if number_anticipating.ndim != 2:  ## 2D Array (n, m)
        raise ValueError(
            f"DimensionalityError: 'number_anticipating' must be 2D (n, m), got shape {number_anticipating.shape}"
        )
    n, m = number_anticipating.shape
    if delta_from_start.ndim != 1:  ## 1D Array (m,)
        raise ValueError(f"DimensionalityError: 'delta_from_start' must be 1D (m,), got shape {delta_from_start.shape}")
    if delta_from_start.shape[0] != m:
        raise ValueError(f"DimensionalityError: 'delta_from_start' length must be {m}, got {delta_from_start.shape[0]}")
    if not (np.isscalar(p50_anticipating) or np.ndim(p50_anticipating) == 0):
        raise ValueError(
            f"DimensionalityError: 'p50_anticipating' must be a scalar, got shape {p50_anticipating.shape}"
        )

    # Then compute anticipating redistribution
    t0 = np.where(delta_from_start == 0)[0][0]
    n, tmax = number_anticipating.shape
    range1 = np.arange(0, tmax) * P_RECORD_HEADWAY
    range2 = np.arange(1, tmax + 1) * P_RECORD_HEADWAY

    v1 = np.exp(range1 / p50_anticipating * np.log(0.5)) - np.exp(range2 / p50_anticipating * np.log(0.5))
    v1 = np.where(v1 < P_PROB_THRESHOLD, 0, v1)
    number_anticipated = np.zeros_like(number_anticipating)
    for t in range(t0, tmax):
        v1_here = v1[(t - t0) :]
        v1_here_2d = v1_here[np.newaxis, :]
        v1_here_sum = ts_sum(v1_here_2d)

        if v1_here_sum[0, 0] > 0:
            v1_normalized = v1_here_2d / v1_here_sum
            for deltat in range(1, t0 + 1):
                if deltat - 1 < v1_normalized.shape[1]:
                    weight = v1_normalized[0, deltat - 1]
                    number_anticipated[:, t0 - deltat] += number_anticipating[:, t] * weight

    return number_anticipated


def ts_postpone(
    number_postponing: np.ndarray,
    delta_to_end: np.ndarray,
    p50_postponing: float,
) -> np.ndarray:
    """Redistribute postponing vehicles after the policy window."""
    # First, check the shapes of the ndarrays
    if number_postponing.ndim != 2:  ## 2D Array (n, m)
        raise ValueError(
            f"DimensionalityError: 'number_postponing' must be 2D (n, m), got shape {number_postponing.shape}"
        )
    n, m = number_postponing.shape
    if delta_to_end.ndim != 1:  ## 1D Array (m,)
        raise ValueError(f"DimensionalityError: 'delta_to_end' must be 1D (m,), got shape {delta_to_end.shape}")
    if delta_to_end.shape[0] != m:
        raise ValueError(f"DimensionalityError: 'delta_to_end' length must be {m}, got {delta_to_end.shape[0]}")
    if not (np.isscalar(p50_postponing) or np.ndim(p50_postponing) == 0):
        raise ValueError(f"DimensionalityError: 'p50_postponing' must be a scalar, got shape {p50_postponing.shape}")

    # Then compute postponement redistribution
    t0 = np.where(delta_to_end == 0)[0][0]
    n, tmax = number_postponing.shape

    range1 = np.arange(0, tmax) * P_RECORD_HEADWAY
    range2 = np.arange(1, tmax + 1) * P_RECORD_HEADWAY

    v1 = np.exp(range1 / p50_postponing * np.log(0.5)) - np.exp(range2 / p50_postponing * np.log(0.5))
    v1 = np.where(v1 < P_PROB_THRESHOLD, 0, v1)
    number_postponed = np.zeros_like(number_postponing)
    for t in range(t0, -1, -1):
        number_postponing_here = number_postponing[:, t]
        v1_here = v1[(t0 - t) :]
        v1_here_2d = v1_here[np.newaxis, :]
        v1_here_sum = ts_sum(v1_here_2d)
        if v1_here_sum[0, 0] > 0:
            v1_normalized = v1_here_2d / v1_here_sum
            for deltat in range(tmax - t0 - 1):
                if deltat < v1_normalized.shape[1]:
                    weight = v1_normalized[0, deltat]
                    number_postponed[:, t0 + 1 + deltat] += number_postponing_here * weight

    return number_postponed


# ---------------------------------------------------------------------------
# BaseFlowsModel
# ---------------------------------------------------------------------------


@define("BaseFlows")
class BaseFlowsModel(Model):
    """Baseline (unmodified) flow timeseries for the Bologna Mobility system.

    Loads inflow and starting timeseries in Bologna Mobility from raw data and exposes the
    raw timeseries needed by downstream models.  Derived quantities (baseline
    traffic, emissions) are computed by :class:`ModifiedZonalFlowsModel` and
    :class:`EmissionsModel` respectively, following the Bologna
    pattern.

    Parameters
    ----------
    ts :
        Shared time-range :class:`TimeseriesIndex`.
    ts_inflow_data :
        Numpy array of vehicle inflow values (one entry per hour, upsampled
        internally to 5-minute resolution).
    ts_starting_data :
        Numpy array of vehicle starting values.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`BaseFlowsModel`."""

        ts: TimeseriesIndex

    @outputs
    class Outputs:
        """Outputs of :class:`BaseFlowsModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        total_base_inflow: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute baseline flow timeseries."""
        ts_inflow = TimeseriesIndex("inflow", vehicle_inflow)
        ts_starting = TimeseriesIndex("starting", vehicle_starting)
        total_base_inflow = Index("total base vehicle inflow", ts_inflow.sum())

        return BaseFlowsModel.Outputs(
            ts_inflow=ts_inflow,
            ts_starting=ts_starting,
            total_base_inflow=total_base_inflow,
        )


@define("PolicyWindow")
class PolicyWindowModel(Model):
    """Time-delta indexes relative to the policy window [start_time, end_time].

    Parameters
    ----------
    ts :
        Shared time-range :class:`TimeseriesIndex`.
    i_p_start_time :
        Policy start time parameter (seconds from midnight).
    i_p_end_time :
        Policy end time parameter (seconds from midnight).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`PolicyWindowModel`."""

        ts: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index

    @outputs
    class Outputs:
        """Outputs of :class:`PolicyWindowModel`."""

        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute policy window time deltas."""
        delta_from_start = TimeseriesIndex(
            "delta time from start",
            graph.piecewise(
                (
                    (inputs.ts - inputs.i_p_start_time) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts >= inputs.i_p_start_time,
                ),
                (np.inf, True),
            ),
        )
        delta_to_end = TimeseriesIndex(
            "delta time to end",
            graph.piecewise(
                (
                    (inputs.i_p_end_time - inputs.ts) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts <= inputs.i_p_end_time,
                ),
                (np.inf, True),
            ),
        )
        delta_before_start = TimeseriesIndex(
            "delta time before start",
            graph.piecewise(
                (
                    (inputs.i_p_start_time - inputs.ts) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts < inputs.i_p_start_time,
                ),
                (np.inf, True),
            ),
        )
        delta_after_end = TimeseriesIndex(
            "delta time after end",
            graph.piecewise(
                (
                    (inputs.ts - inputs.i_p_end_time) / pd.Timedelta("1h").total_seconds(),
                    inputs.ts > inputs.i_p_end_time,
                ),
                (np.inf, True),
            ),
        )

        return PolicyWindowModel.Outputs(
            delta_from_start=delta_from_start,
            delta_to_end=delta_to_end,
            delta_before_start=delta_before_start,
            delta_after_end=delta_after_end,
        )


# ---------------------------------------------------------------------------
# TpmModalShiftModel / ZeroModalShiftModel — ModelVariant("ModalShift")
# ---------------------------------------------------------------------------


@define("TpmModalShift")
class TpmModalShiftModel(Model):
    """Modal-shift fractions computed via a logit public-transport model.

    Used when ``modal_shift_option`` is ``"tpm"``.

    Parameters
    ----------
    i_b_pt_capillarity :
        Behavioural importance of PT capillarity.
    i_b_pt_frequency :
        Behavioural importance of PT frequency.
    i_b_pt_cost :
        Behavioural importance of PT cost.
    i_b_pt_time :
        Behavioural importance of PT time difference.
    i_p_pt_capillarity_modification :
        Policy modification multiplier for capillarity.
    i_p_pt_frequency_modification :
        Policy modification multiplier for frequency.
    i_p_pt_cost_modification :
        Policy modification multiplier for cost.
    i_p_pt_time_modification :
        Policy modification multiplier for time.
    pt_capillarity :
        PT capillarity value.
    pt_frequency :
        PT frequency value.
    pt_avg_cost :
        PT average cost value.
    pt_avg_time_diff :
        PT average time-difference value.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`TpmModalShiftModel`."""

        i_b_pt_capillarity: Index
        i_b_pt_frequency: Index
        i_b_pt_cost: Index
        i_b_pt_time: Index
        i_p_pt_capillarity_modification: Index
        i_p_pt_frequency_modification: Index
        i_p_pt_cost_modification: Index
        i_p_pt_time_modification: Index

    @outputs
    class Outputs:
        """Outputs of :class:`TpmModalShiftModel`."""

        fraction_p_mode_shifted: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute modal shift fractions via logit model."""
        fraction_p_mode_shifted = Index(
            "fraction of possibly mode-shifted vehicles",
            1
            / (
                1
                + np.e
                ** (
                    -(
                        I_B_PT_BASELINE
                        + inputs.i_b_pt_capillarity * (1 + inputs.i_p_pt_capillarity_modification) * PT_CAPILLARITY
                        + inputs.i_b_pt_frequency * (1 + inputs.i_p_pt_frequency_modification) * PT_FREQUENCY
                        + inputs.i_b_pt_cost * (1 + inputs.i_p_pt_cost_modification) * PT_AVG_COST
                        + inputs.i_b_pt_time * (1 + inputs.i_p_pt_time_modification) * PT_AVG_TIME_DIFF
                    )
                )
            ),
        )

        return TpmModalShiftModel.Outputs(fraction_p_mode_shifted=fraction_p_mode_shifted)


@define("ZeroModalShift")
class ZeroModalShiftModel(Model):
    """Modal-shift fractions set to zero (no TPM).

    Used when ``modal_shift_option`` is ``"no"``.

    Declares the same :class:`Inputs` contract as :class:`TpmModalShiftModel`
    (fields are received but unused), ensuring the :class:`ModelVariant`
    contract is symmetric across both variants.

    Parameters
    ----------
    i_b_pt_capillarity :
        Behavioural importance of PT capillarity (declared, unused).
    i_b_pt_frequency :
        Behavioural importance of PT frequency (declared, unused).
    i_b_pt_cost :
        Behavioural importance of PT cost (declared, unused).
    i_b_pt_time :
        Behavioural importance of PT time difference (declared, unused).
    i_p_pt_capillarity_modification :
        Policy modification multiplier for capillarity (declared, unused).
    i_p_pt_frequency_modification :
        Policy modification multiplier for frequency (declared, unused).
    i_p_pt_cost_modification :
        Policy modification multiplier for cost (declared, unused).
    i_p_pt_time_modification :
        Policy modification multiplier for time (declared, unused).
    pt_capillarity :
        PT capillarity value (declared, unused).
    pt_frequency :
        PT frequency value (declared, unused).
    pt_avg_cost :
        PT average cost value (declared, unused).
    pt_avg_time_diff :
        PT average time-difference value (declared, unused).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`ZeroModalShiftModel`.

        Mirrors :class:`TpmModalShiftModel` ``Inputs`` exactly so that both
        variants satisfy the same :class:`ModelVariant` I/O contract.  All
        fields are declared but not used in the zero-shift computation.
        """

        i_b_pt_capillarity: Index
        i_b_pt_frequency: Index
        i_b_pt_cost: Index
        i_b_pt_time: Index
        i_p_pt_capillarity_modification: Index
        i_p_pt_frequency_modification: Index
        i_p_pt_cost_modification: Index
        i_p_pt_time_modification: Index

    @outputs
    class Outputs:
        """Outputs of :class:`ZeroModalShiftModel`."""

        fraction_p_mode_shifted: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute zero modal shift fractions."""
        fraction_p_mode_shifted = Index("fraction of possibly mode-shifted vehicles", 0.0)

        return ZeroModalShiftModel.Outputs(fraction_p_mode_shifted=fraction_p_mode_shifted)


# ---------------------------------------------------------------------------
# ParallelBehaviorModel
# ---------------------------------------------------------------------------


@define("ParallelBehavior")
class ParallelBehaviorModel(Model):
    """Behaviour fractions using the parallel (competitive) decision model.

    Vehicles independently decide whether to be rigid, anticipate, postpone,
    or mode-shift; joint probabilities are computed via :func:`ts_b_choose`.

    Parameters
    ----------
    modal_shift_option :
        Selector for the internal :class:`ModalShiftModel` variant.
    euro_class_split :
        Fleet euro-class proportions.
    pt_capillarity, pt_frequency, pt_avg_cost, pt_avg_time_diff :
        PT metrics (scalar values).
    ts, delta_from_start, delta_to_end :
        Wired from upstream models.
    i_p_start_time, i_p_end_time, i_p_fraction_exempted, i_p_cost :
        Policy parameter indexes.
    i_b_p50_cost, i_b_p50_anticipating, i_b_p50_postponing :
        Behavioural parameter indexes.
    i_b_pt_* / i_p_pt_* :
        PT behavioural and policy parameter indexes.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`ParallelBehaviorModel`."""

        modal_shift_option: Index
        ts: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_fraction_exempted: Index
        i_p_cost: list[Index]
        i_b_p50_cost: Index
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index
        i_b_pt_capillarity: Index
        i_b_pt_frequency: Index
        i_b_pt_cost: Index
        i_b_pt_time: Index
        i_p_pt_capillarity_modification: Index
        i_p_pt_frequency_modification: Index
        i_p_pt_cost_modification: Index
        i_p_pt_time_modification: Index

    @outputs
    class Outputs:
        """Outputs of :class:`ParallelBehaviorModel`."""

        fraction_rigid: TimeseriesIndex
        fraction_anticipating: TimeseriesIndex
        fraction_postponing: TimeseriesIndex
        fraction_mode_shifted: TimeseriesIndex
        fraction_lost: TimeseriesIndex
        fraction_rigid_euro_class: list[TimeseriesIndex]

    @expose
    class Expose:
        """Inspectable intermediates of :class:`ParallelBehaviorModel`."""

        fraction_p_rigid_euro_class: list[Index]
        fraction_p_rigid: Index
        fraction_p_anticipating: TimeseriesIndex
        fraction_p_postponing: TimeseriesIndex
        fraction_p_mode_shifted: Index

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        """Compute parallel behavior fractions."""
        modal_shift_option = inputs.modal_shift_option.value

        # Possibly-rigid fractions per euro class
        fraction_p_rigid_euro_class = [
            Index(
                f"fraction of possibly rigid vehicles per euro_{e}",
                graph.piecewise(
                    (0.0, inputs.i_b_p50_cost == 0),
                    (np.e ** (inputs.i_p_cost[e] / inputs.i_b_p50_cost * np.log(0.5)), True),
                ),
            )
            for e in range(7)
        ]
        _p_rigid_val: Any = sum(fraction_p_rigid_euro_class[e] * euro_class_split[f"euro_{e}"] for e in range(7))
        fraction_p_rigid = Index("fraction of possibly rigid vehicles", _p_rigid_val)

        # Possibly-anticipating fraction — area-based formula
        anticipate_reduction = Index(
            "anticipation reduction",
            (np.e ** (-np.log(2) / inputs.i_b_p50_anticipating) * (1 / P_DWELL_TIME_BM))
            / (1 - np.e ** (-np.log(2) / inputs.i_b_p50_anticipating) * (1 - 1 / P_DWELL_TIME_BM)),
        )
        fraction_p_anticipating = Index(
            "fraction of possibly anticipating vehicles",
            graph.piecewise(
                (0.0, inputs.i_b_p50_anticipating == 0),
                (
                    np.e
                    ** ((inputs.delta_from_start + P_RECORD_HEADWAY / 2) / inputs.i_b_p50_anticipating * np.log(0.5))
                    * anticipate_reduction,
                    True,
                ),
            ),
        )

        fraction_p_postponing = Index(
            "fraction of possibly postponing vehicles",
            graph.piecewise(
                (0.0, inputs.i_b_p50_postponing == 0),
                (
                    np.e ** ((inputs.delta_to_end + P_RECORD_HEADWAY / 2) / inputs.i_b_p50_postponing * np.log(0.5)),
                    True,
                ),
            ),
        )

        # Modal shift variant
        _ms_common = dict(
            i_b_pt_capillarity=inputs.i_b_pt_capillarity,
            i_b_pt_frequency=inputs.i_b_pt_frequency,
            i_b_pt_cost=inputs.i_b_pt_cost,
            i_b_pt_time=inputs.i_b_pt_time,
            i_p_pt_capillarity_modification=inputs.i_p_pt_capillarity_modification,
            i_p_pt_frequency_modification=inputs.i_p_pt_frequency_modification,
            i_p_pt_cost_modification=inputs.i_p_pt_cost_modification,
            i_p_pt_time_modification=inputs.i_p_pt_time_modification,
        )
        modal_shift: ModelVariant = ModelVariant(
            "ModalShift",
            variants={
                "tpm": TpmModalShiftModel(
                    inputs=TpmModalShiftModel.Inputs(**_ms_common),
                ),
                "no": ZeroModalShiftModel(
                    inputs=ZeroModalShiftModel.Inputs(**_ms_common),
                ),
            },
            selector=modal_shift_option,
        )
        fraction_p_mode_shifted: Index = modal_shift.outputs.fraction_p_mode_shifted

        # Fractions using ts_b_choose (competitive allocation)
        fraction_rigid = TimeseriesIndex(
            "fraction of rigid vehicles",
            graph.piecewise(
                (
                    graph.function_call(
                        "ts_b_choose",
                        fraction_p_rigid.node,
                        fraction_p_anticipating.node,
                        fraction_p_postponing.node,
                        fraction_p_mode_shifted.node,
                    )
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (1 - inputs.i_p_fraction_exempted, True),
            ),
        )
        # Post-hoc euro-class rescaling (parallel only)
        fraction_rigid_euro_class = [
            TimeseriesIndex(
                f"fraction rigid vehicles per euro_{e} %",
                graph.piecewise(
                    (
                        fraction_rigid
                        * euro_class_split[f"euro_{e}"]
                        * (fraction_p_rigid_euro_class[e] * euro_class_split[f"euro_{e}"])
                        / fraction_p_rigid,
                        fraction_p_rigid == 0,
                    ),
                    (fraction_rigid * euro_class_split[f"euro_{e}"], True),
                ),
            )
            for e in range(7)
        ]
        fraction_anticipating = TimeseriesIndex(
            "fraction of anticipating vehicles",
            graph.piecewise(
                (
                    graph.function_call(
                        "ts_b_choose",
                        fraction_p_anticipating.node,
                        fraction_p_rigid.node,
                        fraction_p_postponing.node,
                        fraction_p_mode_shifted.node,
                    )
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
            ),
        )
        fraction_postponing = TimeseriesIndex(
            "fraction of postponing vehicles",
            graph.piecewise(
                (
                    graph.function_call(
                        "ts_b_choose",
                        fraction_p_postponing.node,
                        fraction_p_rigid.node,
                        fraction_p_anticipating.node,
                        fraction_p_mode_shifted.node,
                    )
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
            ),
        )
        fraction_mode_shifted = TimeseriesIndex(
            "fraction of mode-shifted vehicles",
            graph.piecewise(
                (
                    graph.function_call(
                        "ts_b_choose",
                        fraction_p_mode_shifted.node,
                        fraction_p_postponing.node,
                        fraction_p_rigid.node,
                        fraction_p_anticipating.node,
                    )
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
            ),
        )
        fraction_lost = TimeseriesIndex(
            "fraction of lost vehicles",
            graph.piecewise(
                (
                    (1 - fraction_p_rigid)
                    * (1 - fraction_p_anticipating)
                    * (1 - fraction_p_postponing)
                    * (1 - fraction_p_mode_shifted)
                    * (1 - inputs.i_p_fraction_exempted),
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (0.0, True),
            ),
        )

        return ParallelBehaviorModel.Outputs(
            fraction_rigid=fraction_rigid,
            fraction_anticipating=fraction_anticipating,
            fraction_postponing=fraction_postponing,
            fraction_mode_shifted=fraction_mode_shifted,
            fraction_lost=fraction_lost,
            fraction_rigid_euro_class=fraction_rigid_euro_class,
        ), ParallelBehaviorModel.Expose(
            fraction_p_rigid_euro_class=fraction_p_rigid_euro_class,
            fraction_p_rigid=fraction_p_rigid,
            fraction_p_anticipating=fraction_p_anticipating,
            fraction_p_postponing=fraction_p_postponing,
            fraction_p_mode_shifted=fraction_p_mode_shifted,
        )


# ---------------------------------------------------------------------------
# FlexibleShiftFormula
# ---------------------------------------------------------------------------


@define("FlexibleShift")
class FlexibleShiftFormula(Model):
    """Closed-form exponential redistribution of anticipating/postponing vehicles.

    Used when ``time_shift_strategy`` is ``"flexible"``.

    Parameters
    ----------
    number_anticipating :
        Timeseries of anticipating vehicle counts.
    total_anticipating :
        Scalar total of anticipating vehicles.
    number_postponing :
        Timeseries of postponing vehicle counts.
    total_postponing :
        Scalar total of postponing vehicles.
    delta_before_start :
        Time before policy window start.
    delta_after_end :
        Time after policy window end.
    delta_from_start :
        Unused in this variant (present for shared Inputs contract).
    delta_to_end :
        Unused in this variant (present for shared Inputs contract).
    i_b_p50_anticipation :
        Median anticipation shift (hours).
    i_b_p50_postponement :
        Median postponement shift (hours).
    i_b_p50_anticipating :
        Unused in this variant (present for shared Inputs contract).
    i_b_p50_postponing :
        Unused in this variant (present for shared Inputs contract).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`FlexibleShiftFormula`."""

        number_anticipating: TimeseriesIndex
        total_anticipating: Index
        number_postponing: TimeseriesIndex
        total_postponing: Index
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index

    @outputs
    class Outputs:
        """Outputs of :class:`FlexibleShiftFormula`."""

        number_anticipated: TimeseriesIndex
        total_anticipated: Index
        number_postponed: TimeseriesIndex
        total_postponed: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute flexible shift redistribution."""
        number_anticipated = TimeseriesIndex(
            "anticipated vehicles",
            (
                np.e ** (-(inputs.delta_before_start - P_RECORD_HEADWAY) / inputs.i_b_p50_anticipation * np.log(2))
                - np.e ** (-inputs.delta_before_start / inputs.i_b_p50_anticipation * np.log(2))
            )
            * inputs.total_anticipating,
        )
        total_anticipated = Index("total vehicles anticipated", number_anticipated.sum())
        number_postponed = TimeseriesIndex(
            "postponed vehicles",
            (
                np.e ** (-(inputs.delta_after_end - P_RECORD_HEADWAY) / inputs.i_b_p50_postponement * np.log(2))
                - np.e ** (-inputs.delta_after_end / inputs.i_b_p50_postponement * np.log(2))
            )
            * inputs.total_postponing,
        )
        total_postponed = Index("total vehicles postponed", number_postponed.sum())

        return FlexibleShiftFormula.Outputs(
            number_anticipated=number_anticipated,
            total_anticipated=total_anticipated,
            number_postponed=number_postponed,
            total_postponed=total_postponed,
        )


# ---------------------------------------------------------------------------
# TimeShiftModel
# ---------------------------------------------------------------------------


@define("TimeShift")
class TimeShiftModel(Model):
    """Translates behaviour fractions into vehicle shift counts.

    Uses :class:`FlexibleShiftFormula` directly (no variant selection).

    Parameters
    ----------
    ts_inflow :
        Baseline inflow timeseries.
    fraction_anticipating, fraction_postponing :
        Wired from :class:`BehaviorModel`.
    fraction_mode_shifted, fraction_lost :
        Wired from :class:`BehaviorModel`.
    delta_from_start, delta_to_end :
        Wired from :class:`PolicyWindowModel`.
    delta_before_start, delta_after_end :
        Wired from :class:`PolicyWindowModel`.
    i_b_p50_anticipating, i_b_p50_postponing :
        Behavioural 50%-likelihood parameters.
    i_b_p50_anticipation, i_b_p50_postponement :
        Behavioural median-shift parameters.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`TimeShiftModel`."""

        ts_inflow: TimeseriesIndex
        fraction_anticipating: TimeseriesIndex
        fraction_postponing: TimeseriesIndex
        fraction_mode_shifted: TimeseriesIndex
        fraction_lost: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index

    @outputs
    class Outputs:
        """Outputs of :class:`TimeShiftModel`."""

        number_anticipating: TimeseriesIndex
        total_anticipating: Index
        number_anticipated: TimeseriesIndex
        total_anticipated: Index
        number_postponing: TimeseriesIndex
        total_postponing: Index
        number_postponed: TimeseriesIndex
        total_postponed: Index
        number_time_shifted: TimeseriesIndex
        total_time_shifted: Index
        number_mode_shifted: TimeseriesIndex
        total_mode_shifted: Index
        number_lost: TimeseriesIndex
        total_lost: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute time shift redistribution."""
        number_anticipating = TimeseriesIndex("anticipating vehicles", inputs.fraction_anticipating * inputs.ts_inflow)
        total_anticipating = Index("total anticipating vehicles", number_anticipating.sum())
        number_postponing = TimeseriesIndex("postponing vehicles", inputs.fraction_postponing * inputs.ts_inflow)
        total_postponing = Index("total postponing vehicles", number_postponing.sum())

        shift_formula = FlexibleShiftFormula(
            inputs=FlexibleShiftFormula.Inputs(
                number_anticipating=number_anticipating,
                total_anticipating=total_anticipating,
                number_postponing=number_postponing,
                total_postponing=total_postponing,
                delta_before_start=inputs.delta_before_start,
                delta_after_end=inputs.delta_after_end,
                delta_from_start=inputs.delta_from_start,
                delta_to_end=inputs.delta_to_end,
                i_b_p50_anticipation=inputs.i_b_p50_anticipation,
                i_b_p50_postponement=inputs.i_b_p50_postponement,
                i_b_p50_anticipating=inputs.i_b_p50_anticipating,
                i_b_p50_postponing=inputs.i_b_p50_postponing,
            ),
        )

        number_anticipated: TimeseriesIndex = shift_formula.outputs.number_anticipated
        total_anticipated: Index = shift_formula.outputs.total_anticipated
        number_postponed: TimeseriesIndex = shift_formula.outputs.number_postponed
        total_postponed: Index = shift_formula.outputs.total_postponed

        number_time_shifted = TimeseriesIndex("time-shifted vehicles", number_anticipated + number_postponed)
        total_time_shifted = Index("total vehicles shifted", total_anticipated + total_postponed)
        number_mode_shifted = TimeseriesIndex("mode-shifted vehicles", inputs.fraction_mode_shifted * inputs.ts_inflow)
        total_mode_shifted = Index("total mode shifted vehicles", number_mode_shifted.sum())
        number_lost = TimeseriesIndex("lost vehicles", inputs.fraction_lost * inputs.ts_inflow)
        total_lost = Index("total lost vehicles", number_lost.sum())

        return TimeShiftModel.Outputs(
            number_anticipating=number_anticipating,
            total_anticipating=total_anticipating,
            number_anticipated=number_anticipated,
            total_anticipated=total_anticipated,
            number_postponing=number_postponing,
            total_postponing=total_postponing,
            number_postponed=number_postponed,
            total_postponed=total_postponed,
            number_time_shifted=number_time_shifted,
            total_time_shifted=total_time_shifted,
            number_mode_shifted=number_mode_shifted,
            total_mode_shifted=total_mode_shifted,
            number_lost=number_lost,
            total_lost=total_lost,
        )


# ---------------------------------------------------------------------------
# AreaModifiedFlowsModel
# ---------------------------------------------------------------------------


def _compute_modified_inflow(
    ts_inflow: TimeseriesIndex,
    fraction_rigid: TimeseriesIndex,
    number_time_shifted: TimeseriesIndex,
    i_p_fraction_exempted: Index,
    i_p_start_time: Index,
    i_p_end_time: Index,
    ts: TimeseriesIndex,
) -> TimeseriesIndex:
    """Shared formula for modified inflow (identical across policies)."""
    return TimeseriesIndex(
        "modified vehicle inflow",
        graph.piecewise(
            (
                (i_p_fraction_exempted + fraction_rigid) * ts_inflow,
                (ts >= i_p_start_time) & (ts <= i_p_end_time),
            ),
            (ts_inflow + number_time_shifted, True),
        ),
    )


@define("AreaModifiedFlows")
class AreaModifiedFlowsModel(Model):
    """Modified inflow and starting under area-based policy.

    Under area policy, vehicles *starting* inside the BM are also subject
    to the pricing rule.  ``modified_starting`` is computed via
    :class:`FlexibleShiftFormula` applied to ``ts_starting``.

    Parameters
    ----------
    euro_class_split :
        Fleet euro-class proportions.
    ts, ts_inflow, ts_starting :
        Baseline timeseries.
    total_base_inflow :
        Scalar total baseline inflow.
    fraction_rigid, fraction_rigid_euro_class :
        Wired from :class:`BehaviorModel`.
    fraction_anticipating, fraction_postponing :
        Wired from :class:`BehaviorModel`.
    fraction_mode_shifted, fraction_lost :
        Wired from :class:`BehaviorModel`.
    number_time_shifted, number_mode_shifted, number_lost :
        Wired from :class:`TimeShiftModel`.
    delta_before_start, delta_after_end :
        Wired from :class:`PolicyWindowModel`.
    delta_from_start, delta_to_end :
        Wired from :class:`PolicyWindowModel`.
    i_p_start_time, i_p_end_time, i_p_fraction_exempted, i_p_cost :
        Policy parameter indexes.
    i_b_p50_anticipation, i_b_p50_postponement :
        Median-shift parameters.
    i_b_p50_anticipating, i_b_p50_postponing :
        50%-likelihood parameters.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`AreaModifiedFlowsModel`."""

        ts: TimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        total_base_inflow: Index
        fraction_rigid: TimeseriesIndex
        fraction_rigid_euro_class: list[TimeseriesIndex]
        fraction_anticipating: TimeseriesIndex
        fraction_postponing: TimeseriesIndex
        fraction_mode_shifted: TimeseriesIndex
        fraction_lost: TimeseriesIndex
        number_time_shifted: TimeseriesIndex
        number_mode_shifted: TimeseriesIndex
        number_lost: TimeseriesIndex
        delta_before_start: TimeseriesIndex
        delta_after_end: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_fraction_exempted: Index
        i_p_cost: list[Index]
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index

    @outputs
    class Outputs:
        """Outputs of :class:`AreaModifiedFlowsModel`."""

        modified_inflow: TimeseriesIndex
        total_modified_inflow: Index
        modified_starting: TimeseriesIndex
        modified_euro_class_split: list[Index]

    @expose
    class Expose:
        """Inside-shift indexes (area-based policy only)."""

        number_anticipating_inside: TimeseriesIndex
        total_anticipating_inside: Index
        number_anticipated_inside: TimeseriesIndex
        total_anticipated_inside: Index
        number_postponing_inside: TimeseriesIndex
        total_postponing_inside: Index
        number_postponed_inside: TimeseriesIndex
        total_postponed_inside: Index
        number_time_shifted_inside: TimeseriesIndex
        total_time_shifted_inside: Index
        number_mode_shifted_inside: TimeseriesIndex
        total_mode_shifted_inside: Index
        number_lost_inside: TimeseriesIndex
        total_lost_inside: Index

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        """Compute modified inflow and starting for area-based policy."""
        modified_inflow = _compute_modified_inflow(
            ts_inflow=inputs.ts_inflow,
            fraction_rigid=inputs.fraction_rigid,
            number_time_shifted=inputs.number_time_shifted,
            i_p_fraction_exempted=inputs.i_p_fraction_exempted,
            i_p_start_time=inputs.i_p_start_time,
            i_p_end_time=inputs.i_p_end_time,
            ts=inputs.ts,
        )
        total_modified_inflow = Index("total modified vehicle inflow", modified_inflow.sum())

        # Inside-shift computation for area-based modified_starting
        # (FlexibleShiftFormula reused with ts_starting instead of ts_inflow)
        number_anticipating_inside = TimeseriesIndex(
            "anticipating vehicles inside av", inputs.fraction_anticipating * inputs.ts_starting
        )
        total_anticipating_inside = Index(
            "total anticipating vehicles inside av",
            number_anticipating_inside.sum(),
        )
        number_postponing_inside = TimeseriesIndex(
            "postponing vehicles inside av", inputs.fraction_postponing * inputs.ts_starting
        )
        total_postponing_inside = Index(
            "total postponing vehicles inside av",
            number_postponing_inside.sum(),
        )

        inside_shift = FlexibleShiftFormula(
            inputs=FlexibleShiftFormula.Inputs(
                number_anticipating=number_anticipating_inside,
                total_anticipating=total_anticipating_inside,
                number_postponing=number_postponing_inside,
                total_postponing=total_postponing_inside,
                delta_before_start=inputs.delta_before_start,
                delta_after_end=inputs.delta_after_end,
                delta_from_start=inputs.delta_from_start,
                delta_to_end=inputs.delta_to_end,
                i_b_p50_anticipation=inputs.i_b_p50_anticipation,
                i_b_p50_postponement=inputs.i_b_p50_postponement,
                i_b_p50_anticipating=inputs.i_b_p50_anticipating,
                i_b_p50_postponing=inputs.i_b_p50_postponing,
            )
        )

        number_anticipated_inside: TimeseriesIndex = inside_shift.outputs.number_anticipated
        total_anticipated_inside: Index = inside_shift.outputs.total_anticipated
        number_postponed_inside: TimeseriesIndex = inside_shift.outputs.number_postponed
        total_postponed_inside: Index = inside_shift.outputs.total_postponed

        number_time_shifted_inside = TimeseriesIndex(
            "time-shifted vehicles inside av", number_anticipated_inside + number_postponed_inside
        )
        total_time_shifted_inside = Index(
            "total vehicles shifted inside av", total_anticipated_inside + total_postponed_inside
        )
        number_mode_shifted_inside = TimeseriesIndex(
            "mode-shifted vehicles inside av", inputs.fraction_mode_shifted * inputs.ts_starting
        )
        total_mode_shifted_inside = Index(
            "total mode shifted vehicles inside av",
            number_mode_shifted_inside.sum(),
        )
        number_lost_inside = TimeseriesIndex("lost vehicles inside av", inputs.fraction_lost * inputs.ts_starting)
        total_lost_inside = Index("total lost vehicles inside av", number_lost_inside.sum())

        modified_starting = TimeseriesIndex(
            "modified starting",
            graph.piecewise(
                (
                    (inputs.i_p_fraction_exempted + inputs.fraction_rigid) * inputs.ts_starting,
                    (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                ),
                (inputs.ts_starting + number_time_shifted_inside, True),
            ),
        )

        # Area euro-class split: fleet-only formula (starting also affected)
        modified_euro_class_split = [
            Index(
                f"modified split euro_{e} %",
                graph.piecewise(
                    (
                        (
                            (inputs.i_p_fraction_exempted * euro_class_split[f"euro_{e}"])
                            + inputs.fraction_rigid_euro_class[e]
                        )
                        / (inputs.i_p_fraction_exempted + inputs.fraction_rigid),
                        (inputs.ts >= inputs.i_p_start_time) & (inputs.ts <= inputs.i_p_end_time),
                    ),
                    (euro_class_split[f"euro_{e}"], True),
                ),
            )
            for e in range(7)
        ]

        return (
            AreaModifiedFlowsModel.Outputs(
                modified_inflow=modified_inflow,
                total_modified_inflow=total_modified_inflow,
                modified_starting=modified_starting,
                modified_euro_class_split=modified_euro_class_split,
            ),
            AreaModifiedFlowsModel.Expose(
                number_anticipating_inside=number_anticipating_inside,
                total_anticipating_inside=total_anticipating_inside,
                number_anticipated_inside=number_anticipated_inside,
                total_anticipated_inside=total_anticipated_inside,
                number_postponing_inside=number_postponing_inside,
                total_postponing_inside=total_postponing_inside,
                number_postponed_inside=number_postponed_inside,
                total_postponed_inside=total_postponed_inside,
                number_time_shifted_inside=number_time_shifted_inside,
                total_time_shifted_inside=total_time_shifted_inside,
                number_mode_shifted_inside=number_mode_shifted_inside,
                total_mode_shifted_inside=total_mode_shifted_inside,
                number_lost_inside=number_lost_inside,
                total_lost_inside=total_lost_inside,
            ),
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# AreaRevenueModel
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@define("AreaRevenues")
class AreaRevenueModel(Model):
    """Revenue computation under area-based policy.

    Under area policy, both *starting* and *inflow* is subject to pricing.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`AreaRevenueModel`."""

        ts: TimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        fraction_rigid: TimeseriesIndex
        fraction_rigid_euro_class: list[TimeseriesIndex]
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_cost: list[Index]

    @outputs
    class Outputs:
        """Outputs of :class:`AreaRevenueModel`."""

        number_paying: TimeseriesIndex
        total_paying: Index
        modified_avg_cost: Index
        total_paid: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute area-based revenue."""
        ts = inputs.ts
        ts_inflow = inputs.ts_inflow
        ts_starting = inputs.ts_starting
        fraction_rigid = inputs.fraction_rigid
        fraction_rigid_euro_class = inputs.fraction_rigid_euro_class
        i_p_start_time = inputs.i_p_start_time
        i_p_end_time = inputs.i_p_end_time
        i_p_cost = inputs.i_p_cost

        number_paying = TimeseriesIndex(
            "paying vehicles",
            graph.piecewise(
                (fraction_rigid * (ts_inflow + ts_starting), (ts >= i_p_start_time) & (ts <= i_p_end_time)),
                (0, True),
            ),
        )
        total_paying = Index("total vehicles paying", number_paying.sum())
        _avg_cost_num: Any = sum(i_p_cost[e] * fraction_rigid_euro_class[e] for e in range(7))
        modified_avg_cost = Index(
            "modified average cost with respect to the vehicles paying",
            graph.piecewise(
                (_avg_cost_num / fraction_rigid, fraction_rigid > 0),
                (0.0, True),
            ),
        )
        total_paid = Index("total paid fees", total_paying * modified_avg_cost)

        return AreaRevenueModel.Outputs(
            number_paying=number_paying,
            total_paying=total_paying,
            modified_avg_cost=modified_avg_cost,
            total_paid=total_paid,
        )


# ---------------------------------------------------------------------------
# [E] TrafficModel
# ---------------------------------------------------------------------------


@define("Traffic")
class TrafficModel(Model):
    """Computes traffic from inflow and starting timeseries.

    Computes baseline ``traffic`` from ``ts_inflow`` + ``ts_starting``, then
    ``modified_traffic`` from ``modified_inflow`` + ``modified_starting``.

    Parameters
    ----------
    ts_inflow :
        Baseline total inflow timeseries.
    ts_starting :
        Baseline starting timeseries.
    modified_inflow :
        Area-level modified inflow (wired from :class:`ModifiedFlowsVariant`).
    modified_starting :
        Area-level modified starting (wired from :class:`ModifiedFlowsVariant`).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`TrafficModel`."""

        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: TimeseriesIndex
        modified_starting: TimeseriesIndex

    @outputs
    class Outputs:
        """Outputs of :class:`TrafficModel`."""

        ts_traffic: TimeseriesIndex
        total_traffic: Index
        modified_traffic: TimeseriesIndex
        total_modified_traffic: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute traffic from inflow and starting timeseries."""
        ts_traffic = TimeseriesIndex(
            "reference traffic",
            graph.function_call("ts_solve", inputs.ts_inflow + inputs.ts_starting),
        )
        total_traffic = Index("max reference traffic", ts_traffic.max())
        modified_traffic = TimeseriesIndex(
            "modified traffic",
            graph.function_call("ts_solve", inputs.modified_inflow + inputs.modified_starting),
        )
        total_modified_traffic = Index("max modified traffic", modified_traffic.max())

        return TrafficModel.Outputs(
            ts_traffic=ts_traffic,
            total_traffic=total_traffic,
            modified_traffic=modified_traffic,
            total_modified_traffic=total_modified_traffic,
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# InducedDemandModel — ModelVariant("InducedDemandFormula")
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@define("NoInducedDemand")
class NoInducedDemand(Model):
    """Pass-through variant: leaves modified inflow/traffic unchanged.

    Used when ``induced_demand_strategy`` is ``"none"``.

    Parameters
    ----------
    ts_traffic, ts_inflow, ts_starting, modified_starting, delta_from_start,
    delta_to_end, i_b_share_induced_demand, i_b_p50_induced_demand :
        Unused in this variant (present for shared Inputs contract).
    modified_traffic :
        Modified traffic timeseries (wired from :class:`TrafficModel`).
    modified_inflow :
        Area-level modified inflow (wired from :class:`ModifiedFlowsVariant`).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`NoInducedDemand`."""

        ts_traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: TimeseriesIndex
        modified_starting: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_b_share_induced_demand: Index
        i_b_p50_induced_demand: Index

    @outputs
    class Outputs:
        """Outputs of :class:`NoInducedDemand`."""

        induced_demand: TimeseriesIndex
        adjusted_modified_inflow: TimeseriesIndex
        adjusted_modified_traffic: TimeseriesIndex
        total_adjusted_modified_traffic: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute no induced demand (pass-through)."""
        induced_demand = TimeseriesIndex(
            "induced demand",
            inputs.modified_inflow.node - inputs.modified_inflow.node,
        )
        adjusted_modified_inflow = TimeseriesIndex("adjusted modified inflow", inputs.modified_inflow.node)
        adjusted_modified_traffic = TimeseriesIndex("adjusted modified traffic", inputs.modified_traffic.node)
        total_adjusted_modified_traffic = Index("max adjusted modified traffic", adjusted_modified_traffic.max())

        return NoInducedDemand.Outputs(
            induced_demand=induced_demand,
            adjusted_modified_inflow=adjusted_modified_inflow,
            adjusted_modified_traffic=adjusted_modified_traffic,
            total_adjusted_modified_traffic=total_adjusted_modified_traffic,
        )


@define("ElemReliefInducedDemand")
class ElemReliefInducedDemand(Model):
    """ELEM-based induced demand from traffic relief during the policy window.

    Used when ``induced_demand_strategy`` is ``"elem_relief"``.

    Parameters
    ----------
    ts_traffic :
        Reference traffic timeseries (wired from :class:`TrafficModel`).
    modified_traffic :
        Modified traffic timeseries (wired from :class:`TrafficModel`).
    ts_inflow :
        Baseline total inflow timeseries.
    ts_starting :
        Baseline starting timeseries.
    modified_inflow :
        Area-level modified inflow (wired from :class:`ModifiedFlowsVariant`).
    modified_starting :
        Area-level modified starting (wired from :class:`ModifiedFlowsVariant`).
    delta_from_start, delta_to_end :
        Wired from :class:`PolicyWindowModel`; used to restrict the traffic
        relief computation to the policy window :math:`[t_s,t_e]`.
    i_b_share_induced_demand :
        Behavioural parameter :math:`\\rho`, the maximum share of additional
        traffic that congestion relief can induce.
    i_b_p50_induced_demand :
        Behavioural parameter :math:`\\Delta T_{50\\%}`, the median traffic
        relief required to activate latent demand.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`ElemReliefInducedDemand`."""

        ts_traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: TimeseriesIndex
        modified_starting: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_b_share_induced_demand: Index
        i_b_p50_induced_demand: Index

    @outputs
    class Outputs:
        """Outputs of :class:`ElemReliefInducedDemand`."""

        induced_demand: TimeseriesIndex
        adjusted_modified_inflow: TimeseriesIndex
        adjusted_modified_traffic: TimeseriesIndex
        total_adjusted_modified_traffic: Index

    def compute(self, inputs: Inputs) -> Outputs:
        # Delta T(t) = T(t) - T_m(t), restricted to t in [t_s, t_e]:
        # delta_from_start(t) is finite iff t >= t_s; delta_to_end(t) is
        # finite iff t <= t_e. The nested piecewise implements the AND of
        # the two conditions without requiring a boolean-combination operator.
        traffic_relief = TimeseriesIndex(
            "traffic relief",
            graph.piecewise(
                (
                    graph.piecewise(
                        (
                            inputs.ts_traffic - inputs.modified_traffic,
                            inputs.delta_to_end < np.inf,
                        ),
                        (0, True),
                    ),
                    inputs.delta_from_start < np.inf,
                ),
                (0, True),
            ),
        )

        # decay(t) = exp(- ln(2)/DeltaT50 * DeltaT(t))
        # Guard: when traffic_relief ≤ 0 (policy worsened traffic at a step, e.g. due
        # to time-shifting bunching at the window boundary), the exponential would
        # overflow to +∞ → induced_demand → −∞.  Physically there is no relief at
        # that step, so decay = 1 (induced_demand = 0) is correct.
        decay = graph.piecewise(
            (
                np.e ** (-traffic_relief / inputs.i_b_p50_induced_demand * np.log(2)),
                traffic_relief > 0,
            ),
            (1.0, True),
        )

        # Lambda(t) = rho * (I(t) + S(t))
        latent_pool = TimeseriesIndex(
            "latent demand pool",
            inputs.i_b_share_induced_demand * (inputs.ts_inflow + inputs.ts_starting),
        )

        # L(t) = Lambda(t) * (1 - decay(t)) = Lambda(t) - Lambda(t) * decay(t)
        induced_demand = TimeseriesIndex(
            "induced demand",
            latent_pool - latent_pool * decay,
        )

        # I_m^adj(t) = I_m(t) + L(t)
        adjusted_modified_inflow = TimeseriesIndex(
            "adjusted modified inflow",
            inputs.modified_inflow + induced_demand,
        )

        # T_m^adj(t) = ts_solve(I_m^adj + S_m)
        adjusted_modified_traffic = TimeseriesIndex(
            "adjusted modified traffic",
            graph.function_call("ts_solve", adjusted_modified_inflow + inputs.modified_starting),
        )
        total_adjusted_modified_traffic = Index("max adjusted modified traffic", adjusted_modified_traffic.max())

        return ElemReliefInducedDemand.Outputs(
            induced_demand=induced_demand,
            adjusted_modified_inflow=adjusted_modified_inflow,
            adjusted_modified_traffic=adjusted_modified_traffic,
            total_adjusted_modified_traffic=total_adjusted_modified_traffic,
        )


@define("InducedDemand")
class InducedDemandModel(Model):
    """Adjusts modified inflow/traffic to account for induced (latent) demand.

    Owns the :class:`InducedDemandFormula` :class:`ModelVariant` internally.

    Parameters
    ----------
    induced_demand_strategy :
        Selector for the internal :class:`InducedDemandFormula` variant.
    ts_traffic :
        Reference traffic timeseries (wired from :class:`TrafficModel`).
    modified_traffic :
        Modified traffic timeseries (wired from :class:`TrafficModel`).
    ts_inflow :
        Baseline total inflow timeseries.
    ts_starting :
        Baseline starting timeseries.
    modified_inflow :
        Area-level modified inflow (wired from :class:`ModifiedFlowsVariant`).
    modified_starting :
        Area-level modified starting (wired from :class:`ModifiedFlowsVariant`).
    delta_from_start, delta_to_end :
        Wired from :class:`PolicyWindowModel`; used to restrict the induced
        demand computation to the policy window :math:`[t_s,t_e]`.
    i_b_share_induced_demand :
        Behavioural parameter :math:`\\rho`, the maximum share of additional
        traffic that congestion relief can induce.
    i_b_p50_induced_demand :
        Behavioural parameter :math:`\\Delta T_{50\\%}`, the median traffic
        relief required to activate latent demand.
    """

    @inputs
    class Inputs:
        """Inputs of :class:`InducedDemandModel`."""

        induced_demand_strategy: Index
        ts_traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        ts_inflow: TimeseriesIndex
        ts_starting: TimeseriesIndex
        modified_inflow: TimeseriesIndex
        modified_starting: TimeseriesIndex
        delta_from_start: TimeseriesIndex
        delta_to_end: TimeseriesIndex
        i_b_share_induced_demand: Index
        i_b_p50_induced_demand: Index

    @outputs
    class Outputs:
        """Outputs of :class:`InducedDemandModel`."""

        induced_demand: TimeseriesIndex
        adjusted_modified_inflow: TimeseriesIndex
        adjusted_modified_traffic: TimeseriesIndex
        total_adjusted_modified_traffic: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute induced demand adjustment."""
        induced_demand_strategy = inputs.induced_demand_strategy.value

        _induced_demand_common = dict(
            ts_traffic=inputs.ts_traffic,
            modified_traffic=inputs.modified_traffic,
            ts_inflow=inputs.ts_inflow,
            ts_starting=inputs.ts_starting,
            modified_inflow=inputs.modified_inflow,
            modified_starting=inputs.modified_starting,
            delta_from_start=inputs.delta_from_start,
            delta_to_end=inputs.delta_to_end,
            i_b_share_induced_demand=inputs.i_b_share_induced_demand,
            i_b_p50_induced_demand=inputs.i_b_p50_induced_demand,
        )
        induced_demand_formula = ModelVariant(
            "InducedDemandFormula",
            variants={
                "none": NoInducedDemand(
                    inputs=NoInducedDemand.Inputs(**_induced_demand_common),
                ),
                "elem_relief": ElemReliefInducedDemand(
                    inputs=ElemReliefInducedDemand.Inputs(**_induced_demand_common),
                ),
            },
            selector=induced_demand_strategy,
        )

        induced_demand: TimeseriesIndex = induced_demand_formula.outputs.induced_demand
        adjusted_modified_inflow: TimeseriesIndex = induced_demand_formula.outputs.adjusted_modified_inflow
        adjusted_modified_traffic: TimeseriesIndex = induced_demand_formula.outputs.adjusted_modified_traffic
        total_adjusted_modified_traffic: Index = induced_demand_formula.outputs.total_adjusted_modified_traffic

        return InducedDemandModel.Outputs(
            induced_demand=induced_demand,
            adjusted_modified_inflow=adjusted_modified_inflow,
            adjusted_modified_traffic=adjusted_modified_traffic,
            total_adjusted_modified_traffic=total_adjusted_modified_traffic,
        )


@define("Emissions")
class EmissionsModel(Model):
    """Baseline and modified emissions at area level.

    Computes the fleet-average emission factor internally from ``euro_class_emission``
    and the module-level ``euro_class_split``, following the Bologna pattern.

    Parameters
    ----------
    ts_traffic :
        Baseline traffic timeseries.
    modified_traffic :
        Modified traffic timeseries.
    modified_euro_class_split :
        Modified fleet euro-class split.
    euro_class_emission :
        NOx emission factors per euro class (g/veh/km).
    """

    @inputs
    class Inputs:
        """Inputs of :class:`EmissionsModel`."""

        ts_traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        modified_euro_class_split: list[Index]

    @outputs
    class Outputs:
        """Outputs of :class:`EmissionsModel`."""

        emissions: TimeseriesIndex
        modified_emissions: TimeseriesIndex
        total_emissions: Index
        total_modified_emissions: Index

    @expose
    class Expose:
        """Inspectable intermediates of :class:`EmissionsModel`."""

        average_emissions: Index
        modified_average_emissions: TimeseriesIndex

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        """Compute baseline and modified emissions."""
        _avg_em_val: Any = sum(euro_class_emission[f"euro_{e}"] * euro_class_split[f"euro_{e}"] for e in range(7))
        average_emissions = Index("average emissions (per vehicle, per km)", _avg_em_val)
        _mod_em_val: Any = sum(euro_class_emission[f"euro_{e}"] * inputs.modified_euro_class_split[e] for e in range(7))
        modified_average_emissions = Index("modified average emissions (per vehicle, per km)", _mod_em_val)

        emissions = TimeseriesIndex("emissions", 2.5 * average_emissions * inputs.ts_traffic)
        modified_emissions = TimeseriesIndex(
            "modified emissions",
            2.5 * modified_average_emissions * inputs.modified_traffic,
        )
        total_emissions = Index("total emissions", emissions.sum())
        total_modified_emissions = Index("total modified emissions", modified_emissions.sum())

        return EmissionsModel.Outputs(
            emissions=emissions,
            modified_emissions=modified_emissions,
            total_emissions=total_emissions,
            total_modified_emissions=total_modified_emissions,
        ), EmissionsModel.Expose(
            average_emissions=average_emissions,
            modified_average_emissions=modified_average_emissions,
        )


# ---------------------------------------------------------------------------
# Root — BolognaMobilityModel
# ---------------------------------------------------------------------------


@define("Bologna Mobility")
class BolognaMobilityModel(Model):
    """Root model for the Bologna Mobility scenario.

    Composes all sub-models and exposes the final KPI outputs.

    All policy parameters (``I_P_*``) and behavioural parameters (``I_B_*``)
    are declared in ``Inputs`` and can be overridden at construction time.
    KPI outputs are declared on ``outputs``; timeseries used by plotting
    helpers are surfaced via ``expose``.

    Sub-models and parameters are accessible as named attributes::

        m = BolognaMobilityModel(inputs=BolognaMobilityModel.default_inputs())
        m.base_flows.outputs.ts_inflow
        m.behavior.outputs.fraction_rigid
        m.time_shift.outputs.number_time_shifted
        m.modified_flows.outputs.modified_flows
        m.emissions.outputs.total_modified_emissions

    Policy parameters (``I_P_*``) and behavioural parameters (``I_B_*``) are
    aliased directly on the model instance::

        m.I_P_start_time
        m.I_P_end_time
        m.I_P_cost          # list[Index], one per euro class
        m.I_P_fraction_exempted
        m.I_P_pt_frequency_modification
        m.I_P_pt_capillarity_modification
        m.I_P_pt_cost_modification
        m.I_P_pt_time_modification
        m.I_B_p50_cost
        m.I_B_p50_anticipating
        m.I_B_p50_postponing
        m.I_B_p50_anticipation
        m.I_B_p50_postponement
        m.I_B_pt_capillarity
        m.I_B_pt_frequency
        m.I_B_pt_cost
        m.I_B_pt_time
    """

    @inputs
    class Inputs:
        """Policy and behavioural parameters of :class:`BolognaMobilityModel`."""

        # Selectors (wrapped in Index for @inputs compatibility)
        modal_shift_option: Index
        induced_demand_strategy: Index

        # Policy parameters (I_P_*)
        i_p_start_time: Index
        i_p_end_time: Index
        i_p_cost: list[Index]
        i_p_fraction_exempted: Index
        i_p_pt_frequency_modification: Index
        i_p_pt_capillarity_modification: Index
        i_p_pt_cost_modification: Index
        i_p_pt_time_modification: Index

        # Behavioural parameters (I_B_*)
        i_b_p50_cost: DistributionIndex
        i_b_p50_anticipating: Index
        i_b_p50_postponing: Index
        i_b_p50_anticipation: Index
        i_b_p50_postponement: Index
        i_b_pt_capillarity: Index
        i_b_pt_frequency: Index
        i_b_pt_cost: Index
        i_b_pt_time: Index

        # Induced demand parameters
        i_b_share_induced_demand: Index
        i_b_p50_induced_demand: Index

    @outputs
    class Outputs:
        """KPI outputs of :class:`BolognaMobilityModel`."""

        total_base_inflow: Index
        total_modified_inflow: Index
        total_traffic: Index
        total_modified_traffic: Index
        total_mode_shifted: Index
        total_time_shifted: Index
        total_lost: Index
        total_paying: Index
        modified_avg_cost: Index
        total_paid: Index
        total_emissions: Index
        total_modified_emissions: Index

    @expose
    class Expose:
        """All sub-model indexes, collected so the engine reaches every node."""

        ts_inflow: TimeseriesIndex
        modified_inflow: TimeseriesIndex
        traffic: TimeseriesIndex
        modified_traffic: TimeseriesIndex
        emissions: TimeseriesIndex
        modified_emissions: TimeseriesIndex
        base_state_indexes: list[Any]
        policy_window_indexes: list[Any]
        behavior_indexes: list[Any]
        time_shift_indexes: list[Any]
        modified_flows_indexes: list[Any]
        emissions_indexes: list[Any]
        total_time_shifted_inside: Index
        total_mode_shifted_inside: Index
        total_lost_inside: Index
        i_b_share_induced_demand: Index
        i_b_p50_induced_demand: Index

    @classmethod
    def default_inputs(cls) -> Inputs:
        """Return the reference-scenario inputs as an :class:`~.Inputs` instance."""
        return cls.Inputs(
            modal_shift_option=Index("modal shift option", "tpm"),
            induced_demand_strategy=Index("induced demand strategy", "none"),
            i_p_start_time=Index(
                "start time",
                (pd.Timestamp("07:30:00") - pd.Timestamp("00:00:00")).total_seconds(),
            ),
            i_p_end_time=Index(
                "end time",
                (pd.Timestamp("19:30:00") - pd.Timestamp("00:00:00")).total_seconds(),
            ),
            i_p_cost=[Index(f"cost euro {e}", 5.00 - e * 0.25) for e in range(7)],
            i_p_fraction_exempted=Index("exempted vehicles %", 0.15),
            i_p_pt_frequency_modification=Index(
                "modification of the actual frequency of the PT",
                0.0,
            ),
            i_p_pt_capillarity_modification=Index(
                "modification of the actual capillarity of the PT",
                0.0,
            ),
            i_p_pt_cost_modification=Index(
                "modification of the actual cost of the PT",
                0.0,
            ),
            i_p_pt_time_modification=Index(
                "modification of the actual time difference of the PT",
                0.0,
            ),
            i_b_p50_cost=DistributionIndex(
                "cost 50% threshold",
                stats.uniform,
                {"loc": 4.00, "scale": 7.00},
            ),
            i_b_p50_anticipating=Index("anticipation 50% likelihood", 0.25),
            i_b_p50_postponing=Index("postponement 50% likelihood", 0.5),
            i_b_p50_anticipation=Index("anticipation distribution 50% threshold", 0.50),
            i_b_p50_postponement=Index("postponement distribution 50% threshold", 0.50),
            i_b_pt_capillarity=Index("importance level of capillarity of the pt", 4.5),
            i_b_pt_frequency=Index("importance level of frequency per stop of the pt", -1.45),
            i_b_pt_cost=Index("importance level of the cost of the pt", -0.30),
            i_b_pt_time=Index("importance level of the time difference of the car and pt", -0.034),
            i_b_share_induced_demand=Index(
                "maximum share of additional traffic induced by congestion relief",
                0.15,
            ),
            i_b_p50_induced_demand=Index("share of induced demand", 0.2),
        )

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        """Compose sub-models from inputs and return KPI outputs with expose timeseries."""
        # ---------------------------------------------------------------
        # Validate selectors
        # ---------------------------------------------------------------
        modal_shift_option = inputs.modal_shift_option.value
        induced_demand_strategy = inputs.induced_demand_strategy.value

        if modal_shift_option not in ("no", "tpm", "active", "tpm+active"):
            raise ValueError(
                f"modal_shift_option must be 'no', 'tpm', 'active', or 'tpm+active', got {modal_shift_option!r}"
            )

        # ---------------------------------------------------------------
        # Shared timeseries index
        # ---------------------------------------------------------------
        ts = TimeseriesIndex(
            "time range",
            np.array(
                [
                    (t - pd.Timestamp("00:00:00")).total_seconds()
                    for t in pd.date_range(start="00:00:00", periods=12 * 24, freq="5min")
                ]
            ),
        )

        # ---------------------------------------------------------------
        # Unpack inputs
        # ---------------------------------------------------------------
        i_p_start_time = inputs.i_p_start_time
        i_p_end_time = inputs.i_p_end_time
        i_p_cost = inputs.i_p_cost
        i_p_fraction_exempted = inputs.i_p_fraction_exempted
        i_p_pt_frequency_modification = inputs.i_p_pt_frequency_modification
        i_p_pt_capillarity_modification = inputs.i_p_pt_capillarity_modification
        i_p_pt_cost_modification = inputs.i_p_pt_cost_modification
        i_p_pt_time_modification = inputs.i_p_pt_time_modification

        i_b_p50_cost = inputs.i_b_p50_cost
        i_b_p50_anticipating = inputs.i_b_p50_anticipating
        i_b_p50_postponing = inputs.i_b_p50_postponing
        i_b_p50_anticipation = inputs.i_b_p50_anticipation
        i_b_p50_postponement = inputs.i_b_p50_postponement
        i_b_pt_capillarity = inputs.i_b_pt_capillarity
        i_b_pt_frequency = inputs.i_b_pt_frequency
        i_b_pt_cost = inputs.i_b_pt_cost
        i_b_pt_time = inputs.i_b_pt_time

        # ---------------------------------------------------------------
        # ---------------------------------------------------------------------------
        # BaseFlowsModel
        # ---------------------------------------------------------------------------
        # ---------------------------------------------------------------
        base_flows = BaseFlowsModel(
            inputs=BaseFlowsModel.Inputs(ts=ts),
        )

        # ---------------------------------------------------------------
        # [C] PolicyWindowModel
        # ---------------------------------------------------------------
        policy_window = PolicyWindowModel(
            inputs=PolicyWindowModel.Inputs(
                ts=ts,
                i_p_start_time=i_p_start_time,
                i_p_end_time=i_p_end_time,
            ),
        )

        # ---------------------------------------------------------------
        # [V,2] BehaviorModel — always ParallelBehaviorModel
        # ---------------------------------------------------------------
        behavior = ParallelBehaviorModel(
            inputs=ParallelBehaviorModel.Inputs(
                modal_shift_option=inputs.modal_shift_option,
                ts=ts,
                delta_from_start=policy_window.outputs.delta_from_start,
                delta_to_end=policy_window.outputs.delta_to_end,
                i_p_start_time=i_p_start_time,
                i_p_end_time=i_p_end_time,
                i_p_fraction_exempted=i_p_fraction_exempted,
                i_p_cost=i_p_cost,
                i_b_p50_cost=i_b_p50_cost,
                i_b_p50_anticipating=i_b_p50_anticipating,
                i_b_p50_postponing=i_b_p50_postponing,
                i_b_pt_capillarity=i_b_pt_capillarity,
                i_b_pt_frequency=i_b_pt_frequency,
                i_b_pt_cost=i_b_pt_cost,
                i_b_pt_time=i_b_pt_time,
                i_p_pt_capillarity_modification=i_p_pt_capillarity_modification,
                i_p_pt_frequency_modification=i_p_pt_frequency_modification,
                i_p_pt_cost_modification=i_p_pt_cost_modification,
                i_p_pt_time_modification=i_p_pt_time_modification,
            ),
        )

        # ---------------------------------------------------------------
        # ---------------------------------------------------------------------------
        # TimeShiftModel
        # ---------------------------------------------------------------------------
        # ---------------------------------------------------------------
        time_shift = TimeShiftModel(
            inputs=TimeShiftModel.Inputs(
                ts_inflow=base_flows.outputs.ts_inflow,
                fraction_anticipating=behavior.outputs.fraction_anticipating,
                fraction_postponing=behavior.outputs.fraction_postponing,
                fraction_mode_shifted=behavior.outputs.fraction_mode_shifted,
                fraction_lost=behavior.outputs.fraction_lost,
                delta_from_start=policy_window.outputs.delta_from_start,
                delta_to_end=policy_window.outputs.delta_to_end,
                delta_before_start=policy_window.outputs.delta_before_start,
                delta_after_end=policy_window.outputs.delta_after_end,
                i_b_p50_anticipating=i_b_p50_anticipating,
                i_b_p50_postponing=i_b_p50_postponing,
                i_b_p50_anticipation=i_b_p50_anticipation,
                i_b_p50_postponement=i_b_p50_postponement,
            ),
        )

        # ---------------------------------------------------------------
        # [V.4] ModifiedInflowModel — ModelVariant on policy
        # ---------------------------------------------------------------
        _inflow_common = dict(
            ts=ts,
            ts_inflow=base_flows.outputs.ts_inflow,
            ts_starting=base_flows.outputs.ts_starting,
            total_base_inflow=base_flows.outputs.total_base_inflow,
            fraction_rigid=behavior.outputs.fraction_rigid,
            fraction_rigid_euro_class=behavior.outputs.fraction_rigid_euro_class,
            fraction_anticipating=behavior.outputs.fraction_anticipating,
            fraction_postponing=behavior.outputs.fraction_postponing,
            fraction_mode_shifted=behavior.outputs.fraction_mode_shifted,
            fraction_lost=behavior.outputs.fraction_lost,
            number_time_shifted=time_shift.outputs.number_time_shifted,
            number_mode_shifted=time_shift.outputs.number_mode_shifted,
            number_lost=time_shift.outputs.number_lost,
            delta_before_start=policy_window.outputs.delta_before_start,
            delta_after_end=policy_window.outputs.delta_after_end,
            delta_from_start=policy_window.outputs.delta_from_start,
            delta_to_end=policy_window.outputs.delta_to_end,
            i_p_start_time=i_p_start_time,
            i_p_end_time=i_p_end_time,
            i_p_fraction_exempted=i_p_fraction_exempted,
            i_p_cost=i_p_cost,
            i_b_p50_anticipation=i_b_p50_anticipation,
            i_b_p50_postponement=i_b_p50_postponement,
            i_b_p50_anticipating=i_b_p50_anticipating,
            i_b_p50_postponing=i_b_p50_postponing,
        )
        modified_flows = AreaModifiedFlowsModel(inputs=AreaModifiedFlowsModel.Inputs(**_inflow_common))

        # ---------------------------------------------------------------
        # [V.5] Revenue — always AreaRevenueModel
        # ---------------------------------------------------------------
        revenue = AreaRevenueModel(
            inputs=AreaRevenueModel.Inputs(
                ts=ts,
                ts_inflow=base_flows.outputs.ts_inflow,
                ts_starting=base_flows.outputs.ts_starting,
                fraction_rigid=behavior.outputs.fraction_rigid,
                fraction_rigid_euro_class=behavior.outputs.fraction_rigid_euro_class,
                i_p_start_time=i_p_start_time,
                i_p_end_time=i_p_end_time,
                i_p_cost=i_p_cost,
            ),
        )

        # ---------------------------------------------------------------
        # ---------------------------------------------------------------------------
        # TrafficModel
        # ---------------------------------------------------------------------------
        # ---------------------------------------------------------------
        traffic = TrafficModel(
            inputs=TrafficModel.Inputs(
                ts_inflow=base_flows.outputs.ts_inflow,
                ts_starting=base_flows.outputs.ts_starting,
                modified_inflow=modified_flows.outputs.modified_inflow,
                modified_starting=modified_flows.outputs.modified_starting,
            ),
        )

        # ---------------------------------------------------------------
        # [V.6] InducedDemandModel
        # ---------------------------------------------------------------
        induced_demand = InducedDemandModel(
            inputs=InducedDemandModel.Inputs(
                induced_demand_strategy=inputs.induced_demand_strategy,
                ts_traffic=traffic.outputs.ts_traffic,
                modified_traffic=traffic.outputs.modified_traffic,
                ts_inflow=base_flows.outputs.ts_inflow,
                ts_starting=base_flows.outputs.ts_starting,
                modified_inflow=modified_flows.outputs.modified_inflow,
                modified_starting=modified_flows.outputs.modified_starting,
                delta_from_start=policy_window.outputs.delta_from_start,
                delta_to_end=policy_window.outputs.delta_to_end,
                i_b_share_induced_demand=inputs.i_b_share_induced_demand,
                i_b_p50_induced_demand=inputs.i_b_p50_induced_demand,
            ),
        )
        i_b_share_induced_demand = induced_demand.inputs.i_b_share_induced_demand
        i_b_p50_induced_demand = induced_demand.inputs.i_b_p50_induced_demand

        # ---------------------------------------------------------------
        # [G] EmissionsModel
        # ---------------------------------------------------------------
        emissions_model = EmissionsModel(
            inputs=EmissionsModel.Inputs(
                ts_traffic=traffic.outputs.ts_traffic,
                modified_traffic=induced_demand.outputs.adjusted_modified_traffic,
                modified_euro_class_split=modified_flows.outputs.modified_euro_class_split,
            ),
        )

        # ---------------------------------------------------------------
        # Named sub-model attributes for structured access
        # ---------------------------------------------------------------
        self.base_flows = base_flows
        self.policy_window = policy_window
        self.behavior = behavior
        self.time_shift = time_shift
        self.modified_flows = modified_flows
        self.revenue = revenue
        self.traffic = traffic
        self.emissions = emissions_model

        # View/dashboard aliases — backward-compatible flat attribute access
        self.TS_inflow = base_flows.outputs.ts_inflow
        self.TS_starting = base_flows.outputs.ts_starting
        self.I_modified_inflow = modified_flows.outputs.modified_inflow
        self.I_traffic = traffic.outputs.ts_traffic
        self.I_modified_traffic = traffic.outputs.modified_traffic
        self.I_emissions = emissions_model.outputs.emissions
        self.I_modified_emissions = emissions_model.outputs.modified_emissions

        # Policy parameters — aliased for direct external access
        self.I_P_start_time = i_p_start_time
        self.I_P_end_time = i_p_end_time
        self.I_P_cost = i_p_cost
        self.I_P_fraction_exempted = i_p_fraction_exempted
        self.I_P_pt_frequency_modification = i_p_pt_frequency_modification
        self.I_P_pt_capillarity_modification = i_p_pt_capillarity_modification
        self.I_P_pt_cost_modification = i_p_pt_cost_modification
        self.I_P_pt_time_modification = i_p_pt_time_modification

        # Behavioural parameters — aliased for direct external access
        self.I_B_p50_cost = i_b_p50_cost
        self.I_B_p50_anticipating = i_b_p50_anticipating
        self.I_B_p50_postponing = i_b_p50_postponing
        self.I_B_p50_anticipation = i_b_p50_anticipation
        self.I_B_p50_postponement = i_b_p50_postponement
        self.I_B_pt_capillarity = i_b_pt_capillarity
        self.I_B_pt_frequency = i_b_pt_frequency
        self.I_B_pt_cost = i_b_pt_cost
        self.I_B_pt_time = i_b_pt_time

        # ---------------------------------------------------------------
        # Return Outputs and Expose
        # ---------------------------------------------------------------
        return (
            BolognaMobilityModel.Outputs(
                total_base_inflow=base_flows.outputs.total_base_inflow,
                total_modified_inflow=modified_flows.outputs.total_modified_inflow,
                total_traffic=traffic.outputs.total_traffic,
                total_modified_traffic=induced_demand.outputs.total_adjusted_modified_traffic,
                total_mode_shifted=time_shift.outputs.total_mode_shifted,
                total_time_shifted=time_shift.outputs.total_time_shifted,
                total_lost=time_shift.outputs.total_lost,
                total_paying=revenue.outputs.total_paying,
                modified_avg_cost=revenue.outputs.modified_avg_cost,
                total_paid=revenue.outputs.total_paid,
                total_emissions=emissions_model.outputs.total_emissions,
                total_modified_emissions=emissions_model.outputs.total_modified_emissions,
            ),
            BolognaMobilityModel.Expose(
                ts_inflow=base_flows.outputs.ts_inflow,
                modified_inflow=induced_demand.outputs.adjusted_modified_inflow,
                traffic=traffic.outputs.ts_traffic,
                modified_traffic=induced_demand.outputs.adjusted_modified_traffic,
                emissions=emissions_model.outputs.emissions,
                modified_emissions=emissions_model.outputs.modified_emissions,
                base_state_indexes=list(base_flows.indexes),
                policy_window_indexes=list(policy_window.indexes),
                behavior_indexes=list(behavior.indexes),
                time_shift_indexes=list(time_shift.indexes),
                modified_flows_indexes=list(modified_flows.indexes),
                emissions_indexes=list(emissions_model.indexes),
                total_mode_shifted_inside=modified_flows.expose.total_mode_shifted_inside,
                total_time_shifted_inside=modified_flows.expose.total_time_shifted_inside,
                total_lost_inside=modified_flows.expose.total_lost_inside,
                i_b_share_induced_demand=i_b_share_induced_demand,
                i_b_p50_induced_demand=i_b_p50_induced_demand,
            ),
        )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def evaluate(model: BolognaMobilityModel, size: int = 1) -> dict:
    """Evaluate *model* and return a dict mapping each index to its samples.

    Parameters
    ----------
    model :
        A constructed :class:`BolognaMobilityModel`.
    size :
        Number of Monte-Carlo samples.

    Returns
    -------
    dict
        Mapping from :class:`~civic_digital_twins.dt_model.GenericIndex` to
        ``np.ndarray`` of shape ``(size, T)`` or ``(size, 1)``.
    """
    ensemble = DistributionEnsemble(model, size)
    result = Evaluation(model).evaluate(
        ensemble,
        functions={
            "ts_solve": executor.LambdaAdapter(ts_solve),
            "ts_b_choose": executor.LambdaAdapter(ts_b_choose),
            "ts_anticipate": executor.LambdaAdapter(ts_anticipate),
            "ts_postpone": executor.LambdaAdapter(ts_postpone),
        },
    )
    subs: dict = {}
    for idx in model.indexes:
        val = result[idx]
        if val.ndim == 0:
            subs[idx] = np.full((size, 1), float(val))
        elif val.ndim == 1:
            subs[idx] = np.expand_dims(val, axis=0)
        else:
            subs[idx] = val
    return subs
