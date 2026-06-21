"""Molveno overtourism model definition — modular decomposition.

The model is split into four concern sub-models plus a root model.  Context
variables (``cv_*``) and presence variables (``pv_*``) are constructed
directly on the root :class:`MolvenoModel` and wired down to each concern
sub-model through its ``Inputs`` dataclass.

:class:`ParkingModel` — *Parking usage*
    **Inputs**: ``pv_tourists``, ``pv_excursionists``, ``cv_weather``,
    ``i_u_tourists_parking``, ``i_u_excursionists_parking``,
    ``i_xa_tourists_per_vehicle``, ``i_xa_excursionists_per_vehicle``,
    ``i_xo_tourists_parking``, ``i_xo_excursionists_parking``,
    ``i_c_parking``
    **Outputs**: ``i_u_parking``

:class:`BeachModel` — *Beach usage*
    **Inputs**: ``pv_tourists``, ``pv_excursionists``, ``cv_weather``,
    ``i_u_tourists_beach``, ``i_u_excursionists_beach``,
    ``i_xo_tourists_beach`` *(uncertain)*, ``i_xo_excursionists_beach``,
    ``i_c_beach``
    **Outputs**: ``i_u_beach``

:class:`AccommodationModel` — *Accommodation usage*
    **Inputs**: ``pv_tourists``,
    ``i_u_tourists_accommodation``, ``i_xa_tourists_accommodation``,
    ``i_c_accommodation``
    **Outputs**: ``i_u_accommodation``

:class:`FoodModel` — *Food-service usage*
    **Inputs**: ``pv_tourists``, ``pv_excursionists``, ``cv_weather``,
    ``i_u_tourists_food``, ``i_u_excursionists_food``,
    ``i_xa_visitors_food``, ``i_xo_visitors_food``, ``i_c_food``
    **Outputs**: ``i_u_food``

:class:`MolvenoModel` — *Root, owns CVs, PVs, and all* ``i_*`` *defaults*
    Creates the three context variables
    (:class:`~civic_digital_twins.dt_model.CategoricalIndex`), the two
    presence variables, and all ``i_*`` indexes with their default values,
    then passes them to the four concern sub-models.  Retains the domain
    attributes (``cvs``, ``pvs``, ``constraints``) required by
    :class:`~dt_model.CrossProductEnsemble`.

Design rules:

* **All** ``i_*`` parameters are ``Inputs`` to the sub-model that uses
  them, including uncertain ``DistributionIndex`` values.  The default
  values are created by :class:`MolvenoModel` and passed down via
  constructors.  A caller who wants to override a parameter simply
  supplies a different index object at construction time.
* Context variables (``cv_*``) and presence variables (``pv_*``) are
  attributes of :class:`MolvenoModel` and are wired as ``Inputs`` to the
  concern sub-models that consume them.
* Each concern sub-model's ``Outputs`` contains only the usage-formula
  index (``i_u_*``).  Capacity indexes (``i_c_*``) remain as ``Inputs``
  because they are parameters, not computed results.
* Each concern sub-model stores its
  :class:`~overtourism_molveno.molveno_model.Constraint` as a
  plain instance attribute (``self.constraint``) because
  :class:`~overtourism_molveno.molveno_model.Constraint` is not a
  :class:`~dt_model.model.index.GenericIndex` and must not appear inside
  an :class:`~dt_model.model.model.IOProxy`.
* :class:`MolvenoModel` subclasses :class:`~dt_model.model.model.Model`
  directly and exposes ``.cvs``, ``.pvs``, and ``.constraints`` attributes
  so that :class:`~dt_model.CrossProductEnsemble`
  and the evaluation code can consume them without modification.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import interpolate, ndimage, stats

from civic_digital_twins.dt_model import (
    CategoricalIndex,
    ConditionalDistributionIndex,
    CrossProductEnsemble,
    DistributionIndex,
    Evaluation,
    EvaluationResult,
    GenericIndex,
    Index,
    Model,
    Scenario,
    define,
    expose,
    graph,
    inputs,
    outputs,
    sample_across,
)
from civic_digital_twins.dt_model.model.index import Distribution
from civic_digital_twins.dt_model.simulation.handle import _get_default_executor
from civic_digital_twins.dt_model.simulation.runner import (
    EvaluationConfig,
    ModelEvaluator,
    ModelOutput,
    ModelRunHandle,
)

try:
    from .molveno_presence_stats import (
        excursionist_presences_stats,
        season,
        tourist_presences_stats,
        weather,
        weekday,
    )
except ImportError:
    from molveno_presence_stats import (
        excursionist_presences_stats,
        season,
        tourist_presences_stats,
        weather,
        weekday,
    )


# ---------------------------------------------------------------------------
# Constraint
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Constraint:
    """Named pairing of a usage formula index and a capacity index.

    Both *usage* and *capacity* are formula-mode or distribution-backed
    :class:`~dt_model.model.index.Index` objects, so the entire constraint is
    expressed in terms of :class:`~dt_model.model.index.GenericIndex` — no
    engine-layer types appear in the public API.

    Identity-based hashing (``eq=False``) keeps ``Constraint`` objects usable
    as dict keys, matching the convention used by ``graph.Node`` and
    ``GenericIndex``.
    """

    name: str
    usage: Index  # formula-mode Index wrapping the usage expression
    capacity: Index  # constant, distribution-backed, or formula-mode Index


# ---------------------------------------------------------------------------
# ParkingModel
# ---------------------------------------------------------------------------


@define("Parking")
class ParkingModel(Model):
    """Concern sub-model — parking usage.

    All parameters (usage factors, conversion factors, capacity) are
    received as ``Inputs`` so that callers can override any default.
    :class:`MolvenoModel` creates the indexes with their default values and
    passes them in.

    The usage formula ``i_u_parking`` is the single contractual ``Output``.
    The :class:`~overtourism_molveno.molveno_model.Constraint` is
    stored as a plain instance attribute ``self.constraint``.

    Attributes
    ----------
    constraint : Constraint
        The parking constraint (usage / capacity pair).
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`ParkingModel`."""

        pv_tourists: ConditionalDistributionIndex
        pv_excursionists: ConditionalDistributionIndex
        cv_weather: CategoricalIndex
        i_u_tourists_parking: Index
        i_u_excursionists_parking: Index
        i_xa_tourists_per_vehicle: Index
        i_xa_excursionists_per_vehicle: Index
        i_xo_tourists_parking: Index
        i_xo_excursionists_parking: Index
        i_c_parking: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`ParkingModel`."""

        i_u_parking: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute parking usage from inputs."""
        i_u_parking = Index(
            "parking usage",
            inputs.pv_tourists
            * inputs.i_u_tourists_parking
            / (inputs.i_xa_tourists_per_vehicle * inputs.i_xo_tourists_parking)
            + inputs.pv_excursionists
            * inputs.i_u_excursionists_parking
            / (inputs.i_xa_excursionists_per_vehicle * inputs.i_xo_excursionists_parking),
        )
        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(name="parking", usage=i_u_parking, capacity=inputs.i_c_parking)
        return ParkingModel.Outputs(i_u_parking=i_u_parking)


# ---------------------------------------------------------------------------
# BeachModel
# ---------------------------------------------------------------------------


@define("Beach")
class BeachModel(Model):
    """Concern sub-model — beach usage.

    All parameters (usage factors, rotation factors, capacity) are received
    as ``Inputs``.  The uncertain rotation factor ``i_xo_tourists_beach`` is
    passed in from :class:`MolvenoModel` so it appears in the root
    ``model.indexes`` and is sampled by
    :class:`~dt_model.CrossProductEnsemble`.

    Attributes
    ----------
    constraint : Constraint
        The beach constraint (usage / capacity pair).
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`BeachModel`."""

        pv_tourists: ConditionalDistributionIndex
        pv_excursionists: ConditionalDistributionIndex
        cv_weather: CategoricalIndex
        i_u_tourists_beach: Index
        i_u_excursionists_beach: Index
        i_xo_tourists_beach: DistributionIndex
        i_xo_excursionists_beach: Index
        i_c_beach: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`BeachModel`."""

        i_u_beach: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute beach usage from inputs."""
        i_u_beach = Index(
            "beach usage",
            inputs.pv_tourists * inputs.i_u_tourists_beach / inputs.i_xo_tourists_beach
            + inputs.pv_excursionists * inputs.i_u_excursionists_beach / inputs.i_xo_excursionists_beach,
        )
        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(name="beach", usage=i_u_beach, capacity=inputs.i_c_beach)
        return BeachModel.Outputs(i_u_beach=i_u_beach)


# ---------------------------------------------------------------------------
# AccommodationModel
# ---------------------------------------------------------------------------


@define("Accommodation")
class AccommodationModel(Model):
    """Concern sub-model — accommodation usage.

    Attributes
    ----------
    constraint : Constraint
        The accommodation constraint (usage / capacity pair).
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`AccommodationModel`."""

        pv_tourists: ConditionalDistributionIndex
        i_u_tourists_accommodation: Index
        i_xa_tourists_accommodation: Index
        i_c_accommodation: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`AccommodationModel`."""

        i_u_accommodation: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute accommodation usage from inputs."""
        i_u_accommodation = Index(
            "accommodation usage",
            inputs.pv_tourists * inputs.i_u_tourists_accommodation / inputs.i_xa_tourists_accommodation,
        )
        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(name="accommodation", usage=i_u_accommodation, capacity=inputs.i_c_accommodation)
        return AccommodationModel.Outputs(i_u_accommodation=i_u_accommodation)


# ---------------------------------------------------------------------------
# FoodModel
# ---------------------------------------------------------------------------


@define("Food")
class FoodModel(Model):
    """Concern sub-model — food-service usage.

    Attributes
    ----------
    constraint : Constraint
        The food-service constraint (usage / capacity pair).
    """

    @inputs
    class Inputs:
        """Contractual inputs of :class:`FoodModel`."""

        pv_tourists: ConditionalDistributionIndex
        pv_excursionists: ConditionalDistributionIndex
        cv_weather: CategoricalIndex
        i_u_tourists_food: Index
        i_u_excursionists_food: Index
        i_xa_visitors_food: Index
        i_xo_visitors_food: Index
        i_c_food: DistributionIndex

    @outputs
    class Outputs:
        """Contractual outputs of :class:`FoodModel`."""

        i_u_food: Index

    def compute(self, inputs: Inputs) -> Outputs:
        """Compute food-service usage from inputs."""
        i_u_food = Index(
            "food usage",
            (inputs.pv_tourists * inputs.i_u_tourists_food + inputs.pv_excursionists * inputs.i_u_excursionists_food)
            / (inputs.i_xa_visitors_food * inputs.i_xo_visitors_food),
        )
        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(name="food", usage=i_u_food, capacity=inputs.i_c_food)
        return FoodModel.Outputs(i_u_food=i_u_food)


# ---------------------------------------------------------------------------
# MolvenoModel  (root)
# ---------------------------------------------------------------------------


@define("base model")
class MolvenoModel(Model):
    """Root overtourism model that wires the four concern sub-models.

    All domain parameters are declared as ``Inputs``; supply defaults via
    :meth:`default_inputs` or override individual fields with
    :func:`dataclasses.replace`::

        m = MolvenoModel(inputs=MolvenoModel.default_inputs())
    """

    @inputs
    class Inputs:
        """All domain parameters of :class:`MolvenoModel`."""

        # Context variables
        cv_weekday: CategoricalIndex
        cv_season: CategoricalIndex
        cv_weather: CategoricalIndex
        # Presence distributions
        pv_tourists: ConditionalDistributionIndex
        pv_excursionists: ConditionalDistributionIndex
        # Distribution-backed uncertainty parameters
        i_c_parking: DistributionIndex
        i_c_beach: DistributionIndex
        i_c_accommodation: DistributionIndex
        i_c_food: DistributionIndex
        i_xo_tourists_beach: DistributionIndex
        # Parking parameters
        i_u_tourists_parking: Index
        i_u_excursionists_parking: Index
        i_xa_tourists_per_vehicle: Index
        i_xa_excursionists_per_vehicle: Index
        i_xo_tourists_parking: Index
        i_xo_excursionists_parking: Index
        # Beach parameters
        i_u_tourists_beach: Index
        i_u_excursionists_beach: Index
        i_xo_excursionists_beach: Index
        # Accommodation parameters
        i_u_tourists_accommodation: Index
        i_xa_tourists_accommodation: Index
        # Food parameters
        i_u_tourists_food: Index
        i_u_excursionists_food: Index
        i_xa_visitors_food: Index
        i_xo_visitors_food: Index
        # Presence-transformation parameters
        i_p_tourists_reduction_factor: Index
        i_p_excursionists_reduction_factor: Index
        i_p_tourists_saturation_level: Index
        i_p_excursionists_saturation_level: Index

    @outputs
    class Outputs:
        """Contractual outputs of :class:`MolvenoModel`."""

        usage_indexes: list[GenericIndex]

    @expose
    class Expose:
        """Sub-model output proxies for inspection."""

        parking: ParkingModel.Outputs
        beach: BeachModel.Outputs
        accommodation: AccommodationModel.Outputs
        food: FoodModel.Outputs

    @classmethod
    def default_inputs(cls) -> Inputs:
        """Return the default domain inputs for all parameters.

        Pass to :class:`MolvenoModel` or override individual fields with
        :func:`dataclasses.replace`::

            m = MolvenoModel(inputs=MolvenoModel.default_inputs())
        """
        cv_weekday = CategoricalIndex("weekday", {d: 1.0 / len(weekday) for d in weekday})
        cv_season = CategoricalIndex("season", {v: season[v] for v in season})
        cv_weather = CategoricalIndex("weather", {v: weather[v] for v in weather})
        pv_tourists = ConditionalDistributionIndex(
            "tourists",
            [cv_weekday, cv_season, cv_weather],
            tourist_presences_stats,
        )
        pv_excursionists = ConditionalDistributionIndex(
            "excursionists",
            [cv_weekday, cv_season, cv_weather],
            excursionist_presences_stats,
        )
        return cls.Inputs(
            cv_weekday=cv_weekday,
            cv_season=cv_season,
            cv_weather=cv_weather,
            pv_tourists=pv_tourists,
            pv_excursionists=pv_excursionists,
            # Distribution-backed uncertainty parameters
            i_c_parking=DistributionIndex("parking capacity", stats.uniform, {"loc": 350.0, "scale": 100.0}),
            i_c_beach=DistributionIndex("beach capacity", stats.uniform, {"loc": 6000.0, "scale": 1000.0}),
            i_c_accommodation=DistributionIndex(
                "accommodation capacity",
                stats.lognorm,
                {"s": 0.125, "loc": 0.0, "scale": 5000.0},
            ),
            i_c_food=DistributionIndex(
                "food service capacity",
                stats.triang,
                {"loc": 3000.0, "scale": 1000.0, "c": 0.5},
            ),
            i_xo_tourists_beach=DistributionIndex(
                "tourists on beach rotation factor",
                stats.uniform,
                {"loc": 1.0, "scale": 2.0},
            ),
            # Parking parameters
            i_u_tourists_parking=Index("tourist parking usage factor", 0.02),
            i_u_excursionists_parking=Index(
                "excursionist parking usage factor",
                graph.piecewise((0.55, cv_weather == "bad"), (0.80, True)),
            ),
            i_xa_tourists_per_vehicle=Index("tourists per vehicle allocation factor", 2.5),
            i_xa_excursionists_per_vehicle=Index("excursionists per vehicle allocation factor", 2.5),
            i_xo_tourists_parking=Index("tourists in parking rotation factor", 1.02),
            i_xo_excursionists_parking=Index("excursionists in parking rotation factor", 3.5),
            # Beach parameters
            i_u_tourists_beach=Index(
                "tourist beach usage factor",
                graph.piecewise((0.25, cv_weather == "bad"), (0.50, True)),
            ),
            i_u_excursionists_beach=Index(
                "excursionist beach usage factor",
                graph.piecewise((0.35, cv_weather == "bad"), (0.80, True)),
            ),
            i_xo_excursionists_beach=Index("excursionists on beach rotation factor", 1.02),
            # Accommodation parameters
            i_u_tourists_accommodation=Index("tourist accommodation usage factor", 0.90),
            i_xa_tourists_accommodation=Index("tourists per accommodation allocation factor", 1.05),
            # Food parameters
            i_u_tourists_food=Index("tourist food service usage factor", 0.20),
            i_u_excursionists_food=Index(
                "excursionist food service usage factor",
                graph.piecewise((0.80, cv_weather == "bad"), (0.40, True)),
            ),
            i_xa_visitors_food=Index("visitors in food service allocation factor", 0.9),
            i_xo_visitors_food=Index("visitors in food service rotation factor", 2.0),
            # Presence-transformation parameters
            i_p_tourists_reduction_factor=Index("tourists reduction factor", 1.0),
            i_p_excursionists_reduction_factor=Index("excursionists reduction factor", 1.0),
            i_p_tourists_saturation_level=Index("tourists saturation level", 10000),
            i_p_excursionists_saturation_level=Index("excursionists saturation level", 10000),
        )

    def compute(self, inputs: Inputs) -> tuple[Outputs, Expose]:
        """Wire concern sub-models from inputs."""
        parking = ParkingModel(
            inputs=ParkingModel.Inputs(  # type: ignore[call-arg]
                pv_tourists=inputs.pv_tourists,
                pv_excursionists=inputs.pv_excursionists,
                cv_weather=inputs.cv_weather,
                i_u_tourists_parking=inputs.i_u_tourists_parking,
                i_u_excursionists_parking=inputs.i_u_excursionists_parking,
                i_xa_tourists_per_vehicle=inputs.i_xa_tourists_per_vehicle,
                i_xa_excursionists_per_vehicle=inputs.i_xa_excursionists_per_vehicle,
                i_xo_tourists_parking=inputs.i_xo_tourists_parking,
                i_xo_excursionists_parking=inputs.i_xo_excursionists_parking,
                i_c_parking=inputs.i_c_parking,
            )
        )
        beach = BeachModel(
            inputs=BeachModel.Inputs(  # type: ignore[call-arg]
                pv_tourists=inputs.pv_tourists,
                pv_excursionists=inputs.pv_excursionists,
                cv_weather=inputs.cv_weather,
                i_u_tourists_beach=inputs.i_u_tourists_beach,
                i_u_excursionists_beach=inputs.i_u_excursionists_beach,
                i_xo_tourists_beach=inputs.i_xo_tourists_beach,
                i_xo_excursionists_beach=inputs.i_xo_excursionists_beach,
                i_c_beach=inputs.i_c_beach,
            )
        )
        accommodation = AccommodationModel(
            inputs=AccommodationModel.Inputs(  # type: ignore[call-arg]
                pv_tourists=inputs.pv_tourists,
                i_u_tourists_accommodation=inputs.i_u_tourists_accommodation,
                i_xa_tourists_accommodation=inputs.i_xa_tourists_accommodation,
                i_c_accommodation=inputs.i_c_accommodation,
            )
        )
        food = FoodModel(
            inputs=FoodModel.Inputs(  # type: ignore[call-arg]
                pv_tourists=inputs.pv_tourists,
                pv_excursionists=inputs.pv_excursionists,
                cv_weather=inputs.cv_weather,
                i_u_tourists_food=inputs.i_u_tourists_food,
                i_u_excursionists_food=inputs.i_u_excursionists_food,
                i_xa_visitors_food=inputs.i_xa_visitors_food,
                i_xo_visitors_food=inputs.i_xo_visitors_food,
                i_c_food=inputs.i_c_food,
            )
        )

        self.constraints = [
            parking.constraint,
            beach.constraint,
            accommodation.constraint,
            food.constraint,
        ]

        return (
            MolvenoModel.Outputs(usage_indexes=[c.usage for c in self.constraints]),
            MolvenoModel.Expose(
                parking=parking.outputs,
                beach=beach.outputs,
                accommodation=accommodation.outputs,
                food=food.outputs,
            ),
        )


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------


def compute_sustainable_area(field: np.ndarray, tt: np.ndarray, ee: np.ndarray) -> float:
    """Compute the sustainable area under the sustainability field.

    Parameters
    ----------
    field : np.ndarray
        Sustainability field of shape ``(N_t, N_e)``.
    tt : np.ndarray
        Tourist parameter axis (1-D, shape ``(N_t,)``).
    ee : np.ndarray
        Excursionist parameter axis (1-D, shape ``(N_e,)``).

    Returns
    -------
    float
        Integral approximation of the sustainable area.
    """
    return field.sum() * functools.reduce(
        lambda x, y: x * y,
        [axis.max() / (axis.size - 1) + 1 for axis in (tt, ee)],
    )


def compute_sustainability_index_with_ci(
    field: np.ndarray,
    tt: np.ndarray,
    ee: np.ndarray,
    presences: list,
    confidence: float = 0.9,
) -> tuple[float, float]:
    """Return the sustainability index and its confidence half-width.

    Parameters
    ----------
    field : np.ndarray
        Sustainability field of shape ``(N_t, N_e)``.
    tt : np.ndarray
        Tourist parameter axis (1-D, shape ``(N_t,)``).
    ee : np.ndarray
        Excursionist parameter axis (1-D, shape ``(N_e,)``).
    presences : list
        List of ``(tourist, excursionist)`` presence pairs.
    confidence : float, optional
        Confidence level for the interval (default 0.9).

    Returns
    -------
    tuple[float, float]
        ``(mean_index, ci_half_width)``.
    """
    index = interpolate.interpn((tt, ee), field, np.array(presences), bounds_error=False, fill_value=0.0)
    m, se = np.mean(index), stats.sem(index)
    h = se * stats.t.ppf((1 + confidence) / 2.0, index.size - 1)
    return float(m), float(h)


def compute_sustainability_by_constraint(
    field_elements: dict,
    tt: np.ndarray,
    ee: np.ndarray,
    presences: list,
    confidence: float = 0.9,
) -> dict[str, tuple[float, float]]:
    """Return (sustainability_index, CI_half_width) per constraint name.

    Parameters
    ----------
    field_elements : dict
        Mapping of constraint name (str) to per-constraint field array ``(N_t, N_e)``.
    tt : np.ndarray
        Tourist parameter axis (1-D, shape ``(N_t,)``).
    ee : np.ndarray
        Excursionist parameter axis (1-D, shape ``(N_e,)``).
    presences : list
        List of ``(tourist, excursionist)`` presence pairs.
    confidence : float, optional
        Confidence level for the interval (default 0.9).

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping of constraint name to ``(mean_index, ci_half_width)``.
    """
    result = {}
    for key, fe in field_elements.items():
        name = key
        index = interpolate.interpn((tt, ee), fe, np.array(presences), bounds_error=False, fill_value=0.0)
        m, se = np.mean(index), stats.sem(index)
        h = se * stats.t.ppf((1 + confidence) / 2.0, index.size - 1)
        result[name] = (float(m), float(h))
    return result


def compute_modal_lines(
    field_elements: dict,
    tt: np.ndarray,
    ee: np.ndarray,
) -> dict[str, tuple[tuple, tuple]]:
    """Compute the modal line per constraint via orthogonal regression (first PC).

    Parameters
    ----------
    field_elements : dict
        Mapping of constraint name (str) to per-constraint field array ``(N_t, N_e)``.
    tt : np.ndarray
        Tourist parameter axis (1-D, shape ``(N_t,)``).
    ee : np.ndarray
        Excursionist parameter axis (1-D, shape ``(N_e,)``).

    Returns
    -------
    dict[str, tuple[tuple, tuple]]
        Mapping of constraint name to ``((t0, t1), (e0, e1))`` line endpoints.
    """
    bounds = [tt.max(), ee.max()]
    modal_lines = {}
    for key, fe in field_elements.items():
        name = key
        matrix = (fe <= 0.5) & (
            (ndimage.shift(fe, (0, 1)) > 0.5)
            | (ndimage.shift(fe, (0, -1)) > 0.5)
            | (ndimage.shift(fe, (1, 0)) > 0.5)
            | (ndimage.shift(fe, (-1, 0)) > 0.5)
        )
        yi, xi = np.nonzero(matrix)
        if len(yi) < 3:
            continue
        pts = np.stack([tt[yi], ee[xi]], axis=1)
        centroid = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - centroid, full_matrices=False)
        direction = Vt[0]
        t_lo, t_hi = -np.inf, np.inf
        for i, bound in enumerate(bounds):
            if abs(direction[i]) > 1e-10:
                ta = -centroid[i] / direction[i]
                tb = (bound - centroid[i]) / direction[i]
                t_lo = max(t_lo, min(ta, tb))
                t_hi = min(t_hi, max(ta, tb))
        if t_lo >= t_hi:
            continue
        p0 = centroid + t_lo * direction
        p1 = centroid + t_hi * direction
        modal_lines[name] = ((p0[0], p1[0]), (p0[1], p1[1]))
    return modal_lines


# ---------------------------------------------------------------------------
# Field helpers
# ---------------------------------------------------------------------------


def _presence_transformation(
    presence: float,
    reduction_factor: float,
    saturation_level: float,
    sharpness: int = 3,
) -> float:
    """Apply the presence saturation transformation used for scatter-plot samples.

    Parameters
    ----------
    presence : float
        Raw sampled presence value.
    reduction_factor : float
        Multiplicative reduction factor for the presence.
    saturation_level : float
        Saturation level; controls where the curve bends.
    sharpness : int, optional
        Controls the steepness of the saturation curve (default 3).

    Returns
    -------
    float
        Transformed presence value.
    """
    tmp = presence * reduction_factor
    return tmp * saturation_level / ((tmp**sharpness + saturation_level**sharpness) ** (1 / sharpness))


def compute_sustainability_field(
    model: MolvenoModel,
    result: Any,  # EvaluationResult
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute the sustainability field and per-constraint field elements.

    Parameters
    ----------
    model : MolvenoModel
        The model whose constraints define the field.
    result : EvaluationResult
        The raw evaluation result from Evaluation.evaluate().

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        ``(field, field_elements)`` where ``field`` has shape ``(N_t, N_e)``
        and ``field_elements`` maps each constraint name (str) to its
        ``(N_t, N_e)`` component array.
    """
    field = np.ones(
        (
            result.parameter_values[model.inputs.pv_tourists].size,
            result.parameter_values[model.inputs.pv_excursionists].size,
        )
    )
    field_elements: dict = {}
    for c in model.constraints:
        usage = np.broadcast_to(result[c.usage], result.full_shape)
        if isinstance(c.capacity.value, Distribution):
            mask = (1.0 - c.capacity.value.cdf(usage)).astype(float)
        else:
            cap = np.broadcast_to(result[c.capacity], result.full_shape)
            mask = (usage <= cap).astype(float)
        field_elem = np.tensordot(mask, result.weights, axes=([-1], [0]))
        field_elements[c.name] = field_elem
        field *= field_elem
    return field, field_elements


# ---------------------------------------------------------------------------
# MolvenoOutput
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class MolvenoOutput(ModelOutput):
    """Evaluation output for the Molveno overtourism model.

    Carries the sustainability field (and per-constraint field elements) computed
    from an :class:`~dt_model.simulation.evaluation.EvaluationResult`, together
    with the parameter axes and presence samples used to produce it.

    Parameters
    ----------
    field : np.ndarray
        Sustainability field of shape ``(N_t, N_e)`` where each entry is
        ``P(all constraints satisfied | tourists=tt[t], excursionists=ee[e])``.
    field_elements : dict
        Per-constraint field arrays ``{name: np.ndarray}``.  Keys are
        constraint name strings in all cases.
    tt : np.ndarray
        Tourist parameter axis (1-D, shape ``(N_t,)``).
    ee : np.ndarray
        Excursionist parameter axis (1-D, shape ``(N_e,)``).
    sample_tourists : list[float]
        Transformed tourist presence samples for scatter-plot overlays.
    sample_excursionists : list[float]
        Transformed excursionist presence samples for scatter-plot overlays.
    """

    field: np.ndarray
    field_elements: dict
    tt: np.ndarray
    ee: np.ndarray
    sample_tourists: list[float]
    sample_excursionists: list[float]
    confidence: float = 0.8

    def __post_init__(self) -> None:
        """Initialise the :class:`ModelOutput` base after dataclass field assignment."""
        super().__init__()

    @functools.cached_property
    def _zip_samples(self) -> list[tuple[float, float]]:
        """Zipped (tourist, excursionist) presence sample pairs."""
        return list(zip(self.sample_tourists, self.sample_excursionists))

    @functools.cached_property
    def sustainable_area(self) -> float:
        """Sustainable area under the sustainability field."""
        return compute_sustainable_area(self.field, self.tt, self.ee)

    @functools.cached_property
    def sustainability_index(self) -> tuple[float, float]:
        """Overall sustainability index and CI half-width at ``self._confidence``."""
        return compute_sustainability_index_with_ci(self.field, self.tt, self.ee, self._zip_samples, self.confidence)

    @functools.cached_property
    def sustainability_by_constraint(self) -> dict[str, tuple[float, float]]:
        """Per-constraint sustainability index and CI half-width."""
        return compute_sustainability_by_constraint(
            self.field_elements, self.tt, self.ee, self._zip_samples, self.confidence
        )

    @functools.cached_property
    def modal_lines(self) -> dict[str, tuple[tuple, tuple]]:
        """Per-constraint modal lines as ``((t0, t1), (e0, e1))`` coordinate pairs."""
        return compute_modal_lines(self.field_elements, self.tt, self.ee)

    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot including derived sustainability metrics.

        Extends the base :meth:`~dt_model.simulation.runner.ModelOutput.to_snapshot`
        with the four derived properties that the frontend needs but that are
        not stored in the checkpoint:

        ``"sustainable_area"``
            Scalar fraction of the parameter space that is sustainable.
        ``"sustainability_index"``
            ``{"value": float, "ci": float}`` — overall sustainability index
            and half-width of the confidence interval.
        ``"sustainability_by_constraint"``
            ``{name: {"value": float, "ci": float}}`` per-constraint.
        ``"modal_lines"``
            ``{name: {"t": [t0, t1], "e": [e0, e1]}}`` per-constraint
            orthogonal-regression modal lines.

        Returns
        -------
        dict[str, Any]
            Snapshot dict including all base fields plus the above entries.
        """
        d = super().to_snapshot()
        d["sustainable_area"] = float(self.sustainable_area)
        idx, ci = self.sustainability_index
        d["sustainability_index"] = {"value": float(idx), "ci": float(ci)}
        d["sustainability_by_constraint"] = {
            k: {"value": float(v), "ci": float(c)} for k, (v, c) in self.sustainability_by_constraint.items()
        }
        d["modal_lines"] = {
            k: {"t": list(t_coords), "e": list(e_coords)} for k, (t_coords, e_coords) in self.modal_lines.items()
        }
        return d


# ---------------------------------------------------------------------------
# MolvenoEvaluator
# ---------------------------------------------------------------------------


class MolvenoEvaluator(ModelEvaluator[MolvenoModel, MolvenoOutput]):
    """Evaluator for the Molveno overtourism model.

    Implements the :class:`~dt_model.simulation.runner.ModelEvaluator` protocol
    for :class:`~overtourism_molveno.molveno_model.MolvenoModel`, producing a
    :class:`MolvenoOutput` that carries the sustainability field and a resume
    payload.

    Parameters
    ----------
    model : MolvenoModel
        The model instance to evaluate.
    t_max : int, optional
        Maximum tourist presence value on the parameter grid (default 10000).
    e_max : int, optional
        Maximum excursionist presence value on the parameter grid (default 10000).
    t_sample : int, optional
        Number of intervals along the tourist axis; grid has ``t_sample + 1``
        points (default 100).
    e_sample : int, optional
        Number of intervals along the excursionist axis; grid has ``e_sample + 1``
        points (default 100).
    target_presence_samples : int, optional
        Number of presence samples drawn for scatter-plot overlays (default 200).
    """

    def __init__(
        self,
        model: MolvenoModel,
        *,
        t_max: int = 10000,
        e_max: int = 10000,
        t_sample: int = 100,
        e_sample: int = 100,
        target_presence_samples: int = 200,
    ) -> None:
        super().__init__(model)
        self._t_max = t_max
        self._e_max = e_max
        self._t_sample = t_sample
        self._e_sample = e_sample
        self._target_presence_samples = target_presence_samples

    def _pre_compute(self, config: EvaluationConfig) -> tuple[np.ndarray, np.ndarray, dict]:
        """Pre-compute parameter axes and presence samples (no result dependency).

        Used by both :meth:`evaluate` and :meth:`run_async` to share the
        synchronous setup work.

        Parameters
        ----------
        config : EvaluationConfig
            Evaluation parameters; ``ensemble_size`` controls the cross-product
            size for the sampling ensemble.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, dict]
            ``(tt, ee, pv_samples)`` where ``tt`` and ``ee`` are the parameter
            axes and ``pv_samples`` maps each presence index to its samples.
        """
        model = self._model
        pvs = [model.inputs.pv_tourists, model.inputs.pv_excursionists]
        tt = np.linspace(0, self._t_max, self._t_sample + 1)
        ee = np.linspace(0, self._e_max, self._e_sample + 1)
        sampling_scenario = Scenario(model, parameter_axes=pvs)
        sampling_ensemble = CrossProductEnsemble(
            sampling_scenario,
            max_categorical_size=config.ensemble_size,
        )
        pv_samples = sample_across(sampling_ensemble, pvs, total=self._target_presence_samples)
        return tt, ee, pv_samples

    def _build_output(
        self,
        result: EvaluationResult,
        tt: np.ndarray,
        ee: np.ndarray,
        pv_samples: dict,
    ) -> MolvenoOutput:
        """Build a :class:`MolvenoOutput` from an evaluated result and pre-computed axes.

        Computes the sustainability field, transforms presence samples, constructs
        the output, and attaches the resume payload via :meth:`attach_resume`.

        Parameters
        ----------
        result : EvaluationResult
            The raw engine result.
        tt : np.ndarray
            Tourist parameter axis.
        ee : np.ndarray
            Excursionist parameter axis.
        pv_samples : dict
            Pre-computed presence samples from :meth:`_pre_compute`.

        Returns
        -------
        MolvenoOutput
            Fully populated output with resume payload attached.
        """
        model = self._model
        field, field_elements = compute_sustainability_field(model, result)
        rf_t = float(np.mean(result[model.inputs.i_p_tourists_reduction_factor]))
        sl_t = float(np.mean(result[model.inputs.i_p_tourists_saturation_level]))
        rf_e = float(np.mean(result[model.inputs.i_p_excursionists_reduction_factor]))
        sl_e = float(np.mean(result[model.inputs.i_p_excursionists_saturation_level]))
        sample_tourists = [_presence_transformation(s, rf_t, sl_t) for s in pv_samples[model.inputs.pv_tourists]]
        sample_excursionists = [
            _presence_transformation(s, rf_e, sl_e) for s in pv_samples[model.inputs.pv_excursionists]
        ]
        output = MolvenoOutput(
            field=field,
            field_elements=field_elements,
            tt=tt,
            ee=ee,
            sample_tourists=sample_tourists,
            sample_excursionists=sample_excursionists,
        )
        self.attach_resume(output, result)
        return output

    def evaluate(self, scenario: Any, config: EvaluationConfig) -> MolvenoOutput:
        """Run a blocking evaluation and return a :class:`MolvenoOutput`.

        Steps:

        1. Pre-compute the ``(t_sample+1) × (e_sample+1)`` parameter axes and
           presence samples via :meth:`_pre_compute`.
        2. Build a :class:`~dt_model.CrossProductEnsemble` for the categorical
           dimension (context variables × their probability weights).
        3. Run :class:`~dt_model.Evaluation` over the full 2-D parameter grid.
        4. Compute the sustainability field, transform presence samples, and
           attach the resume payload via :meth:`_build_output`.

        Parameters
        ----------
        scenario : Scenario
            The scenario to evaluate, optionally carrying value overrides.
        config : EvaluationConfig
            Evaluation parameters; ``config.ensemble_size`` controls the
            maximum categorical cross-product size passed to
            :class:`~dt_model.CrossProductEnsemble`.

        Returns
        -------
        MolvenoOutput
            Contains the sustainability field, per-constraint field elements,
            parameter axes, and a resume payload.
        """
        model = self._model
        tt, ee, pv_samples = self._pre_compute(config)
        ensemble = CrossProductEnsemble(scenario, max_categorical_size=config.ensemble_size)
        result = Evaluation(scenario).evaluate(
            ensemble=ensemble,
            parameters={model.inputs.pv_tourists: tt, model.inputs.pv_excursionists: ee},
        )
        return self._build_output(result, tt, ee, pv_samples)

    def run_async(self, scenario: Any, config: EvaluationConfig) -> ModelRunHandle[MolvenoOutput]:
        """Submit an engine-level async evaluation and return a handle immediately.

        Pre-computes everything that does not depend on the evaluation result
        (parameter grids, ensembles, presence samples) synchronously on the
        calling thread, then submits only the
        :meth:`~dt_model.Evaluation.evaluate` call to the shared
        :func:`~dt_model.simulation.handle._get_default_executor` thread
        pool.  The :class:`~dt_model.simulation.runner.ModelRunHandle`
        post-processor closure completes the rest of the work once the result
        is available.

        This matches Bologna's tier-3 pattern: the future holds a
        :class:`~dt_model.EvaluationResult`, satisfying
        :class:`~dt_model.simulation.runner.ModelRunHandle`'s type contract.

        Parameters
        ----------
        scenario : Scenario
            The scenario to evaluate.
        config : EvaluationConfig
            Evaluation parameters.

        Returns
        -------
        ModelRunHandle[MolvenoOutput]
            Handle whose :meth:`~dt_model.simulation.runner.ModelRunHandle.get`
            returns a :class:`MolvenoOutput`.
        """
        tt, ee, pv_samples = self._pre_compute(config)
        ensemble = CrossProductEnsemble(scenario, max_categorical_size=config.ensemble_size)
        future = _get_default_executor().submit(
            Evaluation(scenario).evaluate,
            ensemble=ensemble,
            parameters={self._model.inputs.pv_tourists: tt, self._model.inputs.pv_excursionists: ee},
        )

        def _post(result: EvaluationResult) -> MolvenoOutput:
            return self._build_output(result, tt, ee, pv_samples)

        return ModelRunHandle(future, _post)

    def input_schema(self) -> dict[str, dict[str, Any]]:
        """Return a schema dict describing the Molveno model's tunable indexes.

        Includes entries for the three categorical context variables
        (``cv_weekday``, ``cv_season``, ``cv_weather``) and the four
        capacity distribution parameters.

        Returns
        -------
        dict[str, dict[str, Any]]
            Maps each index name to a metadata dict describing its type and,
            for categoricals, its full support.

        Examples
        --------
        >>> evaluator.input_schema()
        {"weekday": {"type": "categorical", "support": [...]}, ...}
        """
        model = self._model
        schema: dict[str, dict[str, Any]] = {}
        for idx in model.inputs:
            if isinstance(idx, CategoricalIndex):
                schema[idx.name] = {"type": "categorical", "support": list(idx.support)}
            elif isinstance(idx, DistributionIndex):
                schema[idx.name] = {"type": "distribution"}
        return schema
