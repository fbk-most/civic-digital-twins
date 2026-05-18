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
    NumpyBackend,
    graph,
    sample_across,
)
from civic_digital_twins.dt_model.model.index import Distribution
from civic_digital_twins.dt_model.simulation.evaluation import _get_default_executor
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


class ParkingModel(Model):
    """Concern sub-model — parking usage.

    All parameters (usage factors, conversion factors, capacity) are
    received as ``Inputs`` so that callers can override any default.
    :class:`MolvenoModel` creates the indexes with their default values and
    passes them in.

    The usage formula ``i_u_parking`` is the single contractual ``Output``.
    The :class:`~overtourism_molveno.molveno_model.Constraint` is
    stored as a plain instance attribute ``self.constraint``.

    Parameters
    ----------
    pv_tourists : ConditionalDistributionIndex
        Tourist presence (wired from :class:`MolvenoModel`).
    pv_excursionists : ConditionalDistributionIndex
        Excursionist presence (wired from :class:`MolvenoModel`).
    cv_weather : CategoricalIndex
        Weather context variable (needed for the piecewise usage factor).
    i_u_tourists_parking : Index
        Tourist parking usage factor.
    i_u_excursionists_parking : Index
        Excursionist parking usage factor (piecewise on weather).
    i_xa_tourists_per_vehicle : Index
        Tourists per vehicle allocation factor.
    i_xa_excursionists_per_vehicle : Index
        Excursionists per vehicle allocation factor.
    i_xo_tourists_parking : Index
        Tourists in parking rotation factor.
    i_xo_excursionists_parking : Index
        Excursionists in parking rotation factor.
    i_c_parking : DistributionIndex
        Parking capacity (uncertain).

    Attributes
    ----------
    constraint : Constraint
        The parking constraint (usage / capacity pair).
    """

    @dataclass
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

    @dataclass
    class Outputs:
        """Contractual outputs of :class:`ParkingModel`."""

        i_u_parking: Index

    def __init__(
        self,
        pv_tourists: ConditionalDistributionIndex,
        pv_excursionists: ConditionalDistributionIndex,
        cv_weather: CategoricalIndex,
        i_u_tourists_parking: Index,
        i_u_excursionists_parking: Index,
        i_xa_tourists_per_vehicle: Index,
        i_xa_excursionists_per_vehicle: Index,
        i_xo_tourists_parking: Index,
        i_xo_excursionists_parking: Index,
        i_c_parking: DistributionIndex,
    ) -> None:
        Inputs = ParkingModel.Inputs
        Outputs = ParkingModel.Outputs

        inputs = Inputs(
            pv_tourists=pv_tourists,
            pv_excursionists=pv_excursionists,
            cv_weather=cv_weather,
            i_u_tourists_parking=i_u_tourists_parking,
            i_u_excursionists_parking=i_u_excursionists_parking,
            i_xa_tourists_per_vehicle=i_xa_tourists_per_vehicle,
            i_xa_excursionists_per_vehicle=i_xa_excursionists_per_vehicle,
            i_xo_tourists_parking=i_xo_tourists_parking,
            i_xo_excursionists_parking=i_xo_excursionists_parking,
            i_c_parking=i_c_parking,
        )

        i_u_parking = Index(
            "parking usage",
            inputs.pv_tourists
            * inputs.i_u_tourists_parking
            / (inputs.i_xa_tourists_per_vehicle * inputs.i_xo_tourists_parking)
            + inputs.pv_excursionists
            * inputs.i_u_excursionists_parking
            / (inputs.i_xa_excursionists_per_vehicle * inputs.i_xo_excursionists_parking),
        )

        super().__init__(
            "Parking",
            inputs=inputs,
            outputs=Outputs(i_u_parking=i_u_parking),
        )

        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(name="parking", usage=i_u_parking, capacity=inputs.i_c_parking)


# ---------------------------------------------------------------------------
# BeachModel
# ---------------------------------------------------------------------------


class BeachModel(Model):
    """Concern sub-model — beach usage.

    All parameters (usage factors, rotation factors, capacity) are received
    as ``Inputs``.  The uncertain rotation factor ``i_xo_tourists_beach`` is
    passed in from :class:`MolvenoModel` so it appears in the root
    ``model.indexes`` and is sampled by
    :class:`~dt_model.CrossProductEnsemble`.

    Parameters
    ----------
    pv_tourists : ConditionalDistributionIndex
        Tourist presence (wired from :class:`MolvenoModel`).
    pv_excursionists : ConditionalDistributionIndex
        Excursionist presence (wired from :class:`MolvenoModel`).
    cv_weather : CategoricalIndex
        Weather context variable (needed for the piecewise usage factors).
    i_u_tourists_beach : Index
        Tourist beach usage factor (piecewise on weather).
    i_u_excursionists_beach : Index
        Excursionist beach usage factor (piecewise on weather).
    i_xo_tourists_beach : DistributionIndex
        Tourists on beach rotation factor (uncertain).
    i_xo_excursionists_beach : Index
        Excursionists on beach rotation factor.
    i_c_beach : DistributionIndex
        Beach capacity (uncertain).

    Attributes
    ----------
    constraint : Constraint
        The beach constraint (usage / capacity pair).
    """

    @dataclass
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

    @dataclass
    class Outputs:
        """Contractual outputs of :class:`BeachModel`."""

        i_u_beach: Index

    def __init__(
        self,
        pv_tourists: ConditionalDistributionIndex,
        pv_excursionists: ConditionalDistributionIndex,
        cv_weather: CategoricalIndex,
        i_u_tourists_beach: Index,
        i_u_excursionists_beach: Index,
        i_xo_tourists_beach: DistributionIndex,
        i_xo_excursionists_beach: Index,
        i_c_beach: DistributionIndex,
    ) -> None:
        Inputs = BeachModel.Inputs
        Outputs = BeachModel.Outputs

        inputs = Inputs(
            pv_tourists=pv_tourists,
            pv_excursionists=pv_excursionists,
            cv_weather=cv_weather,
            i_u_tourists_beach=i_u_tourists_beach,
            i_u_excursionists_beach=i_u_excursionists_beach,
            i_xo_tourists_beach=i_xo_tourists_beach,
            i_xo_excursionists_beach=i_xo_excursionists_beach,
            i_c_beach=i_c_beach,
        )

        i_u_beach = Index(
            "beach usage",
            inputs.pv_tourists * inputs.i_u_tourists_beach / inputs.i_xo_tourists_beach
            + inputs.pv_excursionists * inputs.i_u_excursionists_beach / inputs.i_xo_excursionists_beach,
        )

        super().__init__(
            "Beach",
            inputs=inputs,
            outputs=Outputs(i_u_beach=i_u_beach),
        )

        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(name="beach", usage=i_u_beach, capacity=inputs.i_c_beach)


# ---------------------------------------------------------------------------
# AccommodationModel
# ---------------------------------------------------------------------------


class AccommodationModel(Model):
    """Concern sub-model — accommodation usage.

    Parameters
    ----------
    pv_tourists : ConditionalDistributionIndex
        Tourist presence (wired from :class:`MolvenoModel`).
    i_u_tourists_accommodation : Index
        Tourist accommodation usage factor.
    i_xa_tourists_accommodation : Index
        Tourists per accommodation allocation factor.
    i_c_accommodation : DistributionIndex
        Accommodation capacity (uncertain).

    Attributes
    ----------
    constraint : Constraint
        The accommodation constraint (usage / capacity pair).
    """

    @dataclass
    class Inputs:
        """Contractual inputs of :class:`AccommodationModel`."""

        pv_tourists: ConditionalDistributionIndex
        i_u_tourists_accommodation: Index
        i_xa_tourists_accommodation: Index
        i_c_accommodation: DistributionIndex

    @dataclass
    class Outputs:
        """Contractual outputs of :class:`AccommodationModel`."""

        i_u_accommodation: Index

    def __init__(
        self,
        pv_tourists: ConditionalDistributionIndex,
        i_u_tourists_accommodation: Index,
        i_xa_tourists_accommodation: Index,
        i_c_accommodation: DistributionIndex,
    ) -> None:
        Inputs = AccommodationModel.Inputs
        Outputs = AccommodationModel.Outputs

        inputs = Inputs(
            pv_tourists=pv_tourists,
            i_u_tourists_accommodation=i_u_tourists_accommodation,
            i_xa_tourists_accommodation=i_xa_tourists_accommodation,
            i_c_accommodation=i_c_accommodation,
        )

        i_u_accommodation = Index(
            "accommodation usage",
            inputs.pv_tourists * inputs.i_u_tourists_accommodation / inputs.i_xa_tourists_accommodation,
        )

        super().__init__(
            "Accommodation",
            inputs=inputs,
            outputs=Outputs(i_u_accommodation=i_u_accommodation),
        )

        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(
            name="accommodation",
            usage=i_u_accommodation,
            capacity=inputs.i_c_accommodation,
        )


# ---------------------------------------------------------------------------
# FoodModel
# ---------------------------------------------------------------------------


class FoodModel(Model):
    """Concern sub-model — food-service usage.

    Parameters
    ----------
    pv_tourists : ConditionalDistributionIndex
        Tourist presence (wired from :class:`MolvenoModel`).
    pv_excursionists : ConditionalDistributionIndex
        Excursionist presence (wired from :class:`MolvenoModel`).
    cv_weather : CategoricalIndex
        Weather context variable (needed for the piecewise usage factor).
    i_u_tourists_food : Index
        Tourist food-service usage factor.
    i_u_excursionists_food : Index
        Excursionist food-service usage factor (piecewise on weather).
    i_xa_visitors_food : Index
        Visitors in food-service allocation factor.
    i_xo_visitors_food : Index
        Visitors in food-service rotation factor.
    i_c_food : DistributionIndex
        Food-service capacity (uncertain).

    Attributes
    ----------
    constraint : Constraint
        The food-service constraint (usage / capacity pair).
    """

    @dataclass
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

    @dataclass
    class Outputs:
        """Contractual outputs of :class:`FoodModel`."""

        i_u_food: Index

    def __init__(
        self,
        pv_tourists: ConditionalDistributionIndex,
        pv_excursionists: ConditionalDistributionIndex,
        cv_weather: CategoricalIndex,
        i_u_tourists_food: Index,
        i_u_excursionists_food: Index,
        i_xa_visitors_food: Index,
        i_xo_visitors_food: Index,
        i_c_food: DistributionIndex,
    ) -> None:
        Inputs = FoodModel.Inputs
        Outputs = FoodModel.Outputs

        inputs = Inputs(
            pv_tourists=pv_tourists,
            pv_excursionists=pv_excursionists,
            cv_weather=cv_weather,
            i_u_tourists_food=i_u_tourists_food,
            i_u_excursionists_food=i_u_excursionists_food,
            i_xa_visitors_food=i_xa_visitors_food,
            i_xo_visitors_food=i_xo_visitors_food,
            i_c_food=i_c_food,
        )

        i_u_food = Index(
            "food usage",
            (inputs.pv_tourists * inputs.i_u_tourists_food + inputs.pv_excursionists * inputs.i_u_excursionists_food)
            / (inputs.i_xa_visitors_food * inputs.i_xo_visitors_food),
        )

        super().__init__(
            "Food",
            inputs=inputs,
            outputs=Outputs(i_u_food=i_u_food),
        )

        # Constraint stored as a plain attribute — not a GenericIndex.
        self.constraint = Constraint(name="food", usage=i_u_food, capacity=inputs.i_c_food)


# ---------------------------------------------------------------------------
# MolvenoModel  (root)
# ---------------------------------------------------------------------------


class MolvenoModel(Model):
    """Root overtourism model that wires the four concern sub-models.

    ``MolvenoModel`` owns:

    * the three context variables (``cv_weekday``, ``cv_season``, ``cv_weather``);
    * the two presence variables (``pv_tourists``, ``pv_excursionists``);
    * the default values for every ``i_*`` parameter.

    Callers who need to override a parameter can subclass ``MolvenoModel``
    or construct the concern sub-models directly with different values.

    The ``cvs``, ``pvs``, and ``constraints`` attributes are required by
    :class:`~dt_model.CrossProductEnsemble`.

    CVs, PVs, and sub-models are accessible as named attributes::

        m = MolvenoModel()
        m.cv_weather                      # CategoricalIndex
        m.pv_tourists                     # ConditionalDistributionIndex
        m.parking.inputs.i_c_parking      # capacity DistributionIndex
        m.beach.inputs.i_xo_tourists_beach  # rotation DistributionIndex
        m.parking.outputs.i_u_parking     # usage formula Index
        m.parking.constraint              # Constraint object
    """

    @dataclass
    class Inputs:
        """Contractual inputs of :class:`MolvenoModel`."""

        cvs: list[CategoricalIndex]
        pvs: list[ConditionalDistributionIndex]
        domain_indexes: list[GenericIndex]
        capacities: list[GenericIndex]

    @dataclass
    class Outputs:
        """Contractual outputs of :class:`MolvenoModel`."""

        usage_indexes: list[GenericIndex]

    def __init__(self) -> None:
        # ------------------------------------------------------------------
        # Stage 1 — context and presence variables
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Default i_* parameters — created here so callers can override them
        # ------------------------------------------------------------------

        # Parking parameters
        i_u_tourists_parking = Index("tourist parking usage factor", 0.02)
        i_u_excursionists_parking = Index(
            "excursionist parking usage factor",
            graph.piecewise((0.55, cv_weather == "bad"), (0.80, True)),
        )
        i_xa_tourists_per_vehicle = Index("tourists per vehicle allocation factor", 2.5)
        i_xa_excursionists_per_vehicle = Index("excursionists per vehicle allocation factor", 2.5)
        i_xo_tourists_parking = Index("tourists in parking rotation factor", 1.02)
        i_xo_excursionists_parking = Index("excursionists in parking rotation factor", 3.5)
        i_c_parking = DistributionIndex("parking capacity", stats.uniform, {"loc": 350.0, "scale": 100.0})

        # Beach parameters
        i_u_tourists_beach = Index(
            "tourist beach usage factor",
            graph.piecewise((0.25, cv_weather == "bad"), (0.50, True)),
        )
        i_u_excursionists_beach = Index(
            "excursionist beach usage factor",
            graph.piecewise((0.35, cv_weather == "bad"), (0.80, True)),
        )
        i_xo_tourists_beach = DistributionIndex(
            "tourists on beach rotation factor",
            stats.uniform,
            {"loc": 1.0, "scale": 2.0},
        )
        i_xo_excursionists_beach = Index("excursionists on beach rotation factor", 1.02)
        i_c_beach = DistributionIndex("beach capacity", stats.uniform, {"loc": 6000.0, "scale": 1000.0})

        # Accommodation parameters
        i_u_tourists_accommodation = Index("tourist accommodation usage factor", 0.90)
        i_xa_tourists_accommodation = Index("tourists per accommodation allocation factor", 1.05)
        i_c_accommodation = DistributionIndex(
            "accommodation capacity",
            stats.lognorm,
            {"s": 0.125, "loc": 0.0, "scale": 5000.0},
        )

        # Food parameters
        i_u_tourists_food = Index("tourist food service usage factor", 0.20)
        i_u_excursionists_food = Index(
            "excursionist food service usage factor",
            graph.piecewise((0.80, cv_weather == "bad"), (0.40, True)),
        )
        i_xa_visitors_food = Index("visitors in food service allocation factor", 0.9)
        i_xo_visitors_food = Index("visitors in food service rotation factor", 2.0)
        i_c_food = DistributionIndex(
            "food service capacity",
            stats.triang,
            {"loc": 3000.0, "scale": 1000.0, "c": 0.5},
        )

        # Presence-transformation parameters (used in overtourism_molveno.py)
        i_p_tourists_reduction_factor = Index("tourists reduction factor", 1.0)
        i_p_excursionists_reduction_factor = Index("excursionists reduction factor", 1.0)
        i_p_tourists_saturation_level = Index("tourists saturation level", 10000)
        i_p_excursionists_saturation_level = Index("excursionists saturation level", 10000)

        # ------------------------------------------------------------------
        # Stage 2 / 3 — concern sub-models
        # ------------------------------------------------------------------
        parking = ParkingModel(
            pv_tourists=pv_tourists,
            pv_excursionists=pv_excursionists,
            cv_weather=cv_weather,
            i_u_tourists_parking=i_u_tourists_parking,
            i_u_excursionists_parking=i_u_excursionists_parking,
            i_xa_tourists_per_vehicle=i_xa_tourists_per_vehicle,
            i_xa_excursionists_per_vehicle=i_xa_excursionists_per_vehicle,
            i_xo_tourists_parking=i_xo_tourists_parking,
            i_xo_excursionists_parking=i_xo_excursionists_parking,
            i_c_parking=i_c_parking,
        )
        beach = BeachModel(
            pv_tourists=pv_tourists,
            pv_excursionists=pv_excursionists,
            cv_weather=cv_weather,
            i_u_tourists_beach=i_u_tourists_beach,
            i_u_excursionists_beach=i_u_excursionists_beach,
            i_xo_tourists_beach=i_xo_tourists_beach,
            i_xo_excursionists_beach=i_xo_excursionists_beach,
            i_c_beach=i_c_beach,
        )
        accommodation = AccommodationModel(
            pv_tourists=pv_tourists,
            i_u_tourists_accommodation=i_u_tourists_accommodation,
            i_xa_tourists_accommodation=i_xa_tourists_accommodation,
            i_c_accommodation=i_c_accommodation,
        )
        food = FoodModel(
            pv_tourists=pv_tourists,
            pv_excursionists=pv_excursionists,
            cv_weather=cv_weather,
            i_u_tourists_food=i_u_tourists_food,
            i_u_excursionists_food=i_u_excursionists_food,
            i_xa_visitors_food=i_xa_visitors_food,
            i_xo_visitors_food=i_xo_visitors_food,
            i_c_food=i_c_food,
        )

        # ------------------------------------------------------------------
        # Collect domain lists consumed by CrossProductEnsemble
        # ------------------------------------------------------------------
        cvs: list[CategoricalIndex] = [cv_weekday, cv_season, cv_weather]
        pvs: list[ConditionalDistributionIndex] = [pv_tourists, pv_excursionists]
        constraints = [
            parking.constraint,
            beach.constraint,
            accommodation.constraint,
            food.constraint,
        ]
        capacities: list[GenericIndex] = [i_c_parking, i_c_beach, i_c_accommodation, i_c_food]

        # Collect and deduplicate all indexes from sub-models plus the root
        # presence-transformation parameters.  Identity-based deduplication
        # ensures shared indexes (pv_*, cv_*) are not registered twice.
        seen: set[int] = set()
        all_indexes: list[GenericIndex] = []
        for idx in (
            list(parking.indexes)
            + list(beach.indexes)
            + list(accommodation.indexes)
            + list(food.indexes)
            + [
                i_p_tourists_reduction_factor,
                i_p_excursionists_reduction_factor,
                i_p_tourists_saturation_level,
                i_p_excursionists_saturation_level,
            ]
        ):
            if id(idx) not in seen:
                seen.add(id(idx))
                all_indexes.append(idx)

        # domain_indexes: everything that is not a CV, PV, capacity, or usage-formula index.
        cv_pv_ids = {id(x) for x in cvs + pvs}
        cap_ids = {id(x) for x in capacities}
        usage_ids = {id(c.usage) for c in constraints}
        domain_indexes: list[GenericIndex] = [
            idx
            for idx in all_indexes
            if id(idx) not in cv_pv_ids and id(idx) not in cap_ids and id(idx) not in usage_ids
        ]

        # ------------------------------------------------------------------
        # Initialise Model with the declarative Inputs/Outputs API
        # ------------------------------------------------------------------
        Inputs = MolvenoModel.Inputs
        Outputs = MolvenoModel.Outputs
        super().__init__(
            "base model",
            inputs=Inputs(
                cvs=cvs,
                pvs=pvs,
                domain_indexes=domain_indexes,
                capacities=capacities,
            ),
            outputs=Outputs(usage_indexes=[c.usage for c in constraints]),
        )

        self.cvs = cvs
        self.pvs = pvs
        self.domain_indexes = domain_indexes
        self.capacities = capacities
        self.constraints = constraints

        # ------------------------------------------------------------------
        # Attach CVs, PVs, and sub-models as named attributes
        # ------------------------------------------------------------------
        self.cv_weekday = cv_weekday
        self.cv_season = cv_season
        self.cv_weather = cv_weather
        self.pv_tourists = pv_tourists
        self.pv_excursionists = pv_excursionists

        self.parking = parking
        self.beach = beach
        self.accommodation = accommodation
        self.food = food

        self.i_p_tourists_reduction_factor = i_p_tourists_reduction_factor
        self.i_p_excursionists_reduction_factor = i_p_excursionists_reduction_factor
        self.i_p_tourists_saturation_level = i_p_tourists_saturation_level
        self.i_p_excursionists_saturation_level = i_p_excursionists_saturation_level


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
    from functools import reduce

    return field.sum() * reduce(
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
        name = key if isinstance(key, str) else key.name
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
        name = key if isinstance(key, str) else key.name
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
            result.parameter_values[model.pv_tourists].size,
            result.parameter_values[model.pv_excursionists].size,
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
        Per-constraint field arrays ``{constraint_key: np.ndarray}``.  Keys are
        :class:`~overtourism_molveno.molveno_model.Constraint` objects when
        produced by :meth:`MolvenoEvaluator.evaluate`, and plain strings (constraint
        names) when loaded via :meth:`from_dict`.
    tt : np.ndarray
        Tourist parameter axis (1-D, shape ``(N_t,)``).
    ee : np.ndarray
        Excursionist parameter axis (1-D, shape ``(N_e,)``).
    sample_tourists : list[float]
        Transformed tourist presence samples for scatter-plot overlays.
    sample_excursionists : list[float]
        Transformed excursionist presence samples for scatter-plot overlays.
    serialized_resume : dict or None, optional
        Encoded resume payload produced by
        :func:`~dt_model.simulation.runner._encode_result`.  When not ``None``
        the output is immediately marked as resumable.
    """

    def __init__(
        self,
        field: np.ndarray,
        field_elements: dict,
        tt: np.ndarray,
        ee: np.ndarray,
        sample_tourists: list[float],
        sample_excursionists: list[float],
        *,
        serialized_resume: dict | None = None,
        confidence: float = 0.8,
    ) -> None:
        super().__init__()
        self._confidence = confidence
        self.field = field
        self.field_elements = field_elements
        self.tt = tt
        self.ee = ee
        self.sample_tourists = sample_tourists
        self.sample_excursionists = sample_excursionists
        self._serialized_resume: dict | None = serialized_resume
        if serialized_resume is not None:
            self._is_resumable = True

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
        return compute_sustainability_index_with_ci(self.field, self.tt, self.ee, self._zip_samples, self._confidence)

    @functools.cached_property
    def sustainability_by_constraint(self) -> dict[str, tuple[float, float]]:
        """Per-constraint sustainability index and CI half-width."""
        return compute_sustainability_by_constraint(
            self.field_elements, self.tt, self.ee, self._zip_samples, self._confidence
        )

    @functools.cached_property
    def modal_lines(self) -> dict[str, tuple[tuple, tuple]]:
        """Per-constraint modal lines as ``((t0, t1), (e0, e1))`` coordinate pairs."""
        return compute_modal_lines(self.field_elements, self.tt, self.ee)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the output to a JSON-serialisable dict.

        Returns a dict containing both the summary layer (field, per-constraint
        field elements, parameter axes) and the resume payload when available.

        Returns
        -------
        dict[str, Any]
            Dict with keys ``"dt_model_version"``, ``"field"``,
            ``"field_elements"``, ``"tt"``, ``"ee"``, and optionally
            ``"_resume"`` (encoded :class:`~dt_model.simulation.evaluation.EvaluationResult`).
        """
        d: dict[str, Any] = {
            "dt_model_version": _get_dt_model_version(),
            "field": _encode_array(self.field),
            "field_elements": {name: _encode_array(arr) for name, arr in self.field_elements.items()},
            "tt": _encode_array(self.tt),
            "ee": _encode_array(self.ee),
            "sample_tourists": list(self.sample_tourists),
            "sample_excursionists": list(self.sample_excursionists),
            "confidence": self._confidence,
        }
        if self._serialized_resume is not None:
            d["_resume"] = self._serialized_resume
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MolvenoOutput":
        """Reconstruct a :class:`MolvenoOutput` from a serialised dict.

        Always restores the summary layer (field, field_elements, tt, ee).
        Restores the resume payload when the ``"_resume"`` key is present,
        setting :attr:`~dt_model.simulation.runner.ModelOutput.is_resumable`
        to ``True``.

        Concrete pattern used::

            obj = cls.__new__(cls)
            ModelOutput.__init__(obj)      # sets _is_resumable = False
            # populate summary fields ...
            if "_resume" in data:
                obj._serialized_resume = data["_resume"]
                obj._is_resumable = True
            return obj

        Parameters
        ----------
        data : dict[str, Any]
            Dict previously produced by :meth:`to_dict`.

        Returns
        -------
        MolvenoOutput
            Reconstructed instance.  ``field_elements`` uses string keys
            (constraint names) rather than :class:`Constraint` objects.
        """
        obj = cls.__new__(cls)
        ModelOutput.__init__(obj)
        obj.field = _decode_array(data["field"])
        obj.field_elements = {name: _decode_array(arr) for name, arr in data["field_elements"].items()}
        obj.tt = _decode_array(data["tt"])
        obj.ee = _decode_array(data["ee"])
        obj.sample_tourists = list(data["sample_tourists"])
        obj.sample_excursionists = list(data["sample_excursionists"])
        obj._confidence = float(data.get("confidence", 0.8))
        obj._serialized_resume = None
        if "_resume" in data:
            obj._serialized_resume = data["_resume"]
            obj._is_resumable = True
        return obj


# ---------------------------------------------------------------------------
# MolvenoEvaluator
# ---------------------------------------------------------------------------


class MolvenoEvaluator(ModelEvaluator[MolvenoOutput]):
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

    def evaluate(self, scenario: Any, config: EvaluationConfig) -> MolvenoOutput:
        """Run a blocking evaluation and return a :class:`MolvenoOutput`.

        Steps:

        1. Extract categorical restrictions from ``scenario.overrides``.
        2. Build a ``(t_sample+1) × (e_sample+1)`` parameter grid.
        3. Build a :class:`~dt_model.CrossProductEnsemble` for the ensemble
           dimension.
        4. Run :class:`~dt_model.Evaluation` over the grid.
        5. Compute the sustainability field and per-constraint elements from
           the raw result (same logic as ``plot_scenario`` in
           ``overtourism_molveno.py``).
        6. Encode the full result as the resume payload.

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
        model: MolvenoModel = self._model  # type: ignore[assignment]

        # 2. Build parameter grid.
        tt = np.linspace(0, self._t_max, self._t_sample + 1)
        ee = np.linspace(0, self._e_max, self._e_sample + 1)

        # 3. Build ensemble.
        # CrossProductEnsemble respects scenario.effective_outcomes() for CategoricalIndex:
        # a dict override restricts the support to its keys and uses its values as weights.
        # String overrides (concrete pins) are handled via base_substitutions() — the CV is
        # excluded from the cross-product axis and injected as a constant.
        ensemble = CrossProductEnsemble(
            scenario,
            max_categorical_size=config.ensemble_size,
            exclude=model.pvs,
        )

        # 4. Evaluate.
        result = Evaluation(scenario).evaluate(
            ensemble=ensemble,
            parameters={model.pv_tourists: tt, model.pv_excursionists: ee},
        )

        # 5. Compute sustainability field.
        field, field_elements = compute_sustainability_field(model, result)

        # 5b. Generate presence samples for scatter-plot overlays.
        # sample_across requires all CV parents in ensemble.assignments().  When a CV is
        # concretely overridden (str), Scenario.abstract_indexes() excludes it, collapsing it
        # out of the ensemble axis.  Use an unrestricted base scenario so every parent CV
        # is always present in assignments.  The scatter dots then represent the marginal
        # presence distribution, agnostic of any single-value restrictions.
        sampling_ensemble = CrossProductEnsemble(
            type(scenario)(model),
            max_categorical_size=config.ensemble_size,
            exclude=model.pvs,
        )
        pv_samples = sample_across(
            sampling_ensemble, [model.pv_tourists, model.pv_excursionists], total=self._target_presence_samples
        )
        rf_t = float(np.mean(result[model.i_p_tourists_reduction_factor]))
        sl_t = float(np.mean(result[model.i_p_tourists_saturation_level]))
        rf_e = float(np.mean(result[model.i_p_excursionists_reduction_factor]))
        sl_e = float(np.mean(result[model.i_p_excursionists_saturation_level]))
        sample_tourists = [_presence_transformation(s, rf_t, sl_t) for s in pv_samples[model.pv_tourists]]
        sample_excursionists = [_presence_transformation(s, rf_e, sl_e) for s in pv_samples[model.pv_excursionists]]

        # 6. Encode resume payload.
        encoded_resume = _encode_result(result, scenario.model.indexes)

        # 7. Return output.
        return MolvenoOutput(
            field=field,
            field_elements=field_elements,
            tt=tt,
            ee=ee,
            sample_tourists=sample_tourists,
            sample_excursionists=sample_excursionists,
            serialized_resume=encoded_resume,
        )

    def run_async(self, scenario: Any, config: EvaluationConfig) -> ModelRunHandle[MolvenoOutput]:
        """Submit an engine-level async evaluation and return a handle immediately.

        Pre-computes everything that does not depend on the evaluation result
        (parameter grids, ensembles, presence samples) synchronously on the
        calling thread, then submits only the
        :meth:`~dt_model.Evaluation.evaluate` call to the shared
        :func:`~dt_model.simulation.evaluation._get_default_executor` thread
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
        model: MolvenoModel = self._model  # type: ignore[assignment]

        # --- synchronous pre-computation (no result dependency) ---
        tt = np.linspace(0, self._t_max, self._t_sample + 1)
        ee = np.linspace(0, self._e_max, self._e_sample + 1)
        ensemble = CrossProductEnsemble(
            scenario,
            max_categorical_size=config.ensemble_size,
            exclude=model.pvs,
        )
        sampling_ensemble = CrossProductEnsemble(
            type(scenario)(model),
            max_categorical_size=config.ensemble_size,
            exclude=model.pvs,
        )
        pv_samples = sample_across(
            sampling_ensemble, [model.pv_tourists, model.pv_excursionists], total=self._target_presence_samples
        )

        # --- async: only the engine evaluation ---
        future = _get_default_executor().submit(
            Evaluation(scenario).evaluate,
            ensemble=ensemble,
            parameters={model.pv_tourists: tt, model.pv_excursionists: ee},
        )

        # --- post-processor closure (runs on .get() / .poll()) ---
        def _post(result: EvaluationResult) -> MolvenoOutput:
            field, field_elements = compute_sustainability_field(model, result)
            rf_t = float(np.mean(result[model.i_p_tourists_reduction_factor]))
            sl_t = float(np.mean(result[model.i_p_tourists_saturation_level]))
            rf_e = float(np.mean(result[model.i_p_excursionists_reduction_factor]))
            sl_e = float(np.mean(result[model.i_p_excursionists_saturation_level]))
            sample_tourists = [_presence_transformation(s, rf_t, sl_t) for s in pv_samples[model.pv_tourists]]
            sample_excursionists = [_presence_transformation(s, rf_e, sl_e) for s in pv_samples[model.pv_excursionists]]
            encoded_resume = _encode_result(result, scenario.model.indexes)
            return MolvenoOutput(
                field=field,
                field_elements=field_elements,
                tt=tt,
                ee=ee,
                sample_tourists=sample_tourists,
                sample_excursionists=sample_excursionists,
                serialized_resume=encoded_resume,
            )

        return ModelRunHandle(future, _post)

    def structure(self) -> dict[str, dict[str, Any]]:
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
        >>> evaluator.structure()
        {"weekday": {"type": "categorical", "support": [...]}, ...}
        """
        model: MolvenoModel = self._model  # type: ignore[assignment]
        schema: dict[str, dict[str, Any]] = {}
        for cv in model.cvs:
            schema[cv.name] = {"type": "categorical", "support": list(cv.support)}
        for cap in model.capacities:
            schema[cap.name] = {"type": "distribution"}
        return schema

    def _extract_resume_state(self, output: MolvenoOutput) -> ResumeState:
        """Extract the resume payload from a previously saved :class:`MolvenoOutput`.

        Deserialises the raw :class:`~dt_model.simulation.evaluation.EvaluationResult`
        and the original parameter arrays from the encoded resume payload stored
        in ``output._serialized_resume``.

        Parameters
        ----------
        output : MolvenoOutput
            A :class:`MolvenoOutput` for which ``is_resumable`` is ``True``.

        Returns
        -------
        ResumeState
            All state needed to reconstruct an
            :class:`~dt_model.simulation.handle.EvaluationHandle`.
        """
        model: MolvenoModel = self._model  # type: ignore[assignment]
        assert output._serialized_resume is not None, "_extract_resume_state called on non-resumable output"
        resume = output._serialized_resume
        result = _decode_result(resume, model.indexes)
        idx_by_name: dict[str, GenericIndex] = {idx.name: idx for idx in model.indexes}
        parameter_arrays: dict[GenericIndex, np.ndarray] = {
            idx_by_name[name]: _decode_array(encoded)
            for name, encoded in resume["parameter_arrays"].items()
            if name in idx_by_name
        }
        return ResumeState(
            result=result,
            parameters=parameter_arrays,
            functions=None,
            backend=NumpyBackend,
        )
