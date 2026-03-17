"""Molveno overtourism model definition — modular decomposition.

The model is split into five concern sub-models:

:class:`PresenceModel` — *Stage 1, context and presence*
    **Inputs**: —
    **Outputs**: ``cv_weekday``, ``cv_season``, ``cv_weather``,
    ``pv_tourists``, ``pv_excursionists``

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

:class:`MolvenoModel` — *Root, owns all* ``i_*`` *defaults*
    Creates all ``i_*`` indexes with their default values and passes them
    to the concern sub-models.  Forwards ``PresenceModel`` outputs as
    needed.  Retains the domain attributes (``cvs``, ``pvs``,
    ``constraints``) required by
    :class:`~overtourism_molveno.overtourism_metamodel.OvertourismEnsemble`.


Design rules:

* **All** ``i_*`` parameters are ``Inputs`` to the sub-model that uses
  them, including uncertain ``DistributionIndex`` values.  The default
  values are created by :class:`MolvenoModel` and passed down via
  constructors.  A caller who wants to override a parameter simply
  supplies a different index object at construction time.
* All context variables (``cv_*``) and presence variables (``pv_*``) are
  ``Outputs`` of :class:`PresenceModel`.  ``cv_weather`` is also wired as
  an ``Input`` to the three concern sub-models that contain
  weather-dependent piecewise formulas.
* Each concern sub-model's ``Outputs`` contains only the usage-formula
  index (``i_u_*``).  Capacity indexes (``i_c_*``) remain as ``Inputs``
  because they are parameters, not computed results.
* Each concern sub-model stores its
  :class:`~overtourism_molveno.overtourism_metamodel.Constraint` as a
  plain instance attribute (``self.constraint``) because
  :class:`~overtourism_molveno.overtourism_metamodel.Constraint` is not a
  :class:`~dt_model.model.index.GenericIndex` and must not appear inside
  an :class:`~dt_model.model.model.IOProxy`.
* :class:`MolvenoModel` subclasses
  :class:`~overtourism_molveno.overtourism_metamodel.OvertourismModel` so
  that the existing
  :class:`~overtourism_molveno.overtourism_metamodel.OvertourismEnsemble`
  and :func:`~overtourism_molveno.overtourism_molveno.evaluate_scenario`
  code works without modification.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from scipy import stats

from civic_digital_twins.dt_model import piecewise
from civic_digital_twins.dt_model.model.index import DistributionIndex, GenericIndex, Index
from civic_digital_twins.dt_model.model.model import Model

try:
    from .molveno_presence_stats import (
        excursionist_presences_stats,
        season,
        tourist_presences_stats,
        weather,
        weekday,
    )
    from .overtourism_metamodel import (
        CategoricalContextVariable,
        Constraint,
        OvertourismModel,
        PresenceVariable,
        UniformCategoricalContextVariable,
    )
except ImportError:
    from molveno_presence_stats import (
        excursionist_presences_stats,
        season,
        tourist_presences_stats,
        weather,
        weekday,
    )
    from overtourism_metamodel import (
        CategoricalContextVariable,
        Constraint,
        OvertourismModel,
        PresenceVariable,
        UniformCategoricalContextVariable,
    )


# ---------------------------------------------------------------------------
# PresenceModel
# ---------------------------------------------------------------------------


class PresenceModel(Model):
    """Stage 1 — context variables and presence variables.

    Has no inputs of its own: the context variables and presence variables
    are self-contained.  All five are declared as ``Outputs`` so that
    :class:`MolvenoModel` can pick them up and forward them as needed.

    ``cv_weather`` is forwarded as an ``Input`` to :class:`ParkingModel`,
    :class:`BeachModel`, and :class:`FoodModel`.  ``cv_weekday`` and
    ``cv_season`` are consumed only by
    :class:`~overtourism_molveno.overtourism_metamodel.OvertourismEnsemble`
    (via :meth:`~dt_model.model.model.Model.abstract_indexes` on the root
    model) but are declared in ``Outputs`` for consistency and so that the
    root model can hand them to the ensemble explicitly.

    Outputs
    -------
    cv_weekday : UniformCategoricalContextVariable
        Day-of-week context variable.
    cv_season : CategoricalContextVariable
        Season context variable.
    cv_weather : CategoricalContextVariable
        Weather context variable.
    pv_tourists : PresenceVariable
        Tourist presence (grid axis in evaluation).
    pv_excursionists : PresenceVariable
        Excursionist presence (grid axis in evaluation).
    """

    @dataclass
    class Outputs:
        """Contractual outputs of :class:`PresenceModel`."""

        cv_weekday: UniformCategoricalContextVariable
        cv_season: CategoricalContextVariable
        cv_weather: CategoricalContextVariable
        pv_tourists: PresenceVariable
        pv_excursionists: PresenceVariable

    def __init__(self) -> None:
        Outputs = PresenceModel.Outputs

        cv_weekday = UniformCategoricalContextVariable("weekday", list(weekday))
        cv_season = CategoricalContextVariable("season", {v: season[v] for v in season.keys()})
        cv_weather = CategoricalContextVariable("weather", {v: weather[v] for v in weather.keys()})

        pv_tourists = PresenceVariable(
            "tourists",
            [cv_weekday, cv_season, cv_weather],
            tourist_presences_stats,
        )
        pv_excursionists = PresenceVariable(
            "excursionists",
            [cv_weekday, cv_season, cv_weather],
            excursionist_presences_stats,
        )

        super().__init__(
            "Presence",
            outputs=Outputs(
                cv_weekday=cv_weekday,
                cv_season=cv_season,
                cv_weather=cv_weather,
                pv_tourists=pv_tourists,
                pv_excursionists=pv_excursionists,
            ),
        )


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
    The :class:`~overtourism_molveno.overtourism_metamodel.Constraint` is
    stored as a plain instance attribute ``self.constraint``.

    Parameters
    ----------
    pv_tourists : PresenceVariable
        Tourist presence (wired from :class:`PresenceModel`).
    pv_excursionists : PresenceVariable
        Excursionist presence (wired from :class:`PresenceModel`).
    cv_weather : CategoricalContextVariable
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

        pv_tourists: PresenceVariable
        pv_excursionists: PresenceVariable
        cv_weather: CategoricalContextVariable
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
        pv_tourists: PresenceVariable,
        pv_excursionists: PresenceVariable,
        cv_weather: CategoricalContextVariable,
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
            inputs.pv_tourists.node
            * inputs.i_u_tourists_parking.node
            / (inputs.i_xa_tourists_per_vehicle.node * inputs.i_xo_tourists_parking.node)
            + inputs.pv_excursionists.node
            * inputs.i_u_excursionists_parking.node
            / (inputs.i_xa_excursionists_per_vehicle.node * inputs.i_xo_excursionists_parking.node),
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
    :class:`~overtourism_molveno.overtourism_metamodel.OvertourismEnsemble`.

    Parameters
    ----------
    pv_tourists : PresenceVariable
        Tourist presence (wired from :class:`PresenceModel`).
    pv_excursionists : PresenceVariable
        Excursionist presence (wired from :class:`PresenceModel`).
    cv_weather : CategoricalContextVariable
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

        pv_tourists: PresenceVariable
        pv_excursionists: PresenceVariable
        cv_weather: CategoricalContextVariable
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
        pv_tourists: PresenceVariable,
        pv_excursionists: PresenceVariable,
        cv_weather: CategoricalContextVariable,
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
            inputs.pv_tourists.node * inputs.i_u_tourists_beach.node / inputs.i_xo_tourists_beach.node
            + inputs.pv_excursionists.node * inputs.i_u_excursionists_beach.node / inputs.i_xo_excursionists_beach.node,
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
    pv_tourists : PresenceVariable
        Tourist presence (wired from :class:`PresenceModel`).
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

        pv_tourists: PresenceVariable
        i_u_tourists_accommodation: Index
        i_xa_tourists_accommodation: Index
        i_c_accommodation: DistributionIndex

    @dataclass
    class Outputs:
        """Contractual outputs of :class:`AccommodationModel`."""

        i_u_accommodation: Index

    def __init__(
        self,
        pv_tourists: PresenceVariable,
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
            inputs.pv_tourists.node * inputs.i_u_tourists_accommodation.node / inputs.i_xa_tourists_accommodation.node,
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
    pv_tourists : PresenceVariable
        Tourist presence (wired from :class:`PresenceModel`).
    pv_excursionists : PresenceVariable
        Excursionist presence (wired from :class:`PresenceModel`).
    cv_weather : CategoricalContextVariable
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

        pv_tourists: PresenceVariable
        pv_excursionists: PresenceVariable
        cv_weather: CategoricalContextVariable
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
        pv_tourists: PresenceVariable,
        pv_excursionists: PresenceVariable,
        cv_weather: CategoricalContextVariable,
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
            (
                inputs.pv_tourists.node * inputs.i_u_tourists_food.node
                + inputs.pv_excursionists.node * inputs.i_u_excursionists_food.node
            )
            / (inputs.i_xa_visitors_food.node * inputs.i_xo_visitors_food.node),
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


class MolvenoModel(OvertourismModel):
    """Root overtourism model that wires the five concern sub-models.

    ``MolvenoModel`` owns the default values for every ``i_*`` parameter.
    Callers who need to override a parameter can subclass ``MolvenoModel``
    or construct the concern sub-models directly with different values.

    ``MolvenoModel`` is a subclass of
    :class:`~overtourism_molveno.overtourism_metamodel.OvertourismModel` so
    that it is fully compatible with the existing
    :class:`~overtourism_molveno.overtourism_metamodel.OvertourismEnsemble`
    and :func:`~overtourism_molveno.overtourism_molveno.evaluate_scenario`
    code.

    Sub-models are accessible as named attributes:

    * ``model.presence``      — :class:`PresenceModel`
    * ``model.parking``       — :class:`ParkingModel`
    * ``model.beach``         — :class:`BeachModel`
    * ``model.accommodation`` — :class:`AccommodationModel`
    * ``model.food``          — :class:`FoodModel`

    Example usage::

        m = MolvenoModel()
        m.parking.inputs.i_c_parking      # capacity DistributionIndex
        m.beach.inputs.i_xo_tourists_beach  # rotation DistributionIndex
        m.parking.outputs.i_u_parking     # usage formula Index
        m.presence.outputs.pv_tourists    # PresenceVariable
        m.parking.constraint              # Constraint object
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------
        # Stage 1 — presence
        # ------------------------------------------------------------------
        presence = PresenceModel()

        pv_tourists = presence.outputs.pv_tourists
        pv_excursionists = presence.outputs.pv_excursionists
        cv_weather = presence.outputs.cv_weather

        # ------------------------------------------------------------------
        # Default i_* parameters — created here so callers can override them
        # ------------------------------------------------------------------

        # Parking parameters
        i_u_tourists_parking = Index("tourist parking usage factor", 0.02)
        i_u_excursionists_parking = Index(
            "excursionist parking usage factor",
            piecewise((0.55, cv_weather == "bad"), (0.80, True)),
        )
        i_xa_tourists_per_vehicle = Index("tourists per vehicle allocation factor", 2.5)
        i_xa_excursionists_per_vehicle = Index("excursionists per vehicle allocation factor", 2.5)
        i_xo_tourists_parking = Index("tourists in parking rotation factor", 1.02)
        i_xo_excursionists_parking = Index("excursionists in parking rotation factor", 3.5)
        i_c_parking = DistributionIndex("parking capacity", stats.uniform, {"loc": 350.0, "scale": 100.0})

        # Beach parameters
        i_u_tourists_beach = Index(
            "tourist beach usage factor",
            piecewise((0.25, cv_weather == "bad"), (0.50, True)),
        )
        i_u_excursionists_beach = Index(
            "excursionist beach usage factor",
            piecewise((0.35, cv_weather == "bad"), (0.80, True)),
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
            piecewise((0.80, cv_weather == "bad"), (0.40, True)),
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
        # Collect domain lists expected by OvertourismModel / OvertourismEnsemble
        # ------------------------------------------------------------------
        cvs = [presence.outputs.cv_weekday, presence.outputs.cv_season, presence.outputs.cv_weather]
        pvs = [pv_tourists, pv_excursionists]
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
            list(presence.indexes)
            + list(parking.indexes)
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

        # domain_indexes: everything that is not a CV, PV, capacity, or
        # usage-formula index (OvertourismModel keeps those lists separate).
        cv_pv_ids = {id(x) for x in cvs + pvs}
        cap_ids = {id(x) for x in capacities}
        usage_ids = {id(c.usage) for c in constraints}
        domain_indexes: list[GenericIndex] = [
            idx
            for idx in all_indexes
            if id(idx) not in cv_pv_ids and id(idx) not in cap_ids and id(idx) not in usage_ids
        ]

        # ------------------------------------------------------------------
        # Initialise OvertourismModel (the DeprecationWarning for the legacy
        # flat-list path is suppressed inside OvertourismModel.__init__)
        # ------------------------------------------------------------------
        super().__init__(
            "base model",
            cvs=cvs,
            pvs=pvs,
            indexes=domain_indexes,
            capacities=capacities,
            constraints=constraints,
        )

        # ------------------------------------------------------------------
        # Attach sub-models as named attributes for structured access
        # ------------------------------------------------------------------
        self.presence = presence
        self.parking = parking
        self.beach = beach
        self.accommodation = accommodation
        self.food = food

        # Presence-transformation parameters — kept as top-level attributes
        # so that overtourism_molveno.py can reference them by the original
        # names without modification.
        self.I_P_tourists_reduction_factor = i_p_tourists_reduction_factor
        self.I_P_excursionists_reduction_factor = i_p_excursionists_reduction_factor
        self.I_P_tourists_saturation_level = i_p_tourists_saturation_level
        self.I_P_excursionists_saturation_level = i_p_excursionists_saturation_level


# ---------------------------------------------------------------------------
# Module-level aliases — preserved for backward compatibility with
# overtourism_molveno.py and the existing test suite
# ---------------------------------------------------------------------------

#: Shared model instance (mirrors the previous ``M_Base`` module-level object).
M_Base = MolvenoModel()

# Context variables — exposed at module level for the script / test imports
CV_weekday = M_Base.presence.outputs.cv_weekday
CV_season = M_Base.presence.outputs.cv_season
CV_weather = M_Base.presence.outputs.cv_weather

# Presence variables
PV_tourists = M_Base.presence.outputs.pv_tourists
PV_excursionists = M_Base.presence.outputs.pv_excursionists

# Presence-transformation parameters
I_P_tourists_reduction_factor = M_Base.I_P_tourists_reduction_factor
I_P_excursionists_reduction_factor = M_Base.I_P_excursionists_reduction_factor
I_P_tourists_saturation_level = M_Base.I_P_tourists_saturation_level
I_P_excursionists_saturation_level = M_Base.I_P_excursionists_saturation_level
