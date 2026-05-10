"""Evaluation plan for the engine control layer.

An :class:`EvaluationPlan` encodes the *structure* of a model evaluation as
a DAG of :class:`Region` instances — groups of computation-graph nodes
evaluated as a unit.  Plans are built once via
:meth:`~simulation.evaluation.Evaluation.build_plan` and reused across
multiple :meth:`~simulation.evaluation.Evaluation.execute_plan` calls with
different ensembles and parameter grids.

The build *strategy* controls how the computation graph is partitioned into
regions:

- ``"monolithic"`` — one region containing all linearised nodes.
- ``"regional"`` — splits at :class:`~engine.frontend.graph.variant_selector`
  boundaries; shared pre-selector nodes form one unconditional region, each
  variant branch forms a guarded region, and the merge nodes form a final
  unconditional region.
- In the limit, each :class:`~engine.frontend.graph.Node` could be its own
  region (the plan DAG mirrors the computation graph exactly).
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

from ..engine.frontend import graph
from ..model.index import GenericIndex
from ..model.model import Model
from ..model.model_variant import ModelVariant

__all__ = [
    "EvaluationPlan",
    "Region",
    "RegionGuard",
]


@dataclasses.dataclass(frozen=True)
class RegionGuard:
    """Variant-branch execution guard for a :class:`Region`.

    A region carrying this guard is evaluated only for the scenario subset
    where ``selector_node`` evaluates to ``branch_key``.

    This is the variant-specific case of conditional region execution.
    Future generalizations may introduce other guard types.

    Parameters
    ----------
    selector_node:
        Graph node that produces a branch-key string per scenario
        (the :attr:`~engine.frontend.graph.variant_selector.selector_node`).
    branch_key:
        The branch key this region is responsible for.
    """

    selector_node: graph.Node
    branch_key: str


@dataclasses.dataclass(frozen=True)
class Region:
    """A partition of the computation graph evaluated as a unit.

    A region is one node of the evaluation DAG.  It contains a topologically
    sorted list of computation-graph nodes that are evaluated together in a
    single executor pass.

    Parameters
    ----------
    nodes:
        Topologically sorted computation-graph nodes in this region.
    has_timeseries:
        ``True`` when any node in :attr:`nodes` is a
        :class:`~engine.frontend.graph.timeseries_constant` or
        :class:`~engine.frontend.graph.timeseries_placeholder`; controls
        trailing-singleton injection during shape normalisation.
    guard:
        Execution guard, or ``None`` for an unconditional region (always
        evaluated for all scenarios).  When a :class:`RegionGuard` is
        present the region is evaluated only for the scenario subset where
        :attr:`RegionGuard.selector_node` equals
        :attr:`RegionGuard.branch_key`; results are then scattered back into
        the full-scenario array before the merge region executes.
    """

    nodes: tuple[graph.Node, ...]
    has_timeseries: bool
    guard: RegionGuard | None = dataclasses.field(default=None)


@dataclasses.dataclass(frozen=True)
class EvaluationPlan:
    """A DAG of :class:`Region` instances encoding the evaluation structure.

    The plan partitions the model's computation graph into regions
    (sub-graphs) and orders them as a directed acyclic graph.  Different
    build strategies produce different partitionings — from a single
    all-inclusive region (monolithic) to one region per computation-graph
    node (maximally split).

    :attr:`regions` are stored in **topological order**:
    ``dependencies[i] ⊂ {0, …, i − 1}`` — every predecessor of region *i*
    has a smaller index and is therefore evaluated first.

    Parameters
    ----------
    model:
        The model this plan was built from.
    nodes_of_interest:
        Indexes selected for evaluation; their transitive dependencies are
        included in the plan's regions.
    regions:
        Computation regions in topological order.
    dependencies:
        ``dependencies[i]`` is the (possibly empty) set of region indices
        that must complete before region *i* can be evaluated.  Parallel to
        :attr:`regions`.
    """

    model: Model | ModelVariant
    nodes_of_interest: tuple[GenericIndex, ...]
    regions: tuple[Region, ...]
    dependencies: tuple[frozenset[int], ...]
