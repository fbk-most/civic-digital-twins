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
  boundaries recursively.  At each nesting level: shared pre-selector nodes
  form one region guarded by all ancestor guards, each variant branch recurses
  with one additional guard appended, and the merge nodes form a final region
  guarded by the ancestor guards.  Single-level and nested
  :class:`~model.model_variant.ModelVariant` graphs are both supported.
- In the limit, each :class:`~engine.frontend.graph.Node` could be its own
  region (the plan DAG mirrors the computation graph exactly).
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses

from ..axes import Axis
from ..engine.frontend import graph
from ..model.index import GenericIndex
from ..model.model import Model
from ..model.model_variant import ModelVariant
from .scenario import Scenario

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
    domain_axes:
        The DOMAIN axes carried by any node in :attr:`nodes`, in canonical
        (name-sorted) order.  Controls how many trailing dimensions shape
        normalisation reserves, and which position each one occupies.  Empty
        for a region whose nodes are all scalar.
    guards:
        Ordered tuple of execution guards (outermost first), or ``()`` for an
        unconditional region.  The executor evaluates the region only for
        coordinates where *every* guard's selector equals its branch key (AND
        of all masks).
    """

    nodes: tuple[graph.Node, ...]
    domain_axes: tuple[Axis, ...] = ()
    guards: tuple[RegionGuard, ...] = ()


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

    def scoped_abstract_indexes(
        self,
        scenario: Scenario,
    ) -> dict[tuple[RegionGuard, ...], frozenset[GenericIndex]]:
        """Group scenario-abstract indexes by their region-scope guard chain.

        Each scenario-abstract index whose :attr:`~model.index.GenericIndex.node`
        appears in a :class:`Region`'s ``nodes`` tuple is assigned to that
        region's :attr:`Region.guards` tuple as the bucket key.  Multiple
        regions that share the same guard chain (e.g. a region's "shared"
        and "merge" pair) are merged into a single entry.  Buckets with
        no abstract indexes are dropped.

        Parameters
        ----------
        scenario:
            The scenario whose abstract indexes are grouped, seen from
            the scenario's perspective (overrides and ``parameter_axes``
            applied).

        Returns
        -------
        dict[tuple[RegionGuard, ...], frozenset[GenericIndex]]
            Mapping from guard chain (empty tuple for unconditional
            regions) to the abstract indexes that belong to that scope.

        Raises
        ------
        ValueError
            If an abstract index's node appears in more than one region.
            :meth:`~simulation.evaluation.Evaluation.build_plan` is expected
            to partition region nodes disjointly; downstream per-scope
            sampling in :class:`~simulation.ensemble.DistributionEnsemble`
            requires this invariant.
        """
        node_to_idx: dict[graph.Node, GenericIndex] = {idx.node: idx for idx in scenario.abstract_indexes()}
        buckets: dict[tuple[RegionGuard, ...], set[GenericIndex]] = {}
        seen: set[GenericIndex] = set()
        for region in self.regions:
            bucket = {node_to_idx[node] for node in region.nodes if node in node_to_idx}
            if not bucket:
                continue
            duplicates = bucket & seen
            if duplicates:
                names = ", ".join(sorted(getattr(idx, "name", repr(idx)) for idx in duplicates))
                raise ValueError(
                    f"Abstract indexes [{names}] appear in multiple regions of the plan; "
                    f"build_plan should partition region nodes disjointly."
                )
            seen |= bucket
            existing = buckets.get(region.guards)
            if existing is not None:
                existing |= bucket
            else:
                buckets[region.guards] = bucket
        return {key: frozenset(value) for key, value in buckets.items()}
