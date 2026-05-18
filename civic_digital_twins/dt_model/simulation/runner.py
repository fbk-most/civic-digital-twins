# SPDX-License-Identifier: Apache-2.0
"""Scenario runner protocol: ModelEvaluator, ModelOutput, and EvaluationConfig.

This module defines the stable ABCs and data types that sit between the
:mod:`~dt_model` engine layer and higher application levels (web APIs, CLIs,
UIs).  Domain packages subclass :class:`ModelEvaluator` and
:class:`ModelOutput` to expose a uniform evaluation lifecycle to any
application.

The lifecycle covered here is::

    Scenario → ModelEvaluator.evaluate(scenario, config) → ModelOutput
                                │
                                ├─ ModelOutput.to_dict()   → save to storage
                                └─ ModelOutput.from_dict() ← load from storage
                                         │
                                         └─ ModelEvaluator.resume(scenario, output, config)
                                                  → EvaluationHandle  (extend ensemble)

See :class:`ModelEvaluator` for the full protocol surface.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib.metadata
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from concurrent.futures import Future
from typing import Any, Generic, Self, TypeVar

import numpy as np

from ..engine.numpybackend.executor import Functor, NumpyBackend, State
from ..model.axis import Axis
from ..model.index import GenericIndex
from ..model.model import Model
from ..model.model_variant import ModelVariant
from .evaluation import Evaluation, EvaluationResult
from .handle import EvaluationHandle
from .scenario import Scenario

__all__ = [
    "EvaluationConfig",
    "IncompatibleResultError",
    "ModelEvaluator",
    "ModelOutput",
    "ModelRunHandle",
    "ResumeState",
]


# ---------------------------------------------------------------------------
# Version helper
# ---------------------------------------------------------------------------


def _get_dt_model_version() -> str:
    """Return the installed ``civic-digital-twins`` package version string.

    Used by :meth:`ModelOutput.to_dict` implementations to tag serialised
    outputs with the version that produced them.

    Returns
    -------
    str
        The package version string (e.g. ``"0.10.0"``).

    Raises
    ------
    importlib.metadata.PackageNotFoundError
        If the ``civic-digital-twins`` package metadata is not available.
        This should not occur in a correctly installed environment; with
        ``uv sync`` the metadata is always present.  The error is not
        silenced so that broken environments fail loudly rather than
        producing outputs tagged with an uninformative fallback string.
    """
    return importlib.metadata.version("civic-digital-twins")


# ---------------------------------------------------------------------------
# EvaluationConfig
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvaluationConfig:
    """Configuration for a single :meth:`ModelEvaluator.evaluate` call.

    Serves as a stable, extensible container for evaluation parameters.
    Subclass to add domain-specific or convergence-related fields in later
    milestones.

    Parameters
    ----------
    ensemble_size : int
        Total number of Monte Carlo samples drawn in one blocking
        :meth:`ModelEvaluator.evaluate` call.  Equivalent to
        :meth:`~simulation.evaluation.Evaluation.evaluate_incremental`'s
        ``initial_ensemble_size`` parameter.  Also used as the increment
        size when :meth:`ModelEvaluator.resume` extends a saved evaluation.
    """

    ensemble_size: int


# ---------------------------------------------------------------------------
# ResumeState
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ResumeState:
    """All state needed to reconstruct an :class:`~simulation.handle.EvaluationHandle`.

    Returned by :meth:`ModelEvaluator._extract_resume_state` and consumed by
    the :meth:`ModelEvaluator.resume` template method to reconstruct an
    :class:`~simulation.handle.EvaluationHandle` from a previously saved
    :class:`ModelOutput`.

    The concrete evaluator's ``_extract_resume_state`` implementation:

    - Deserialises ``result``, ``parameters``, and ``parameter_axes`` from
      the resume payload stored by :meth:`ModelOutput.to_dict`.
    - Re-injects ``functions`` and ``backend`` from its own domain knowledge
      (the same values used in :meth:`ModelEvaluator.evaluate`).

    Parameters
    ----------
    result : EvaluationResult
        The previously computed evaluation result to resume from.
    parameters : dict[GenericIndex, np.ndarray]
        PARAMETER axis dict that was passed to the original evaluation.
    parameter_axes : dict[str, np.ndarray] or None, optional
        Named PARAMETER axes dict, or ``None`` when correlated axes were not
        used.
    functions : dict[str, Functor] or None, optional
        User-defined functions to inject into the executor (e.g. Bologna's
        ``ts_solve``).  ``None`` when no external functions are required.
    backend : type[NumpyBackend], optional
        The computation backend.  Defaults to
        :class:`~engine.numpybackend.executor.NumpyBackend`.
    """

    result: EvaluationResult
    parameters: dict[GenericIndex, np.ndarray]
    parameter_axes: dict[str, np.ndarray] | None = None
    functions: dict[str, Functor] | None = None
    backend: type[NumpyBackend] = dataclasses.field(default=NumpyBackend)


# ---------------------------------------------------------------------------
# IncompatibleResultError
# ---------------------------------------------------------------------------


class IncompatibleResultError(Exception):
    """Raised when a saved :class:`ModelOutput` cannot be used to resume evaluation.

    Thrown by :meth:`ModelEvaluator.resume` when
    :attr:`ModelOutput.is_resumable` is ``False``.

    The output is still valid for analysis and re-plotting via its summary
    layer — only extension of the ensemble is not possible.

    Examples
    --------
    >>> try:
    ...     evaluator.resume(scenario, output, config)
    ... except IncompatibleResultError as exc:
    ...     print(f"Cannot resume: {exc}. Re-plotting is still possible.")
    """


# ---------------------------------------------------------------------------
# ModelOutput
# ---------------------------------------------------------------------------


class ModelOutput(ABC):
    """Abstract base class for domain-specific evaluation outputs.

    A ``ModelOutput`` carries two layers of data:

    **Summary layer** (stable, always readable)
        Post-processed KPIs, derived arrays needed for visualisation, parameter
        grid values, and scenario metadata.  Always serialised; always readable
        after :meth:`from_dict`.  Used for re-plotting and analysis.

    **Resume payload** (versioned, best-effort)
        Full raw :class:`~simulation.evaluation.EvaluationResult` arrays,
        weights, and the parameter grid — everything needed to reconstruct an
        :class:`~simulation.handle.EvaluationHandle` and extend the ensemble in
        a later session.  Written by :meth:`to_dict`; loaded by
        :meth:`from_dict` only when the serialised ``dt_model_version`` is
        compatible with the running version.

    Subclasses must:

    1. Call ``super().__init__()`` in their ``__init__``.
    2. Implement :meth:`to_dict` to serialise both layers, including
       ``"dt_model_version": _get_dt_model_version()`` in the top-level dict.
    3. Implement :meth:`from_dict` to reconstruct both layers when compatible,
       calling ``self._is_resumable = True`` only when the resume payload
       loads successfully.

    The :attr:`is_resumable` property is **not abstract**; its value is
    determined entirely by whether :meth:`from_dict` succeeded in loading the
    resume payload.  Subclasses must not override it.

    See Also
    --------
    ModelEvaluator : The evaluator ABC that produces and consumes ``ModelOutput``.
    IncompatibleResultError : Raised by :meth:`ModelEvaluator.resume` when not resumable.
    """

    def __init__(self) -> None:
        self._is_resumable: bool = False

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialise both the summary layer and the resume payload to a dict.

        The returned dict must always include:

        - ``"dt_model_version"``: the result of :func:`_get_dt_model_version`.
        - All summary data (KPIs, derived arrays, scenario metadata).
        - The resume payload (raw node arrays, weights, parameter grid).

        Numpy arrays should be encoded as base64 + dtype + shape for
        JSON-serialisable output, or kept as :class:`numpy.ndarray` when the
        application layer uses a binary-friendly persistence format (e.g.
        pickle or ``numpy.savez``).

        Returns
        -------
        dict[str, Any]
            Serialised output.  The exact schema is defined by the concrete
            subclass.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct a :class:`ModelOutput` from a serialised dict.

        Always succeeds for the summary layer.  Attempts to load the resume
        payload; sets ``self._is_resumable = True`` only when the payload is
        present and the serialised ``dt_model_version`` is compatible with the
        running version.

        Concrete implementations should follow this pattern::

            obj = cls.__new__(cls)
            ModelOutput.__init__(obj)          # sets _is_resumable = False
            # ... populate summary fields ...
            if _resume_payload_is_compatible(data):
                # ... populate resume fields ...
                obj._is_resumable = True
            return obj

        Parameters
        ----------
        data : dict[str, Any]
            Dict previously produced by :meth:`to_dict`.

        Returns
        -------
        Self
            A reconstructed instance of the concrete subclass.
        """

    @property
    def is_resumable(self) -> bool:
        """``True`` iff the resume payload is present and version-compatible.

        Set to ``False`` on construction.  Flipped to ``True`` by
        :meth:`from_dict` only when the resume payload loaded successfully.

        This property is concrete and must not be overridden by subclasses;
        its value is controlled exclusively by the ``_is_resumable`` flag.

        Returns
        -------
        bool
            Whether :meth:`ModelEvaluator.resume` can use this output to
            reconstruct an :class:`~simulation.handle.EvaluationHandle`.
        """
        return self._is_resumable


# ---------------------------------------------------------------------------
# ModelRunHandle
# ---------------------------------------------------------------------------

OutputT = TypeVar("OutputT", bound=ModelOutput)


class ModelRunHandle(Generic[OutputT]):
    """Async handle wrapping a :class:`concurrent.futures.Future` and a post-processor.

    Returned by :meth:`ModelEvaluator.run_async`.  The future carries a raw
    :class:`~simulation.evaluation.EvaluationResult`; the post-processor
    converts it to the domain-specific :class:`ModelOutput` subtype ``OutputT``.

    The future is obtained from either
    :attr:`~simulation.handle.AsyncEvaluationHandle.future` (Bologna, tier 3
    via :meth:`~simulation.evaluation.Evaluation.submit_evaluate`) or
    :func:`~dt_model.simulation.evaluation._get_default_executor` with
    :meth:`~simulation.evaluation.Evaluation.evaluate` as the submitted
    callable (Molveno, thread-pool submit of the engine call).

    Parameters
    ----------
    future : Future[EvaluationResult]
        The in-flight or completed engine evaluation.
    post_process : Callable[[EvaluationResult], OutputT]
        Domain-specific function that converts the raw
        :class:`~simulation.evaluation.EvaluationResult` into a ``ModelOutput``
        subclass instance.

    Examples
    --------
    >>> handle = evaluator.run_async(scenario, config)
    >>> done, output = handle.poll()
    >>> if not done:
    ...     output = handle.get()   # blocks until complete
    """

    def __init__(
        self,
        future: Future[EvaluationResult],
        post_process: Callable[[EvaluationResult], OutputT],
    ) -> None:
        self._future = future
        self._post_process = post_process

    def get(self) -> OutputT:
        """Block until the evaluation completes and return the :class:`ModelOutput`.

        Applies the post-processor to the resolved
        :class:`~simulation.evaluation.EvaluationResult`.  Subsequent calls
        return a freshly post-processed result (the future is cached by the
        executor).

        Returns
        -------
        OutputT
            The domain-specific :class:`ModelOutput` for this evaluation.
        """
        return self._post_process(self._future.result())

    def poll(self) -> tuple[bool, OutputT | None]:
        """Non-blocking status check.

        .. note::

            ``(False, None)`` means the evaluation is not yet complete.
            Intermediate progress is not observable through this interface:
            a handle that has completed zero rounds and one that has completed
            some rounds both return ``(False, None)``.  See :issue:`188` for
            the planned ``extend()`` / partial-result API at this level.

        Returns
        -------
        tuple[bool, OutputT | None]
            ``(True, output)`` if the evaluation is complete;
            ``(False, None)`` if it is still running.
        """
        if not self._future.done():
            return False, None
        return True, self._post_process(self._future.result())

    def cancel(self) -> bool:
        """Attempt to cancel the underlying future.

        Has no effect if the future has already started or completed.

        Returns
        -------
        bool
            ``True`` if the future was successfully cancelled.
        """
        return self._future.cancel()


# ---------------------------------------------------------------------------
# EvaluationResult codec helpers
# ---------------------------------------------------------------------------


def _encode_array(arr: np.ndarray) -> dict[str, Any]:
    """Encode a numpy array to a JSON-serialisable dict.

    Uses base64 encoding of the raw bytes together with dtype and shape
    metadata so that the round-trip is lossless for all numeric dtypes.
    Object-dtype arrays (e.g. categorical string assignments) are encoded
    as a JSON list to avoid the ``frombuffer`` limitation on object buffers.

    Parameters
    ----------
    arr : np.ndarray
        Array to encode.

    Returns
    -------
    dict[str, Any]
        Dict with keys ``"data"`` (base64 string or list), ``"dtype"`` (str),
        ``"shape"`` (list of int), and optionally ``"encoding"`` (``"json"``
        for object-dtype arrays).
    """
    if arr.dtype == object:
        # Object arrays (e.g. categorical string assignments) cannot be
        # round-tripped via tobytes()/frombuffer.  Store as a JSON-safe list.
        return {
            "data": arr.tolist(),
            "dtype": "object",
            "shape": list(arr.shape),
            "encoding": "json",
        }
    return {
        "data": base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _decode_array(d: dict[str, Any]) -> np.ndarray:
    """Decode a numpy array from a dict produced by :func:`_encode_array`.

    Handles both base64-encoded numeric arrays and JSON-encoded object arrays
    (those produced with ``"encoding": "json"`` by :func:`_encode_array`).

    Parameters
    ----------
    d : dict[str, Any]
        Dict with keys ``"data"``, ``"dtype"``, ``"shape"``, and optionally
        ``"encoding"``.

    Returns
    -------
    np.ndarray
        The decoded array.  Returns a writable copy (``frombuffer`` would
        give a read-only view for the numeric path).
    """
    if d.get("encoding") == "json":
        return np.array(d["data"], dtype=object).reshape(tuple(d["shape"]))
    raw = base64.b64decode(d["data"].encode("ascii"))
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(tuple(d["shape"])).copy()


def _encode_result(result: EvaluationResult, indexes: Iterable[GenericIndex]) -> dict[str, Any]:
    """Encode an :class:`~simulation.evaluation.EvaluationResult` as a serialisable dict.

    Iterates over *indexes* and stores each node's array under the index
    name.  Also encodes the axis layout, factorized weights, parameter
    arrays, named axis values, and axis sizes — everything
    :func:`_decode_result` needs to reconstruct a fully functional
    :class:`~simulation.evaluation.EvaluationResult`.

    Parameters
    ----------
    result : EvaluationResult
        The result to encode.
    indexes : Iterable[GenericIndex]
        Model indexes used to map graph nodes to stable string names.
        Indexes whose nodes are absent from *result* are silently skipped.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable dict suitable for embedding in a
        :meth:`ModelOutput.to_dict` payload.
    """
    nodes: dict[str, Any] = {}
    for idx in indexes:
        try:
            nodes[idx.name] = _encode_array(result[idx])
        except KeyError:
            pass  # index not computed in this evaluation

    axis_layout = [
        [ax.name, ax.role, pos]
        for ax, pos in result._axis_layout.items()  # type: ignore[attr-defined]
    ]
    factorized_weights = {
        ax.name: _encode_array(w)
        for ax, w in result._factorized_weights.items()  # type: ignore[attr-defined]
    }
    parameter_arrays = {idx.name: _encode_array(arr) for idx, arr in result.parameter_values.items()}
    named_axis_values = {name: _encode_array(arr) for name, arr in result.named_axis_values.items()}
    axis_sizes = {
        f"{ax.name}:{ax.role}": size
        for ax, size in result._axis_sizes.items()  # type: ignore[attr-defined]
    }
    return {
        "nodes": nodes,
        "axis_layout": axis_layout,
        "factorized_weights": factorized_weights,
        "parameter_arrays": parameter_arrays,
        "named_axis_values": named_axis_values,
        "axis_sizes": axis_sizes,
    }


def _decode_result(data: dict[str, Any], indexes: Iterable[GenericIndex]) -> EvaluationResult:
    """Reconstruct an :class:`~simulation.evaluation.EvaluationResult` from an encoded dict.

    Matches stored arrays back to model indexes by name, then constructs a
    new :class:`~simulation.evaluation.EvaluationResult` whose node arrays,
    axis layout, and weights are compatible with those produced by a fresh
    :meth:`~simulation.evaluation.Evaluation.execute_plan` call on the same
    scenario — so that :func:`~simulation.handle._merge_results` can merge
    the loaded result with new samples.

    Parameters
    ----------
    data : dict[str, Any]
        Dict previously produced by :func:`_encode_result`.
    indexes : Iterable[GenericIndex]
        Model indexes used to map names back to graph nodes and
        :class:`~model.index.GenericIndex` keys.

    Returns
    -------
    EvaluationResult
        Reconstructed result.  Graph node identity is that of the current
        model, so the result is valid for the current session.
    """
    idx_by_name: dict[str, GenericIndex] = {idx.name: idx for idx in indexes}
    axis_role: dict[str, str] = {row[0]: row[1] for row in data["axis_layout"]}

    state_values: dict = {}
    for name, encoded in data["nodes"].items():
        if name in idx_by_name:
            state_values[idx_by_name[name].node] = _decode_array(encoded)
    state = State(values=state_values)

    axis_layout: dict[Axis, int] = {Axis(row[0], row[1]): int(row[2]) for row in data["axis_layout"]}
    factorized_weights: dict[Axis, np.ndarray] = {
        Axis(name, axis_role[name]): _decode_array(encoded) for name, encoded in data["factorized_weights"].items()
    }
    parameter_arrays: dict[GenericIndex, np.ndarray] = {
        idx_by_name[name]: _decode_array(encoded)
        for name, encoded in data["parameter_arrays"].items()
        if name in idx_by_name
    }
    named_axis_values: dict[str, np.ndarray] = {
        name: _decode_array(encoded) for name, encoded in data["named_axis_values"].items()
    }
    axis_sizes: dict[Axis, int] = {}
    for key, size in data["axis_sizes"].items():
        ax_name, ax_role = key.split(":", 1)
        axis_sizes[Axis(ax_name, ax_role)] = int(size)

    return EvaluationResult(
        state=state,
        axis_layout=axis_layout,
        parameter_arrays=parameter_arrays,
        axis_sizes=axis_sizes,
        factorized_weights=factorized_weights,
        named_axis_values=named_axis_values,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_value(val: Any) -> str:
    """Format a :data:`~model.index.DomainValue` or ``None`` for human display.

    Used by :meth:`ModelEvaluator.get_index_diffs` to produce readable
    ``"was X \u2192 now Y"`` diff strings.  Delegates to ``str()`` for most
    types; adds special handling for ``None`` and numpy arrays.

    Parameters
    ----------
    val : Any
        The value to format.  Typically a :data:`~model.index.DomainValue`
        or ``None``.

    Returns
    -------
    str
        A human-readable string representation.
    """
    if val is None:
        return "(none)"
    if isinstance(val, np.ndarray):
        return f"array{val.shape}"
    return str(val)


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------


class ModelEvaluator(ABC, Generic[OutputT]):
    """Abstract base class for domain-specific scenario evaluators.

    Each domain package subclasses :class:`ModelEvaluator` and binds the
    type parameter ``OutputT`` to its concrete :class:`ModelOutput` subclass.
    The application layer then drives the uniform lifecycle::

        evaluator = DomainEvaluator(model)
        output    = evaluator.evaluate(scenario, config)
        data      = output.to_dict()                        # save
        output2   = DomainOutput.from_dict(data)            # load
        handle    = evaluator.resume(scenario, output2, config)  # extend

    **Implementation tiers for** :meth:`run_async`:

    1. *Sync only* — implement :meth:`evaluate` only; leave :meth:`run_async`
       at its default (raises :exc:`NotImplementedError`).
    2. *Protocol-level async* — implement :meth:`run_async` to submit the
       blocking :meth:`evaluate` to :func:`~simulation.evaluation._get_default_executor`.
       Suitable for models that use
       :class:`~simulation.ensemble.CrossProductEnsemble` (e.g. Molveno).
    3. *Engine-level async* — implement :meth:`run_async` to call
       :meth:`~simulation.evaluation.Evaluation.submit_evaluate` and wrap the
       returned :attr:`~simulation.handle.AsyncEvaluationHandle.future` in a
       :class:`ModelRunHandle`.  Suitable for models that use
       :class:`~simulation.ensemble.DistributionEnsemble` (e.g. Bologna).

    Parameters
    ----------
    model : Model or ModelVariant
        The model this evaluator operates on.  Stored as ``self._model`` and
        used by the default implementations of :meth:`get_index_diffs`,
        :meth:`get_model_values`, and :meth:`structure`.
    """

    def __init__(self, model: Model | ModelVariant) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, scenario: Scenario, config: EvaluationConfig) -> OutputT:
        """Run a blocking evaluation and return the domain output.

        The subclass builds the appropriate ensemble, calls
        :class:`~simulation.evaluation.Evaluation`, runs domain-specific
        post-processing, and returns a fully populated :class:`ModelOutput`.

        Parameters
        ----------
        scenario : Scenario
            The scenario to evaluate, carrying optional value overrides.
        config : EvaluationConfig
            Evaluation parameters; ``config.ensemble_size`` controls how many
            Monte Carlo samples are drawn.

        Returns
        -------
        OutputT
            The domain-specific evaluation output.
        """

    @abstractmethod
    def structure(self) -> dict[str, dict[str, Any]]:
        """Return a schema dict describing the model's tunable indexes.

        Maps each index name to a metadata dict::

            {
                "parking_cost": {"type": "scalar", "default": 8.0, "unit": "\u20ac"},
                "weather":      {"type": "categorical", "support": ["good", "bad"]},
            }

        Used by scenario-creation UIs to know what parameters exist and
        what values are valid.  A typed schema protocol will replace this
        plain dict in a future milestone.

        Returns
        -------
        dict[str, dict[str, Any]]
            Index name \u2192 metadata dict.
        """

    @abstractmethod
    def _extract_resume_state(self, output: OutputT) -> ResumeState:
        """Extract the resume payload from a previously saved output.

        Called by :meth:`resume` (template method).  The subclass
        deserialises ``result``, ``parameters``, and ``parameter_axes`` from
        ``output``'s resume payload, and re-injects ``functions`` and
        ``backend`` from its own domain knowledge (same values used in
        :meth:`evaluate`).

        Parameters
        ----------
        output : OutputT
            A :class:`ModelOutput` for which ``is_resumable`` is ``True``.

        Returns
        -------
        ResumeState
            All state needed to reconstruct an
            :class:`~simulation.handle.EvaluationHandle`.
        """

    # ------------------------------------------------------------------
    # Optional async interface
    # ------------------------------------------------------------------

    def run_async(self, scenario: Scenario, config: EvaluationConfig) -> ModelRunHandle[OutputT]:
        """Submit an async evaluation and return a handle immediately.

        Not implemented by default.  Override in tier-2 or tier-3
        subclasses to enable non-blocking evaluation; see class docstring
        for the three implementation tiers.

        Parameters
        ----------
        scenario : Scenario
            The scenario to evaluate.
        config : EvaluationConfig
            Evaluation parameters.

        Returns
        -------
        ModelRunHandle[OutputT]
            Handle whose :meth:`~ModelRunHandle.get` returns the output.

        Raises
        ------
        NotImplementedError
            Always, in the default implementation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_async(). Override it or use evaluate() directly."
        )

    # ------------------------------------------------------------------
    # Default introspection methods
    # ------------------------------------------------------------------

    def get_index_diffs(self, scenario: Scenario) -> dict[str, str]:
        """Return human-readable diff strings for each overridden index.

        Compares ``scenario.overrides`` against the model's own values
        (the no-overrides baseline).  Returns one entry per overridden
        index formatted as ``"was X \u2192 now Y"``.

        Parameters
        ----------
        scenario : Scenario
            The scenario whose overrides are described.

        Returns
        -------
        dict[str, str]
            ``{index_name: "was X \u2192 now Y"}`` for each overridden index;
            empty dict when no overrides are active.
        """
        return {
            idx.name: f"was {_format_value(getattr(idx, 'value', None))} \u2192 now {_format_value(override_val)}"
            for idx, override_val in scenario.overrides.items()
        }

    def get_model_values(self, scenario: Scenario) -> dict[str, Any]:
        """Return the effective value of every model index under *scenario*.

        For indexes that have an active override the override value is
        returned; for all others the model's own ``idx.value`` is used
        (which may be ``None`` for abstract indexes with no override).

        Parameters
        ----------
        scenario : Scenario
            The scenario providing active overrides.

        Returns
        -------
        dict[str, Any]
            ``{index_name: effective_value}`` for every index in the model.
        """
        active = scenario.overrides
        return {
            idx.name: (active[idx] if idx in active else getattr(idx, "value", None)) for idx in self._model.indexes
        }

    # ------------------------------------------------------------------
    # Resume template method
    # ------------------------------------------------------------------

    def resume(
        self,
        scenario: Scenario,
        output: OutputT,
        config: EvaluationConfig,
    ) -> EvaluationHandle:
        """Reconstruct an :class:`~simulation.handle.EvaluationHandle` from a saved output.

        Template method.  Checks that *output* is resumable, delegates
        deserialisation to :meth:`_extract_resume_state`, builds a fresh
        :class:`~simulation.evaluation.Evaluation` plan, and returns an
        :class:`~simulation.handle.EvaluationHandle` ready for
        :meth:`~simulation.handle.EvaluationHandle.extend`.

        .. note::
            :class:`~simulation.handle.EvaluationHandle` is constructed
            directly here rather than via
            :meth:`~simulation.evaluation.Evaluation.evaluate_incremental`.
            This is an intentional exception to the convention documented on
            that class, necessary to seed the handle with a pre-existing
            result.

        .. note::
            The :class:`~simulation.handle.EvaluationHandle` is seeded with
            a fresh :func:`numpy.random.default_rng` with no fixed seed, so
            samples drawn by subsequent :meth:`~simulation.handle.EvaluationHandle.extend`
            calls are not reproducible across sessions.  If reproducibility
            matters, add an optional ``rng=`` parameter to this method in a
            future milestone.

        Parameters
        ----------
        scenario : Scenario
            The scenario used to rebuild the evaluation plan.
        output : OutputT
            A previously produced :class:`ModelOutput`.  Must have
            ``is_resumable == True``.
        config : EvaluationConfig
            Evaluation parameters (currently unused; reserved for future
            convergence-loop support).

        Returns
        -------
        EvaluationHandle
            Handle seeded with the saved result; call
            :meth:`~simulation.handle.EvaluationHandle.extend` to draw
            additional Monte Carlo samples.

        Raises
        ------
        IncompatibleResultError
            If ``output.is_resumable`` is ``False``.
        """
        if not output.is_resumable:
            raise IncompatibleResultError(
                f"{type(output).__name__} is not resumable. "
                "The resume payload may be absent or was produced by an "
                "incompatible version of civic-digital-twins. "
                "Re-plotting from the summary layer is still possible."
            )
        state = self._extract_resume_state(output)
        evaluation = Evaluation(scenario)
        plan = evaluation.build_plan()
        return EvaluationHandle(
            evaluation=evaluation,
            plan=plan,
            result=state.result,
            rng=np.random.default_rng(),
            parameters=state.parameters,
            parameter_axes=state.parameter_axes,
            functions=state.functions,
            backend=state.backend,
        )
