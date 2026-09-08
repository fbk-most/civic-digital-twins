# SPDX-License-Identifier: Apache-2.0
"""Scenario runner protocol: ModelEvaluator, ModelOutput, and EvaluationConfig.

This module defines the stable ABCs and data types that sit between the
:mod:`~dt_model` engine layer and higher application levels (web APIs, CLIs,
UIs).  Domain packages subclass :class:`ModelEvaluator` and
:class:`ModelOutput` to expose a uniform evaluation lifecycle to any
application.

**One-shot evaluation**::

    Scenario → ModelEvaluator.evaluate(scenario, config) → ModelOutput

**Incremental evaluation**::

    run = evaluator.start(scenario, config)    → IncrementalRun
    run = evaluator.resume(scenario, out, cfg) → IncrementalRun
               │
               ├─ .extend(n)
               ├─ .snapshot()               → ModelOutput  (no resume payload)
               └─ .snapshot(resumable=True) → ModelOutput  (with resume payload)
                                                  │
                              ┌───────────────────┘
                              ▼
                 ModelOutput.to_dict()   → save to storage
                 ModelOutput.from_dict() ← load from storage
                              │
                 evaluator.resume(scenario, output, config) → IncrementalRun

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

from ..axes import Axis
from ..engine.numpybackend.executor import Functor, NumpyBackend, State
from ..model.index import DistributionIndex, GenericIndex, Index
from ..model.model import Model
from ..model.model_variant import ModelVariant
from .axis_layout import AxisLayout
from .ensemble import DistributionEnsemble
from .evaluation import Evaluation, EvaluationResult
from .handle import AsyncEvaluationHandle, EvaluationHandle
from .scenario import Scenario

__all__ = [
    "EvaluationConfig",
    "IncompatibleResultError",
    "IncrementalRun",
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
        Number of Monte Carlo samples drawn per batch.  Used as the
        ``initial_ensemble_size`` in :meth:`ModelEvaluator.evaluate` and
        :meth:`ModelEvaluator.start`, and as the default increment when
        :meth:`IncrementalRun.extend` is called without an explicit *n*.
    """

    ensemble_size: int


# ---------------------------------------------------------------------------
# ResumeState
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ResumeState:
    """All state needed to reconstruct an :class:`~simulation.handle.EvaluationHandle`.

    Returned by :meth:`ModelEvaluator.extract_resume_state` and consumed by
    the :meth:`ModelEvaluator.resume` template method to reconstruct an
    :class:`~simulation.handle.EvaluationHandle` from a previously saved
    :class:`ModelOutput`.

    The concrete evaluator's ``extract_resume_state`` implementation:

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
# Codec helpers (used by ModelOutput._serialize/_deserialize and
# ModelEvaluator.attach_resume/extract_resume_state)
# ---------------------------------------------------------------------------


def _looks_like_encoded_array(val: Any) -> bool:
    """Return ``True`` when *val* is a dict produced by :func:`_encode_array`."""
    return isinstance(val, dict) and "data" in val and "dtype" in val and "shape" in val


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
    # The numeric path stores raw bytes with the array's native dtype string
    # (e.g. "float64"), which carries no explicit byte order.  Decoding therefore
    # assumes the same endianness as the encoding host — sound for the intended
    # save/resume-on-the-same-machine workflow, but not a portable wire format
    # across architectures of differing endianness.
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

    factorized_weights = {ax.name: _encode_array(w) for ax, w in result.factorized_weights.items()}
    parameter_arrays = {idx.name: _encode_array(arr) for idx, arr in result.parameter_values.items()}
    named_axis_values = {name: _encode_array(arr) for name, arr in result.named_axis_values.items()}
    layout_dict = result.layout.to_dict()
    return {
        "nodes": nodes,
        "axis_layout": layout_dict["axis_layout"],
        "factorized_weights": factorized_weights,
        "parameter_arrays": parameter_arrays,
        "named_axis_values": named_axis_values,
        "axis_sizes": layout_dict["axis_sizes"],
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

    state_values: dict = {}
    for name, encoded in data["nodes"].items():
        if name in idx_by_name:
            state_values[idx_by_name[name].node] = _decode_array(encoded)
    state = State(values=state_values)

    layout = AxisLayout.from_dict(data)
    # factorized_weights is serialised keyed by axis name alone (it only ever holds
    # ENSEMBLE axes); recover each axis (and its role) from the reconstructed
    # layout.  This is unambiguous because axis names are globally unique within
    # an EvaluationResult (see Axis docstring): no two axes — across PARAMETER,
    # ENSEMBLE, or DOMAIN — share a name, so the lookup cannot collide on role.
    factorized_weights: dict[Axis, np.ndarray] = {}
    for name, encoded in data["factorized_weights"].items():
        ax = layout.find_axis(name)
        assert ax is not None, f"_decode_result: factorized_weights axis {name!r} not found in axis_layout."
        factorized_weights[ax] = _decode_array(encoded)
    parameter_arrays: dict[GenericIndex, np.ndarray] = {
        idx_by_name[name]: _decode_array(encoded)
        for name, encoded in data["parameter_arrays"].items()
        if name in idx_by_name
    }
    named_axis_values: dict[str, np.ndarray] = {
        name: _decode_array(encoded) for name, encoded in data["named_axis_values"].items()
    }

    return EvaluationResult(
        state=state,
        axis_layout=layout,
        parameter_arrays=parameter_arrays,
        factorized_weights=factorized_weights,
        named_axis_values=named_axis_values,
    )


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

    1. Call ``super().__init__()`` in their ``__init__``.  To mark the
       output as resumable, call :meth:`_store_resume` with the encoded
       payload produced by :func:`_encode_result`.
    2. Implement :meth:`_serialize` / :meth:`_deserialize` **only if** the
       subclass is not a dataclass or has fields that need special handling;
       the base provides a dataclass-aware default.  The base :meth:`to_dict`
       stamps ``"dt_model_version"`` and appends the resume payload
       automatically; :meth:`from_dict` handles object construction and resume
       payload loading.

    The :attr:`is_resumable` property is **not abstract**; its value is
    determined entirely by whether :meth:`_store_resume` was called.
    Subclasses must not override it.

    See Also
    --------
    ModelEvaluator : The evaluator ABC that produces and consumes ``ModelOutput``.
    IncompatibleResultError : Raised by :meth:`ModelEvaluator.resume` when not resumable.
    """

    def __init__(self) -> None:
        self._is_resumable: bool = False
        self._serialized_resume: dict[str, Any] | None = None

    def _store_resume(self, payload: dict[str, Any]) -> None:
        """Store a resume payload and mark this output as resumable.

        Called by subclass ``__init__`` implementations to record the
        serialised :class:`~simulation.evaluation.EvaluationResult`
        payload produced by :func:`_encode_result`.  Sets
        :attr:`is_resumable` to ``True``.

        Parameters
        ----------
        payload : dict[str, Any]
            The encoded result dict returned by :func:`_encode_result`.
        """
        self._serialized_resume = payload
        self._is_resumable = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise both the summary layer and the resume payload to a dict.

        Concrete template method.  Stamps ``"dt_model_version"`` automatically,
        merges in the domain-specific summary returned by
        :meth:`_serialize`, and appends ``"_resume"`` when a payload
        has been stored via :meth:`_store_resume`.

        Returns
        -------
        dict[str, Any]
            Serialised output including version stamp and, if resumable, the
            resume payload under the ``"_resume"`` key.
        """
        data: dict[str, Any] = {"dt_model_version": _get_dt_model_version()}
        data.update(self._serialize())
        if self._serialized_resume is not None:
            data["_resume"] = self._serialized_resume
        return data

    def to_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot suitable for API responses.

        Concrete default.  Stamps ``"dt_model_version"`` and merges in
        :meth:`_serialize` — identical to :meth:`to_dict` but **without**
        the ``"_resume"`` payload.

        Override in subclasses to append derived / computed fields that the
        frontend needs but that are not stored in the checkpoint
        (e.g. sustainability metrics computed from the raw arrays).  Call
        ``super().to_snapshot()`` and add entries to the returned dict.

        The ``"_resume"`` key is structurally absent from the return value;
        it is never included even if :attr:`is_resumable` is ``True``.

        Returns
        -------
        dict[str, Any]
            Snapshot dict.  Contains ``"dt_model_version"`` and all fields
            from :meth:`_serialize`, plus any derived fields added by the
            subclass override.
        """
        data: dict[str, Any] = {"dt_model_version": _get_dt_model_version()}
        data.update(self._serialize())
        return data

    def _serialize(self) -> dict[str, Any]:
        """Return the summary layer as a JSON-compatible dict.

        Default implementation for :func:`~dataclasses.dataclass` subclasses.
        Inspects every dataclass field:

        - :class:`numpy.ndarray` values are encoded with :func:`_encode_array`.
        - ``dict`` values whose values are all :class:`numpy.ndarray` are encoded
          per-value with :func:`_encode_array`.
        - All other values are stored as-is (assumed JSON-serialisable).

        Override this method for outputs that are not dataclasses or that have
        fields requiring special handling.

        Returns
        -------
        dict[str, Any]
            Summary layer dict.  Must not contain the keys
            ``"dt_model_version"`` or ``"_resume"``.

        Raises
        ------
        NotImplementedError
            If the subclass is not a dataclass and has not overridden this method.
        """
        if not dataclasses.is_dataclass(self):
            raise NotImplementedError(
                f"{type(self).__name__} is not a dataclass. Override _serialize() to provide custom serialization."
            )
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            if isinstance(val, np.ndarray):
                out[f.name] = _encode_array(val)
            elif isinstance(val, dict) and all(isinstance(v, np.ndarray) for v in val.values()):
                out[f.name] = {k: _encode_array(v) for k, v in val.items()}
            else:
                out[f.name] = val
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct a :class:`ModelOutput` from a serialised dict.

        Concrete template method.  Creates an empty instance, delegates
        summary-field population to :meth:`_deserialize`, then
        restores the resume payload from the ``"_resume"`` key if present.

        Parameters
        ----------
        data : dict[str, Any]
            Dict previously produced by :meth:`to_dict`.

        Returns
        -------
        Self
            Reconstructed instance.  :attr:`is_resumable` is ``True`` iff
            ``"_resume"`` was present in *data*.
        """
        obj = cls.__new__(cls)
        ModelOutput.__init__(obj)
        obj._deserialize(data)
        if "_resume" in data:
            obj._store_resume(data["_resume"])
        return obj

    def _deserialize(self, data: dict[str, Any]) -> None:
        """Populate summary fields from a serialised dict.

        Default implementation for :func:`~dataclasses.dataclass` subclasses.
        For each dataclass field present in *data*:

        - If the stored value looks like an :func:`_encode_array` envelope
          (dict with ``"data"``, ``"dtype"``, ``"shape"`` keys) it is decoded
          with :func:`_decode_array`.
        - If the stored value is a ``dict`` whose values all look like
          array envelopes, each is decoded with :func:`_decode_array`.
        - Otherwise the value is set as-is.

        Override this method for outputs that are not dataclasses or that
        require special handling (e.g. backward-compatibility shims, optional
        keys with defaults).

        Parameters
        ----------
        data : dict[str, Any]
            Dict previously produced by :meth:`to_dict`.

        Raises
        ------
        NotImplementedError
            If the subclass is not a dataclass and has not overridden this method.
        """
        if not dataclasses.is_dataclass(self):
            raise NotImplementedError(
                f"{type(self).__name__} is not a dataclass. Override _deserialize() to provide custom deserialization."
            )
        for f in dataclasses.fields(self):  # type: ignore[arg-type]
            if f.name not in data:
                if f.default is not dataclasses.MISSING:
                    setattr(self, f.name, f.default)
                elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                    setattr(self, f.name, f.default_factory())  # type: ignore[misc]
                continue
            raw = data[f.name]
            if _looks_like_encoded_array(raw):
                setattr(self, f.name, _decode_array(raw))
            elif isinstance(raw, dict) and all(_looks_like_encoded_array(v) for v in raw.values()):
                setattr(self, f.name, {k: _decode_array(v) for k, v in raw.items()})
            else:
                setattr(self, f.name, raw)

    @property
    def is_resumable(self) -> bool:
        """``True`` iff the resume payload is available on this output.

        Set to ``True`` whenever :meth:`_store_resume` is called — either
        directly in a subclass ``__init__`` after evaluation, or by
        :meth:`from_dict` when the serialised dict contains a ``"_resume"``
        key.  Remains ``False`` when the checkpoint was saved without a
        resume payload or was produced by an incompatible version.

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
ModelT = TypeVar("ModelT", bound="Model | ModelVariant")


class ModelRunHandle(Generic[OutputT]):
    """Async handle wrapping a :class:`concurrent.futures.Future` and a post-processor.

    Returned by :meth:`ModelEvaluator.run_async`.  The future carries a raw
    :class:`~simulation.evaluation.EvaluationResult`; the post-processor
    converts it to the domain-specific :class:`ModelOutput` subtype ``OutputT``.

    The future is obtained from either
    :attr:`~simulation.handle.AsyncEvaluationHandle.future` (Bologna, tier 3
    via :meth:`AsyncEvaluationHandle.evaluate`) or
    :func:`~dt_model.simulation.handle._get_default_executor` with
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
# IncrementalRun
# ---------------------------------------------------------------------------


class IncrementalRun(Generic[OutputT]):
    """An in-progress incremental evaluation that can be extended and snapshotted.

    Returned by :meth:`ModelEvaluator.start` and :meth:`ModelEvaluator.resume`.
    Wraps an :class:`~simulation.handle.EvaluationHandle` together with the
    context needed to produce domain-specific :class:`ModelOutput` instances
    via :meth:`snapshot`.

    Do not construct directly; obtain via :meth:`ModelEvaluator.start` or
    :meth:`ModelEvaluator.resume`.

    Parameters
    ----------
    handle : EvaluationHandle
        The underlying incremental engine handle.
    evaluator : ModelEvaluator
        The evaluator that owns this run; used for :meth:`~ModelEvaluator.post_process`
        and :meth:`~ModelEvaluator.attach_resume` in :meth:`snapshot`.
    scenario : Scenario
        The scenario under evaluation.
    config : EvaluationConfig
        Evaluation parameters; ``config.ensemble_size`` is the default
        batch size for :meth:`extend`.
    """

    def __init__(
        self,
        handle: EvaluationHandle,
        evaluator: ModelEvaluator[Any, OutputT],
        scenario: Scenario,
        config: EvaluationConfig,
    ) -> None:
        self._handle = handle
        self._evaluator = evaluator
        self._scenario = scenario
        self._config = config

    @property
    def result(self) -> EvaluationResult:
        """The current accumulated :class:`~simulation.evaluation.EvaluationResult`.

        Returns the raw engine result.  Call :meth:`snapshot` to get the
        domain-specific :class:`ModelOutput`.
        """
        return self._handle.result

    def extend(self, n: int | None = None) -> None:
        """Draw additional Monte Carlo samples and merge into the accumulated result.

        Parameters
        ----------
        n : int, optional
            Number of new samples to draw.  Defaults to
            ``config.ensemble_size`` when ``None``.
        """
        self._handle.extend(n if n is not None else self._config.ensemble_size)

    def snapshot(self, *, resumable: bool = False) -> OutputT:
        """Materialise the current accumulated result as a domain :class:`ModelOutput`.

        Calls :meth:`~ModelEvaluator.post_process` on the current
        :attr:`result` without advancing the ensemble.  Each call produces
        an independent :class:`ModelOutput`; the run is unaffected.

        Parameters
        ----------
        resumable : bool, optional
            When ``True``, attaches the full resume payload to the returned
            output (via :meth:`~ModelEvaluator.attach_resume`) so that
            :meth:`ModelEvaluator.resume` can later reconstruct this handle
            from the saved output.  Defaults to ``False``.

        Returns
        -------
        OutputT
            The domain-specific evaluation output.
            :attr:`~ModelOutput.is_resumable` is ``True`` iff *resumable*
            was ``True``.
        """
        output = self._evaluator.post_process(self._scenario, self._handle.result)
        if resumable:
            self._evaluator.attach_resume(output, self._handle.result)
        return output


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


def _own_index_value(idx: GenericIndex) -> Any:
    """Return *idx*'s own default value, mirroring the pre-#211 ``.value`` semantics.

    Used by :meth:`ModelEvaluator.get_index_diffs` and
    :meth:`ModelEvaluator.get_model_values` as the no-override baseline: the
    frozen distribution for a :class:`~model.index.DistributionIndex`, else
    the concrete scalar/array default (``None`` when abstract or
    formula-backed).
    """
    if isinstance(idx, DistributionIndex):
        return idx.frozen_distribution
    if isinstance(idx, Index):
        return idx.concrete_default
    return None


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------


class ModelEvaluator(ABC, Generic[ModelT, OutputT]):
    """Abstract base class for domain-specific scenario evaluators.

    Each domain package subclasses :class:`ModelEvaluator` and binds
    ``ModelT`` to its concrete :class:`~dt_model.model.model.Model` or
    :class:`~dt_model.model.model_variant.ModelVariant` subclass and
    ``OutputT`` to its concrete :class:`ModelOutput` subclass.
    The application layer then drives the uniform lifecycle::

        evaluator = DomainEvaluator(model)

        # one-shot (no resume payload):
        output = evaluator.evaluate(scenario, config)

        # incremental (resumable snapshot):
        run    = evaluator.start(scenario, config)
        run.extend(200)
        output = run.snapshot(resumable=True)
        data   = output.to_dict()                          # save
        output2 = DomainOutput.from_dict(data)             # load
        run2   = evaluator.resume(scenario, output2, config)

    **Abstract interface**: subclasses must implement :meth:`input_schema`.
    They should implement *either* :meth:`post_process` (letting the base
    handle :meth:`evaluate`, :meth:`start`, and :meth:`run_async`) or
    override those methods directly for full control.

    **Override points for** :meth:`evaluate` **/**:meth:`start`:

    1. *Simple DistributionEnsemble model* — implement :meth:`post_process`
       only; override :attr:`eval_functions` if the model needs custom engine
       callables.  The base templates handle everything else.
    2. *Parametric / complex model* — override :meth:`evaluate` (and
       :meth:`start` / :meth:`run_async`) in full, then also override
       :meth:`extract_resume_state` to reconstruct the parameter arrays.
       Call :meth:`attach_resume` explicitly in :meth:`~IncrementalRun.snapshot`
       overrides when you want the output to be resumable.

    The :class:`ModelOutput` side of the contract uses :meth:`_serialize` /
    :meth:`_deserialize` for the summary layer.

    Parameters
    ----------
    model : ModelT
        The model this evaluator operates on.  Stored as ``self._model`` and
        used by the default implementations of :meth:`get_index_diffs`,
        :meth:`get_model_values`, and :meth:`input_schema`.
    """

    def __init__(self, model: ModelT) -> None:
        self._model: ModelT = model

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def input_schema(self) -> dict[str, dict[str, Any]]:
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

    # ------------------------------------------------------------------
    # Overridable engine configuration
    # ------------------------------------------------------------------

    @property
    def eval_functions(self) -> dict[str, Functor] | None:
        """Custom functions injected into the engine executor.

        Return a ``{name: functor}`` dict or ``None`` (default) when no
        custom functions are needed.  Override in subclasses that use
        model-specific callables (e.g. ``_ts_solve`` in the Bologna model).

        The value is used by the default :meth:`evaluate`, :meth:`run_async`,
        and :meth:`extract_resume_state` implementations.

        Returns
        -------
        dict[str, Functor] or None
        """
        return None

    @property
    def eval_backend(self) -> type[NumpyBackend]:
        """Computation backend class.

        Defaults to :class:`~dt_model.engine.numpybackend.executor.NumpyBackend`.
        Override only when a different backend is required.

        Returns
        -------
        type[NumpyBackend]
        """
        return NumpyBackend

    def make_ensemble(self, scenario: Scenario, config: EvaluationConfig) -> Any:
        """Construct the ensemble for a blocking :meth:`evaluate` call.

        Default implementation returns a
        :class:`~dt_model.simulation.ensemble.DistributionEnsemble` of size
        ``config.ensemble_size``.  Override for models that use a different
        ensemble type (e.g. :class:`~dt_model.simulation.ensemble.CrossProductEnsemble`
        with a parameter grid — in that case override :meth:`evaluate` as a whole).

        Parameters
        ----------
        scenario : Scenario
            The scenario being evaluated.
        config : EvaluationConfig
            Evaluation parameters.

        Returns
        -------
        Any
            An ensemble compatible with :meth:`~simulation.evaluation.Evaluation.evaluate`.
        """
        return DistributionEnsemble(scenario, config.ensemble_size)

    def attach_resume(self, output: ModelOutput, result: EvaluationResult) -> None:
        """Encode *result* and store it as the resume payload on *output*.

        Convenience wrapper around :func:`_encode_result` +
        :meth:`~ModelOutput._store_resume`.  Called by
        :meth:`IncrementalRun.snapshot` when ``resumable=True``.  Evaluators
        that override :meth:`start` or need programmatic control over the
        resume payload can call this directly.

        Parameters
        ----------
        output : ModelOutput
            The output object to attach the resume payload to.
        result : EvaluationResult
            The raw evaluation result to encode.
        """
        output._store_resume(_encode_result(result, self._model.indexes))

    def post_process(self, scenario: Scenario, result: EvaluationResult) -> OutputT:
        """Convert a raw :class:`~simulation.evaluation.EvaluationResult` to ``OutputT``.

        Called by the default :meth:`evaluate` and :meth:`run_async`
        implementations.  Override when using these defaults; leave
        unimplemented if you override :meth:`evaluate` directly.

        Parameters
        ----------
        scenario : Scenario
            The scenario that was evaluated.
        result : EvaluationResult
            The raw engine result.

        Returns
        -------
        OutputT
            The domain-specific :class:`ModelOutput` for this evaluation.

        Raises
        ------
        NotImplementedError
            Always, in the base implementation.  Override in subclasses that
            rely on the default :meth:`evaluate` template.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement post_process() when using the default evaluate() template."
        )

    def evaluate(self, scenario: Scenario, config: EvaluationConfig) -> OutputT:
        """Run a blocking evaluation and return the domain output.

        Concrete template method.  Builds the ensemble via
        :meth:`make_ensemble`, calls
        :class:`~simulation.evaluation.Evaluation` with
        :attr:`eval_functions` and :attr:`eval_backend`, then delegates
        post-processing to :meth:`post_process`.

        The returned output is **not resumable** (:attr:`~ModelOutput.is_resumable`
        is always ``False``).  Use :meth:`start` to obtain an
        :class:`IncrementalRun` from which :meth:`~IncrementalRun.snapshot`
        can produce a resumable output.

        Override entirely for models that require a parameter grid or a
        different ensemble setup (e.g. Molveno).  Override
        :meth:`post_process` instead when only the output construction
        differs (e.g. Bologna).

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
        ensemble = self.make_ensemble(scenario, config)
        result = Evaluation(scenario).evaluate(
            ensemble=ensemble,
            functions=self.eval_functions,
            backend=self.eval_backend,
        )
        return self.post_process(scenario, result)

    def start(
        self,
        scenario: Scenario,
        config: EvaluationConfig,
        *,
        rng: np.random.Generator | None = None,
    ) -> IncrementalRun[OutputT]:
        """Run an initial batch and return an incremental handle.

        Draws ``config.ensemble_size`` samples, executes the plan, and wraps
        the result in an :class:`IncrementalRun`.  Use
        :meth:`IncrementalRun.extend` to draw additional samples and
        :meth:`IncrementalRun.snapshot` to produce a :class:`ModelOutput`.

        Parameters
        ----------
        scenario : Scenario
            The scenario to evaluate.
        config : EvaluationConfig
            Evaluation parameters.  ``config.ensemble_size`` sets the initial
            batch size and the default increment for subsequent
            :meth:`~IncrementalRun.extend` calls.
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.  When ``None``, a
            fresh :func:`numpy.random.default_rng` is created.

        Returns
        -------
        IncrementalRun[OutputT]
            Handle seeded with the initial result.
        """
        handle = EvaluationHandle.evaluate(
            Evaluation(scenario),
            config.ensemble_size,
            ensemble_recipe=self.make_ensemble(scenario, config),
            functions=self.eval_functions,
            backend=self.eval_backend,
            rng=rng,
        )
        return IncrementalRun(handle, self, scenario, config)

    def run_async(self, scenario: Scenario, config: EvaluationConfig) -> ModelRunHandle[OutputT]:
        """Submit an engine-level async evaluation and return a handle immediately.

        Concrete tier-3 default.  Calls
        :meth:`AsyncEvaluationHandle.evaluate` with
        :attr:`eval_functions` and :attr:`eval_backend`, then wraps the
        result in a :class:`ModelRunHandle` whose post-processor is
        :meth:`post_process`.  The returned output is **not resumable**
        (consistent with the one-shot :meth:`evaluate`).

        Override for models that need synchronous pre-computation or a
        parameter grid before the async engine call (e.g. Molveno).

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
        """
        async_handle = AsyncEvaluationHandle.evaluate(
            Evaluation(scenario),
            config.ensemble_size,
            functions=self.eval_functions,
            backend=self.eval_backend,
        )
        return ModelRunHandle(
            future=async_handle.future,
            post_process=lambda result: self.post_process(scenario, result),
        )

    def extract_resume_state(self, output: OutputT) -> ResumeState:
        """Extract the resume payload from a previously saved output.

        Concrete default for parameter-free
        :class:`~dt_model.simulation.ensemble.DistributionEnsemble`
        evaluations.  Decodes the stored result (encoded by
        :meth:`attach_resume` via :func:`_encode_result`) and re-injects
        :attr:`eval_functions` and :attr:`eval_backend`.

        Override when the evaluation used a parameter grid (the parameter
        arrays must be decoded and passed back as ``parameters``).

        .. note::
            The default implementation reads ``parameters`` directly from
            ``result.parameter_values`` as decoded by :func:`_decode_result`.
            This correctly handles both parameter-free evaluations (empty dict)
            and evaluations with a parameter grid, so overriding is only needed
            for unusual resume-state requirements beyond parameters and functions.

        Parameters
        ----------
        output : OutputT
            A :class:`ModelOutput` for which ``is_resumable`` is ``True``.

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
        assert output._serialized_resume is not None, "extract_resume_state called on non-resumable output"
        result = _decode_result(output._serialized_resume, self._model.indexes)
        return ResumeState(
            result=result,
            parameters=dict(result.parameter_values),
            functions=self.eval_functions,
            backend=self.eval_backend,
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
            idx.name: f"was {_format_value(_own_index_value(idx))} \u2192 now {_format_value(override_val)}"
            for idx, override_val in scenario.overrides.items()
        }

    def get_model_values(self, scenario: Scenario) -> dict[str, Any]:
        """Return the effective value of every model index under *scenario*.

        For indexes that have an active override the override value is
        returned; for all others the index's own default value is used via
        :func:`_own_index_value` (which may be ``None`` for abstract indexes
        with no override).

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
        return {idx.name: (active[idx] if idx in active else _own_index_value(idx)) for idx in self._model.indexes}

    # ------------------------------------------------------------------
    # Resume template method
    # ------------------------------------------------------------------

    def resume(
        self,
        scenario: Scenario,
        output: OutputT,
        config: EvaluationConfig,
        *,
        rng: np.random.Generator | None = None,
    ) -> IncrementalRun[OutputT]:
        """Reconstruct an :class:`IncrementalRun` from a previously saved output.

        Template method.  Checks that *output* is resumable, delegates
        deserialisation to :meth:`extract_resume_state`, rebuilds the
        evaluation plan, and wraps the result in an :class:`IncrementalRun`.
        Use :meth:`~IncrementalRun.extend` to draw additional samples and
        :meth:`~IncrementalRun.snapshot` to materialise a :class:`ModelOutput`.

        Parameters
        ----------
        scenario : Scenario
            The scenario used to rebuild the evaluation plan.
        output : OutputT
            A previously produced :class:`ModelOutput`.  Must have
            ``is_resumable == True`` (i.e. produced via
            :meth:`IncrementalRun.snapshot` with ``resumable=True``).
        config : EvaluationConfig
            Evaluation parameters.  ``config.ensemble_size`` is used as the
            default increment size for subsequent
            :meth:`~IncrementalRun.extend` calls.
        rng : numpy.random.Generator, optional
            Random number generator for reproducible extension sampling.
            When ``None``, a fresh :func:`numpy.random.default_rng` is used.

        Returns
        -------
        IncrementalRun[OutputT]
            Handle seeded with the saved result.

        .. note::

            The resumed handle rebuilds its ensemble recipe from
            :class:`~simulation.ensemble.DistributionEnsemble` using the
            public *scenario* argument, but does **not** restore the frozen
            sample snapshot (``_ensemble``).  When
            ``extend(extra_parameters=…)`` is called on the underlying handle,
            the abstract index values are reconstructed from the saved result
            state so that the common-random-numbers guarantee is preserved.

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
        state = self.extract_resume_state(output)
        evaluation = Evaluation(scenario)
        plan = evaluation.build_plan()
        # Rebuild the sampler recipe from the (public) scenario so the resumed
        # handle can draw further samples.  draw_batch() takes the per-call size,
        # so the recipe's nominal size is immaterial; exclude the parameter
        # indexes exactly as EvaluationHandle.evaluate() does.
        ensemble_recipe = DistributionEnsemble(scenario, config.ensemble_size, exclude=frozenset(state.parameters))
        handle = EvaluationHandle(
            evaluation=evaluation,
            plan=plan,
            result=state.result,
            rng=rng if rng is not None else np.random.default_rng(),
            parameters=state.parameters,
            parameter_axes=state.parameter_axes,
            ensemble_recipe=ensemble_recipe,
            functions=state.functions,
            backend=state.backend,
        )
        return IncrementalRun(handle, self, scenario, config)
