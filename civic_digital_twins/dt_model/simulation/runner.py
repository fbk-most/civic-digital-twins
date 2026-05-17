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

import dataclasses
import importlib.metadata
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any, Generic, Self, TypeVar

import numpy as np

from ..engine.numpybackend.executor import Functor, NumpyBackend
from ..model.index import GenericIndex
from .evaluation import EvaluationResult

__all__ = [
    "EvaluationConfig",
    "IncompatibleResultError",
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

    Two backing strategies are supported:

    - **Engine-level async (tier 3)**: the future comes from
      :meth:`~simulation.evaluation.Evaluation.submit_evaluate` via
      :attr:`~simulation.handle.AsyncEvaluationHandle.future`
      (e.g. Bologna with :class:`~simulation.ensemble.DistributionEnsemble`).
    - **Protocol-level async (tier 2)**: the future wraps a blocking
      :meth:`ModelEvaluator.evaluate` call submitted to a thread pool
      (e.g. Molveno with :class:`~simulation.ensemble.CrossProductEnsemble`).

    Parameters
    ----------
    future : Future[EvaluationResult]
        The in-flight or completed computation.
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
