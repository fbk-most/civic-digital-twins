"""
The compileflags package defines the flags used by the compiler engine.

We centralize the definition of flags to avoid defining flags into each package
and ending up with incompatible compiler engine flags.
"""

from typing import Callable
import os


TRACE = 1 << 0
"""Indicates that we should trace execution."""

BREAK = 1 << 1
"""Indicates that we should break execution after evaluation."""

_flagnames: dict[str, int] = {
    "break": BREAK,
    "trace": TRACE,
}
"""Maps the lowercase name of the flag to its value."""


def from_environ(
    varname: str = "DTMODEL_ENGINE_FLAGS",
    getenv: Callable[[str], str|None] = os.getenv,
) -> int:
    """Read flags from a specific environment variable.

    The format for the flags is the following:

        <key>[,<key>,...]

    where <key> is the case-insensitive name of an existing flag.

    For example:

        export DTMODEL_ENGINE_FLAGS=trace,break

    cases this function to return:

        TRACE|BREAK

    Arguments
    ---------
    varname: the name of the environment variable (default: `DTMODEL_ENGINE_FLAGS`).
    getenv: the function to read the environment variable (default: os.getenv).
    """
    flags: int = 0
    for value in (getenv(varname) or "").split(","):
        flags |= _flagnames.get(value.strip().lower(), 0)
    return flags


defaults = from_environ()
"""Default compile flags initialized from the `DTMODEL_ENGINE_FLAGS` environment variable."""
