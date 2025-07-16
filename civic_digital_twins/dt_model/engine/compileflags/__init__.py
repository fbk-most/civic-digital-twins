"""
The compileflags package defines the flags used by the compiler engine.

We centralize the definition of flags to avoid defining flags into each package
and ending up with incompatible compiler engine flags.
"""

TRACE = 1 << 0
"""Indicates that we should trace execution."""

BREAK = 1 << 1
"""Indicates that we should break execution after evaluation."""
