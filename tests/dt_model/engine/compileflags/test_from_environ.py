"""Tests for the civic_digital_twins.dt_model.engine.compileflags.from_environ func."""

# SPDX-License-Identifier: Apache-2.0

from civic_digital_twins.dt_model.engine import compileflags


def test_from_environ():
    """Tests that the from_environ function works as intended."""
    # Test for the case where the environment variable has not been set
    result = compileflags.from_environ(
        "antani",
        lambda x: None,
    )
    assert result == 0

    # Test for the case where it has been set
    result = compileflags.from_environ(
        "antani",
        lambda x: "break,trace",
    )
    assert result == compileflags.BREAK | compileflags.TRACE
