"""Tests for the dt_model.examples.molveno.overtourism module."""

import random

import numpy as np
from sympy import Symbol

from dt_model import Constraint
from dt_model.examples.molveno.overtourism import (
    C_accommodation,
    C_beach,
    C_food,
    C_parking,
    CV_season,
    CV_weather,
    CV_weekday,
    M_Base,
    M_MoreParking,
    PV_excursionists,
    PV_tourists,
)


def compare_constraint_results(got, expect):
    """Helper function to compare constraint results and return any failures."""
    # Ensure that we have the expected constraints
    if len(got) != len(expect):
        return [f"Constraint count mismatch: expected {len(expect)}, got {len(got)}"]

    # Collect all differences for reporting
    failures = []

    # Match constraints by index since we can't match by identity
    got_keys = list(got.keys())
    expect_keys = list(expect.keys())

    for i in range(len(got_keys)):
        got_key = got_keys[i]
        expect_key = expect_keys[i]

        expect_c = expect[expect_key]
        got_c = got[got_key]

        # Basic shape check
        if expect_c.shape != got_c.shape:
            failures.append(f"Shape mismatch for constraint {i}: {expect_c.shape} vs {got_c.shape}")
            continue

        # Check if values are close enough
        if not np.allclose(expect_c, got_c, rtol=1e-5, atol=1e-8):
            diff_info = f"\n--- expected/constraint {i}\n+++ got/constraint {i}\n"

            # Convert arrays to formatted strings for comparison line by line
            for j in range(expect_c.shape[0]):
                row_expect = [f"{x:.8f}" for x in expect_c[j]]
                row_got = [f"{x:.8f}" for x in got_c[j]]

                # If this row has differences
                if not np.allclose(expect_c[j], got_c[j], rtol=1e-5, atol=1e-8):
                    diff_info += f"-{row_expect}\n"
                    diff_info += f"+{row_got}\n"
                else:
                    diff_info += f" {row_expect}\n"

            failures.append(diff_info)

    return failures


def test_fixed_ensemble():
    """Evaluate the model using a fixed ensemble."""
    # Reference the base model
    model = M_Base

    # Reset the model to ensure we can re-evaluate it
    model.reset()

    # Manually create a specific ensemble to use
    fixed_orig_situation = {
        CV_weekday: Symbol("monday"),
        CV_season: Symbol("high"),
        CV_weather: Symbol("good"),
    }

    # Manually create fixed tourist and excursionist values
    tourists = np.array([1000, 2000, 5000, 10000, 20000, 50000])
    excursionists = np.array([1000, 2000, 5000, 10000, 20000, 50000])

    # Reset the random seed to ensure reproducibility
    #
    # See https://xkcd.com/221/
    np.random.seed(4)
    random.seed(4)

    # Evaluate model with fixed inputs and a single ensemble member
    model.evaluate(
        {PV_tourists: tourists, PV_excursionists: excursionists},
        [(1.0, fixed_orig_situation)],
    )

    # Obtain the constraints evaluation results
    got = model.field_elements
    assert got is not None

    # Define the expected constraints evaluation result
    expect: dict[Constraint, np.ndarray] = {
        C_parking: np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        C_beach: np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        C_accommodation: np.array(
            [
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
            ]
        ),
        C_food: np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.77777778, 0.0],
                [1.0, 1.0, 1.0, 0.77777778, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    }

    # Use the helper function to compare results
    failures = compare_constraint_results(got, expect)

    # If we have any failures, report them all at once
    if failures:
        failure_message = "Model comparison failed:\n" + "\n".join(failures)
        assert False, failure_message


def test_more_parking_model():
    """Test the more parking model."""
    # Reference the modified model
    model = M_MoreParking

    # Reset the model to ensure we can re-evaluate it
    model.reset()

    # Manually create a specific ensemble to use
    fixed_orig_situation = {
        CV_weekday: Symbol("monday"),
        CV_season: Symbol("high"),
        CV_weather: Symbol("good"),
    }

    # Manually create fixed tourist and excursionist values
    tourists = np.array([1000, 2000, 5000, 10000, 20000, 50000])
    excursionists = np.array([1000, 2000, 5000, 10000, 20000, 50000])

    # Reset the random seed to ensure reproducibility
    np.random.seed(4)
    random.seed(4)

    # Evaluate model with fixed inputs and a single ensemble member
    model.evaluate(
        {PV_tourists: tourists, PV_excursionists: excursionists},
        [(1.0, fixed_orig_situation)],
    )

    # Obtain the constraints evaluation results
    got = model.field_elements
    assert got is not None

    # Define the expected constraints evaluation result for the more parking model
    expect = {
        0: np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],           # Now allows all excursionist values at 1000 tourists
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.74985994],    # Now allows more excursionists at 2000 tourists
                [1.0, 1.0, 1.0, 1.0, 0.35994398, 0.0],    # Now allows more excursionists at 5000 tourists
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        1: np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        2: np.array(
            [
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
            ]
        ),
        3: np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.77777778, 0.0],
                [1.0, 1.0, 1.0, 0.77777778, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    }

    # Use the helper function to compare results
    failures = compare_constraint_results(got, expect)

    # If we have any failures, report them all at once
    if failures:
        failure_message = "Model comparison failed:\n" + "\n".join(failures)
        assert False, failure_message

    # Verify the model name was correctly set during variation
    assert model.name == "larger parking model"
