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
    PV_excursionists,
    PV_tourists,
)


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

    # Ensure that we have the expected constraints
    assert len(got) == len(expect)

    # Collect all differences for reporting
    failures = []

    # Proceed to check ~equality for each constraint
    for key in expect.keys():
        expect_c = expect[key]
        got_c = got[key]

        # Basic shape check
        if expect_c.shape != got_c.shape:
            failures.append(f"Shape mismatch for {key}: {expect_c.shape} vs {got_c.shape}")
            continue

        # Check if values are close enough
        if not np.allclose(expect_c, got_c, rtol=1e-5, atol=1e-8):
            diff_info = f"\n--- expected/{key}\n+++ got/{key}\n"

            # Convert arrays to formatted strings for comparison line by line
            for i in range(expect_c.shape[0]):
                row_expect = [f"{x:.8f}" for x in expect_c[i]]
                row_got = [f"{x:.8f}" for x in got_c[i]]

                # If this row has differences
                if not np.allclose(expect_c[i], got_c[i], rtol=1e-5, atol=1e-8):
                    diff_info += f"-{row_expect}\n"
                    diff_info += f"+{row_got}\n"
                else:
                    diff_info += f" {row_expect}\n"

            failures.append(diff_info)

    # If we have any failures, report them all at once
    if failures:
        failure_message = "Model comparison failed:\n" + "\n".join(failures)
        assert False, failure_message
