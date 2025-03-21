"""Tests for the dt_model.examples.molveno.overtourism module."""

import random

import numpy as np

from dt_model.examples.molveno.overtourism import (
    C_accommodation,
    C_beach,
    C_food,
    C_parking,
    Constraint,
    CV_season,
    CV_weather,
    CV_weekday,
    M_Base,
    PV_excursionists,
    PV_tourists,
    Symbol,
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
        C_parking: np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        C_beach: np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        C_accommodation: np.array([
            [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
            [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
            [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
            [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
            [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
            [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
        ]),
        C_food: np.array([
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.77777778, 0.0],
            [1.0, 1.0, 1.0, 0.77777778, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
    }

    for key, value in got.items():
        print(key, value)

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
            # Calculate differences for diagnosis
            abs_diff = np.abs(expect_c - got_c)
            rel_diff = abs_diff / (np.abs(expect_c) + 1e-10)  # Avoid division by zero

            # Find the worst offenders (highest absolute difference)
            max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

            diff_info = (
                f"Constraint {key} has differences:\n"
                f"  Max absolute diff: {np.max(abs_diff):.6e} at index {max_diff_idx}\n"
                f"  Mean absolute diff: {np.mean(abs_diff):.6e}\n"
                f"  Max relative diff: {np.max(rel_diff):.6f}\n"
                f"  Values at max diff: orig={expect_c[max_diff_idx]:.6f}, yak={got_c[max_diff_idx]:.6f}\n"
                f"  First few pairs of values (orig, yak):\n"
            )

            # Sample some value pairs for comparison
            flat_expect = expect_c.flatten()
            flat_got = got_c.flatten()
            sample_size = min(5, len(flat_expect))

            for i in range(sample_size):
                diff_info += f"    [{i}]: {flat_expect[i]:.6f}, {flat_got[i]:.6f} (diff: {abs_diff.flatten()[i]:.6e})\n"

            failures.append(diff_info)

    # If we have any failures, report them all at once
    if failures:
        failure_message = "Model comparison failed:\n" + "\n".join(failures)
        assert False, failure_message
