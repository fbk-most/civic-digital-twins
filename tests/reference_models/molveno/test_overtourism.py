"""Tests for the dt_model.reference_models.molveno.overtourism module."""

import random

import numpy as np

from civic_digital_twins.dt_model import Constraint, ContextVariable, Ensemble, Evaluation
from civic_digital_twins.dt_model.internal.sympyke.symbol import Symbol, SymbolValue
from civic_digital_twins.dt_model.model.instantiated_model import InstantiatedModel
from civic_digital_twins.dt_model.reference_models.molveno.overtourism import (
    CV_season,
    CV_weather,
    CV_weekday,
    M_Base,
    PV_excursionists,
    PV_tourists,
)


def compare_constraint_results(got: dict[Constraint, np.ndarray], expect: dict[str, np.ndarray]) -> list[str]:
    """Compare constraint results and return any failures."""
    # Ensure that we have the expected constraints
    if len(got) != len(expect):
        return [f"Constraint count mismatch: expected {len(expect)}, got {len(got)}"]

    # Collect all differences for reporting
    failures: list[str] = []

    # Match constraints by name
    got_by_name = {constraint.name: result for constraint, result in got.items()}

    for name, expected_result in expect.items():
        if name not in got_by_name:
            failures.append(f"Constraint '{name}' not found in results")
            continue

        actual_result = got_by_name[name]

        # Basic shape check
        if expected_result.shape != actual_result.shape:
            failures.append(f"Shape mismatch for constraint '{name}': {expected_result.shape} vs {actual_result.shape}")
            continue

        # Check if values are close enough
        if not np.allclose(expected_result, actual_result, rtol=1e-5, atol=1e-8):
            diff_info = f"\n--- expected/{name}\n+++ got/{name}\n"

            # Convert arrays to formatted strings for comparison line by line
            for j in range(expected_result.shape[0]):
                row_expect = [f"{x:.8f}" for x in expected_result[j]]
                row_got = [f"{x:.8f}" for x in actual_result[j]]

                # If this row has differences
                if not np.allclose(expected_result[j], actual_result[j], rtol=1e-5, atol=1e-8):
                    diff_info += f"-{row_expect}\n"
                    diff_info += f"+{row_got}\n"
                else:
                    diff_info += f" {row_expect}\n"

            failures.append(diff_info)

    # Check for unexpected constraints
    for name in got_by_name:
        if name not in expect:
            failures.append(f"Unexpected constraint found: '{name}'")

    return failures


def test_fixed_ensemble():
    """Evaluate the model using a fixed ensemble."""
    # Reference the base model
    model = M_Base

    # Instantiate the model
    instantiated_model = InstantiatedModel(model)

    # Manually create a specific ensemble to use
    fixed_orig_situation: dict[ContextVariable, SymbolValue] = {
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
    evaluation = Evaluation(instantiated_model, [(1.0, fixed_orig_situation)])
    evaluation.evaluate_grid({PV_tourists: tourists, PV_excursionists: excursionists})

    # Obtain the constraints evaluation results
    got = evaluation.field_elements
    assert got is not None

    # Define the expected constraints evaluation result
    expect: dict[str, np.ndarray] = {
        "parking": np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "beach": np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        "accommodation": np.array(
            [
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
                [1.0, 1.0, 8.91250437e-01, 8.09024620e-06, 0.0, 0.0],
            ]
        ),
        "food": np.array(
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

    # Verify the model name was correctly set
    assert model.name == "base model"
    assert instantiated_model.name == "base model"


def test_multiple_ensemble_members():
    """Test with multiple ensemble members to catch shape issues."""
    instantiated_model = InstantiatedModel(M_Base)

    # Create an ensemble with multiple members
    ensemble = []
    for i in range(10):  # Use 10 ensemble members
        situation = {
            CV_weekday: Symbol("monday"),
            CV_season: Symbol("high"),
            CV_weather: Symbol("good" if i % 2 == 0 else "bad"),
        }
        ensemble.append((0.1, situation))  # Equal weights summing to 1

    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])

    # Evaluate the model with multiple ensemble members
    evaluation = Evaluation(instantiated_model, ensemble)
    result = evaluation.evaluate_grid({PV_tourists: tourists, PV_excursionists: excursionists})

    # Just check that it runs without error and returns a result
    assert result is not None
    assert evaluation.field_elements is not None


def test_bug_37():
    """Test ensuring that we handle https://github.com/fbk-most/dt-model/issues/37."""
    instantiated_model = InstantiatedModel(M_Base)

    # Create a situation containing multiple weather symbols
    #
    # See https://github.com/fbk-most/dt-model/issues/37.
    situation: dict[ContextVariable, list[SymbolValue]] = {
        CV_weather: [Symbol("good"), Symbol("unsettled"), Symbol("bad")]
    }
    ensemble = Ensemble(instantiated_model, situation, cv_ensemble_size=20)

    # This would have failed before https://github.com/fbk-most/dt-model/pull/38.
    tourists = np.array([1000, 5000, 10000])
    excursionists = np.array([1000, 5000, 10000])
    evaluation = Evaluation(instantiated_model, ensemble)
    result = evaluation.evaluate_grid({PV_tourists: tourists, PV_excursionists: excursionists})

    # Just check that it runs without error and returns a result
    assert result is not None
    assert evaluation.field_elements is not None
