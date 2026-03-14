"""Tests for civic_digital_twins.dt_model.symbols.ContextVariable classes."""

# SPDX-License-Identifier: Apache-2.0

import pytest
import scipy
from overtourism_molveno.overtourism_metamodel import (
    CategoricalContextVariable,
    ContinuousContextVariable,
    UniformCategoricalContextVariable,
)


@pytest.fixture
def uniform_cv():
    """Create a UniformCategoricalContextVariable."""
    return UniformCategoricalContextVariable(
        "Uniform",
        ["a", "b", "c", "d"],
    )


@pytest.fixture
def categorical_cv():
    """Create a CategoricalContextVariable."""
    return CategoricalContextVariable(
        "Categorical",
        {
            "a": 0.1,
            "b": 0.2,
            "c": 0.3,
            "d": 0.4,
        },
    )


@pytest.fixture
def continuous_cv():
    """Create a ContinuousContextVariable."""
    return ContinuousContextVariable("Continuous", scipy.stats.norm(3, 1))


@pytest.mark.parametrize(
    "cv_fixture_name,sizes,values",
    [
        ("uniform_cv", [1, 2, 4, 8], ["a", "b", "c"]),
        ("categorical_cv", [1, 2, 4, 8], ["a", "b", "c"]),
        ("continuous_cv", [1, 2, 4, 8], [2.1, 3.0, 3.9]),
    ],
)
def test_cv(cv_fixture_name, sizes, values, request):
    """Test ContextVariable classes."""
    cv = request.getfixturevalue(cv_fixture_name)

    # Test basic sample() functionality
    # Behavior depends on whether the distribution is discrete or continuous
    for s in sizes:
        result = cv.sample(s)
        # Verify returned list has expected length
        assert isinstance(result, list), f"sample({s}) should return a list"

        support_size = cv.support_size()

        # For continuous distributions (support_size == -1): always returns nr samples
        # For discrete distributions: returns min(s, support_size) items
        if support_size == -1:
            # Continuous distribution always returns exactly nr samples
            assert len(result) == s, f"sample({s}) should return {s} items, got {len(result)}"
        else:
            # Discrete distribution returns min(s, support_size) items
            expected_len = s if s < support_size else support_size
            assert len(result) == expected_len, f"sample({s}) should return {expected_len} items, got {len(result)}"

        # Verify each element is a (float, Any) tuple
        for prob, value in result:
            assert isinstance(prob, (int, float)), f"Probability should be numeric, got {type(prob)}"
            assert prob > 0 and prob <= 1, f"Probability should be in (0, 1], got {prob}"
        # Verify probabilities sum to approximately 1.0
        prob_sum = sum(prob for prob, _ in result)
        assert abs(prob_sum - 1.0) < 1e-9, f"Probabilities should sum to 1.0, got {prob_sum}"

    # Test force_sample() always returns exactly nr samples
    for s in sizes:
        result = cv.sample(s, force_sample=True)
        assert isinstance(result, list), f"sample({s}, force_sample=True) should return a list"
        assert len(result) == s, f"sample({s}, force_sample=True) should return {s} items, got {len(result)}"
        # Verify each element is a (float, Any) tuple
        for prob, value in result:
            assert isinstance(prob, (int, float)), f"Probability should be numeric, got {type(prob)}"
            assert prob > 0, f"Probability should be positive, got {prob}"
        # Verify probabilities sum to approximately 1.0
        prob_sum = sum(prob for prob, _ in result)
        assert abs(prob_sum - 1.0) < 1e-9, f"Probabilities should sum to 1.0, got {prob_sum}"

    # Test sample() with subset parameter
    for s in sizes:
        result = cv.sample(s, subset=values)
        assert isinstance(result, list), f"sample({s}, subset={values}) should return a list"
        subset_size = len(values)
        # For continuous distributions with subset: samples from the subset
        # For discrete with subset: min(s, subset_size) items from subset
        support_size = cv.support_size()
        if support_size == -1:
            # Continuous: subset acts as evaluation points for PDF
            assert len(result) <= s, f"sample({s}, subset={values}) should return at most {s} items"
        else:
            expected_len = s if s < subset_size else subset_size
            assert len(result) == expected_len, (
                f"sample({s}, subset={values}) should return {expected_len} items, got {len(result)}"
            )
        # Verify each element is a (float, Any) tuple
        for prob, value in result:
            assert isinstance(prob, (int, float)), f"Probability should be numeric, got {type(prob)}"
            assert prob > 0, f"Probability should be positive, got {prob}"
        # Verify probabilities sum to approximately 1.0
        prob_sum = sum(prob for prob, _ in result)
        assert abs(prob_sum - 1.0) < 1e-9, f"Probabilities should sum to 1.0, got {prob_sum}"

    # Test force_sample() with subset parameter
    for s in sizes:
        result = cv.sample(s, subset=values, force_sample=True)
        assert isinstance(result, list), f"sample({s}, subset={values}, force_sample=True) should return a list"
        assert len(result) == s, (
            f"sample({s}, subset={values}, force_sample=True) should return {s} items, got {len(result)}"
        )
        # Verify each element is a (float, Any) tuple
        for prob, value in result:
            assert isinstance(prob, (int, float)), f"Probability should be numeric, got {type(prob)}"
            assert prob > 0, f"Probability should be positive, got {prob}"
        # Verify probabilities sum to approximately 1.0
        prob_sum = sum(prob for prob, _ in result)
        assert abs(prob_sum - 1.0) < 1e-9, f"Probabilities should sum to 1.0, got {prob_sum}"
