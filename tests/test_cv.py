"""Tests for ContextVariable classes."""

# SPDX-License-Identifier: Apache-2.0


from dt_model import (
    UniformCategoricalContextVariable,
    CategoricalContextVariable,
    ContinuousContextVariable,
)

import pytest
import scipy


@pytest.fixture
def uniform_cv():
    return UniformCategoricalContextVariable("Uniform", ['a', 'b', 'c', 'd'])


@pytest.fixture
def categorical_cv():
    return CategoricalContextVariable("Categorical", {'a': 0.1, 'b': 0.2, 'c': 0.3, 'd': 0.4})


@pytest.fixture
def continuous_cv():
    return ContinuousContextVariable("Continuous", scipy.stats.norm(3, 1))

@pytest.mark.parametrize(
    "cv_fixture_name,sizes,values",
    [
        ("uniform_cv", [1,2,4,8], ['a', 'b', 'c']),
        ("categorical_cv", [1,2,4,8], ['a', 'b', 'c']),
        ("continuous_cv", [1,2,4,8], [2.1, 3.0, 3.9]),
    ]
)
def test_cv(cv_fixture_name, sizes, values, request):
    cv = request.getfixturevalue(cv_fixture_name)
    print(f'Testing: {cv.name} (support size = {cv.support_size()})')
    for s in sizes:
        print(f'    Size {s}: {cv.sample(s)}')
    for s in sizes:
        print(f'    Size {s} - force_sample: {cv.sample(s, force_sample=True)}')
    for s in sizes:
        print(f'    Size {s} - subset {values}: {cv.sample(s, subset=values)}')
    for s in sizes:
        print(f'    Size {s} - subset {values} - force_sample: {cv.sample(s, subset=values, force_sample=True)}')
