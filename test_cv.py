from dt_model import UniformCategoricalContextVariable, CategoricalContextVariable, ContinuousContextVariable

import scipy


def test_cv(cv, sizes, values):
    print(f'Testing: {cv.name} (support size = {cv.support_size()})')
    for s in sizes:
        print(f'    Size {s}: {cv.sample(s)}')
    for s in sizes:
        print(f'    Size {s} - force_sample: {cv.sample(s, force_sample=True)}')
    for s in sizes:
        print(f'    Size {s} - subset {values}: {cv.sample(s, subset=values)}')
    for s in sizes:
        print(f'    Size {s} - subset {values} - force_sample: {cv.sample(s, subset=values, force_sample=True)}')


uniform_cv = UniformCategoricalContextVariable("Uniform", ['a', 'b', 'c', 'd'])
test_cv(uniform_cv, [1,2,4,8],['a', 'b', 'c'])

cat_cv = CategoricalContextVariable("Categorical", {'a': 0.1, 'b': 0.2, 'c': 0.3, 'd': 0.4})
test_cv(cat_cv, [1,2,4,8],['a', 'b', 'c'])

cont_cv = ContinuousContextVariable("Continuous", scipy.stats.norm(3, 1))
test_cv(cont_cv, [1,2,4,8],[2.1, 3.0, 3.9])
