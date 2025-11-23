from sigil.tuning import GridSearch
from sigil.linear import LinearRegression
import pytest
import numpy as np


def test_grid_search_shape():
    noise = np.random.randn(100, 1)
    X = 2 * np.random.rand(100, 1)
    Y = ((3 * X) + 10 + noise).flatten()

    param_grid = {
        "learning_rate": [0.01, 0.03, 0.1, 0.3],
        "n_iterations": [500, 1000, 5000],
    }

    grid_search = GridSearch(LinearRegression, param_grid, cv=5)
    grid_search.fit(X, Y)

    assert( isinstance(grid_search.best_params, dict))
    assert( isinstance(grid_search.best_score, float))

def test_grid_search_invalid_params():
    noise = np.random.randn(100, 1)
    X = 2 * np.random.rand(100, 1)
    Y = ((3 * X) + 10 + noise).flatten()

    param_grid = {
        "learning_rate": [0.01, 0.03, 0.1, 0.3],
        "n_iterations": [500, 1000, 5000],
        "invalid_param": [1, 2, 3]
    }

    with pytest.raises(ValueError):
        grid_search = GridSearch(LinearRegression, param_grid, cv=5)
        grid_search.fit(X, Y)

def test_grid_search_no_params():
    noise = np.random.randn(100, 1)
    X = 2 * np.random.rand(100, 1)
    Y = ((3 * X) + 10 + noise).flatten()

    param_grid = {}

    with pytest.raises(ValueError):
        grid_search = GridSearch(LinearRegression, param_grid, cv=5)
        grid_search.fit(X, Y)   
