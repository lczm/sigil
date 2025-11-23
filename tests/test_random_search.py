from sigil.tuning import RandomSearch
from sigil.linear import LinearRegression
import pytest
import numpy as np


def test_random_search_discrete():
    noise = np.random.randn(100, 1)
    X = 2 * np.random.rand(100, 1)
    # y = 3x + 10 + noise
    Y = ((3 * X) + 10 + noise).flatten()

    params = {
        "learning_rate": [0.01, 0.03, 0.1, 0.3],
        "n_iterations": [500, 1000, 3000],
    }

    random_search = RandomSearch(LinearRegression, params, cv=5)
    random_search.fit(X, Y)

    assert isinstance(random_search.best_params, dict)
    assert isinstance(random_search.best_score, float)


def test_random_search_continuous():
    noise = np.random.randn(100, 1)
    X = 2 * np.random.rand(100, 1)
    # y = 3x + 10 + noise
    Y = ((3 * X) + 10 + noise).flatten()

    params = {
        "learning_rate": (0.0, 0.1),
        "n_iterations": (500, 1000),
    }

    random_search = RandomSearch(LinearRegression, params, cv=5)
    random_search.fit(X, Y)

    assert isinstance(random_search.best_params, dict)
    assert isinstance(random_search.best_score, float)


def test_random_search_invalid_params():
    noise = np.random.randn(100, 1)
    X = 2 * np.random.rand(100, 1)
    # y = 3x + 10 + noise
    Y = ((3 * X) + 10 + noise).flatten()

    params = {
        "learning_rate": [0.01, 0.03, 0.1, 0.3],
        "n_iterations": [500, 1000, 3000],
        "invalid_param": [1, 2, 3],
    }

    with pytest.raises(ValueError):
        random_search = RandomSearch(LinearRegression, params, cv=5)
        random_search.fit(X, Y)


def test_random_search_no_params():
    noise = np.random.randn(100, 1)
    X = 2 * np.random.rand(100, 1)
    # y = 3x + 10 + noise
    Y = ((3 * X) + 10 + noise).flatten()

    params = {}

    with pytest.raises(ValueError):
        random_search = RandomSearch(LinearRegression, params, cv=5)
        random_search.fit(X, Y)


def test_random_search_invalid_range():
    noise = np.random.randn(100, 1)
    X = 2 * np.random.rand(100, 1)
    # y = 3x + 10 + noise
    Y = ((3 * X) + 10 + noise).flatten()

    params = {
        "learning_rate": (0.5, 0.1),  # Invalid range
        "n_iterations": (500, 1000),
    }

    with pytest.raises(ValueError):
        random_search = RandomSearch(LinearRegression, params, cv=5)
        random_search.fit(X, Y)
