from sigil.linear import LogisticRegression
import numpy as np
import pytest

def test_logistic_regression_shape():
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 2, 100)

    model = LogisticRegression()
    model.fit(X, y)

    z = np.random.rand(5, 2)
    predictions = model.predict(z)

    # Output should be a numpy array of 0s and 1s
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == z.shape[0]
    assert np.isin(predictions, [0, 1]).all()

# simple example to test that it forms the right decision boundary
def test_fit_and_gate():
    # x1 x2 -> y
    # 0  0  -> 0
    # 0  1  -> 0
    # 1  0  -> 0
    # 1  1  -> 1
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])

    model = LogisticRegression(learning_rate=0.1, n_iterations=5000)
    model.fit(X, y)

    predictions = model.predict(X)
    assert np.array_equal(predictions, y)

def test_not_trained():
    model = LogisticRegression()
    z = np.array([[1, 2], [3, 4]])

    with pytest.raises(RuntimeError):
        model.predict(z)
