from sigil.linear import LinearRegression
import pytest
import numpy as np

def test_linear_regression_shape():
    X = np.random.rand(100, 2)  # 100 samples, 2 features
    y = np.random.rand(100) 

    model = LinearRegression()
    model.fit(X, y)

    # Make predictions on another set of data with SAME dimensions
    z = np.random.rand(2, 2)
    predictions = model.predict(z)

    # The output should be a numpy array
    assert isinstance(predictions, np.ndarray)
    # The input samples should match the output predicted samples
    assert predictions.shape[0] == z.shape[0]

def test_valid_output():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegression()
    model.fit(X, y)

    z = np.array([[5, 6], [6, 7]])  # Valid input for prediction
    predictions = model.predict(z)

    # The output should be a numpy array of the same length as z
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == z.shape[0]
    assert np.allclose(predictions, np.array([11, 13]), atol=1e-1)

def test_shape_mismatch():
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([3, 5, 7, 9])

    model = LinearRegression()
    model.fit(X, y)

    z_invalid = np.array([[5, 6, 7], [6, 7, 8]])  # Invalid shape

    with pytest.raises(ValueError):
        model.predict(z_invalid)
