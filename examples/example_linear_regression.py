from sigil.linear import LinearRegression
import numpy as np
import time

if __name__ == "__main__":
    # Sample data
    start=time.time()
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([3, 5, 7, 9])

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions on another set of data with SAME dimensions
    z = np.array([[5, 6], [6, 7]])
    predictions = model.predict(z)
    end=time.time()
    print("Predictions:", predictions)
    print(f"Time taken: {end-start:.2f}")
