from sigil.linear import QuadraticRegression
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__ == "__main__":
    # Sample data (1 feature)
    start = time.time()
    X = np.array([[1], [3], [2], [5], [7], [9], [10]])
    y = np.array([3, 5, 7, 9, 15, 19, 23])

    # Train model
    model = QuadraticRegression()
    model.fit(X, y)

    # Predictions on new data
    z = np.array([[5], [6], [13]])
    predictions = model.predict(z)
    end = time.time()

    # Plot training data
    plt.scatter(X, y, color="blue", label="Training data")

    # Plot predictions
    plt.scatter(z, predictions, color="red", label="Predictions")

    # Plot regression line
    # Basically, we generate points between min and max of X for a smooth line
    # Then we use the model to predict y values for these points, then draw the line
    minimum_value = min(min(X), min(z))
    maximum_value = max(max(X), max(z))
    x_line = np.linspace(minimum_value, maximum_value, 100).reshape(-1,1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color="green", label="Regression line")


    plt.title("Quadratic Regression Example")
    plt.xlabel("Feature 1 (One Dimension)")
    plt.ylabel("Target")
    plt.legend()
    plt.show()

    print("Predictions:", predictions)
    print(f"Time taken: {end-start:.2f}")
