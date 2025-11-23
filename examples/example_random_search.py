from sigil.tuning import RandomSearch
from sigil.linear import LinearRegression
import numpy as np


noise = np.random.randn(100, 1)
X = 2 * np.random.rand(100, 1)
# y = 3x + 10 + noise
Y = ((3 * X) + 10 + noise).flatten()

params = {
    "discrete": {
        "learning_rate": [0.01, 0.03, 0.1, 0.3],
        "n_iterations": [500, 1000, 5000],
    },
    "continuous": {
        "learning_rate": (0.0, 0.5),
        "n_iterations": (500, 5000),
    },
}

for key, values in params.items():
    print(f"Random search with {key} parameters:")

    random_search = RandomSearch(LinearRegression, values, cv=5)
    random_search.fit(X, Y)

    print(f"Results for {key} parameters:")
    print("Best Parameters:", random_search.best_params)
    print("Best Score:", random_search.best_score)
