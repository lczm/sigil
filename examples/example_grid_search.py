from sigil.tuning import GridSearch
from sigil.linear import LinearRegression
import numpy as np


noise = np.random.randn(100, 1)
X = 2 * np.random.rand(100, 1)
# y = 3x + 10 + noise
Y = ((3 * X) + 10 + noise).flatten()

param_grid = {
    "learning_rate": [0.01, 0.03, 0.1, 0.3],
    "n_iterations": [500, 1000, 5000],
}

grid_search = GridSearch(LinearRegression, param_grid, cv=5)
grid_search.fit(X, Y)

best_model = grid_search.best_model
print("Best Parameters:", grid_search.best_params)
print("Best Score:", grid_search.best_score)
