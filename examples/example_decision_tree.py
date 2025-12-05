import numpy as np
from sigil.tree import DecisionTreeClassifier, DecisionTreeRegressor

X_classifier = np.array([[1, 2], [1.5, 1.8], [5, 6], [6, 7], [1, 0.5]])
y_classifier = np.array([0, 0, 1, 1, 0])

clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_classifier, y_classifier)

test_input_cls = np.array([[1.1, 2.1], [5.5, 6.5]])
preds_cls = clf.predict(test_input_cls)
print(f"Inputs:\n{test_input_cls}")
# Should be [0, 1]
print(f"Predictions: {preds_cls}")

print("\n--- Testing Decision Tree Regressor ---")
X_regression = np.array([[1], [2], [3], [4], [5]])
y_regression = np.array([1.1, 1.9, 3.1, 3.9, 5.1])

reg = DecisionTreeRegressor(max_depth=3)
reg.fit(X_regression, y_regression)

test_input_reg = np.array([[1.5], [4.5]])
preds_reg = reg.predict(test_input_reg)
print(f"Inputs:\n{test_input_reg}")
# Should be approx 1.9 and 5.1
print(f"Predictions: {preds_reg}")
