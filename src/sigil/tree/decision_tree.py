from abc import abstractmethod
from collections import Counter
from typing import Optional, Tuple, Union, cast
from sigil.models import BaseModel
import numpy as np


TreeValue = Union[int, float, str]


class Node:
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        value: Optional[TreeValue] = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self) -> bool:
        return self.value is not None


class BaseDecisionTree(BaseModel):
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features: Optional[int] = None
        self.root: Optional[Node] = None

    def build(self, X: np.ndarray, **kwargs) -> None:
        # if not initialized, set to the input feature size
        if not self.n_features:
            self.n_features = X.shape[1]
        # otherwise, if there already exists a value, take the minimum
        # so we don't use more features than available
        else:
            self.n_features = min(self.n_features, X.shape[1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.n_features is None:
            self.build(X)
        self.root = self._grow_tree(X, y)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise ValueError(
                "The model has not been trained yet. Please call fit() first."
            )
        predictions = [self._traverse_tree(x, self.root) for x in input_data]
        return np.array(predictions)

    # we implement this where we can, but trees do not use step-wise training
    def step(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError("Decision Trees do not implement step-wise training.")

    # for the individual tree methods to implement
    @abstractmethod
    def _calculate_impurity(self, y: np.ndarray) -> float:
        pass

    # for the individual tree methods to implement
    @abstractmethod
    def _calculate_leaf_value(self, y: np.ndarray) -> Union[int, float, str]:
        pass

    # training phase, this builds the structure when fitting the data
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        n_samples, n_features = X.shape
        n_unique_features = len(np.unique(y))

        # this is a recursive function so we need a base case to terminate
        # the termination here can be either
        # 1 - reaching max depth
        # 2 - 1 class left
        # 3 - too little samples (compared to the init)
        if (
            depth >= self.max_depth
            or n_unique_features == 1
            or n_samples < self.min_samples_split
        ):
            return Node(value=self._calculate_leaf_value(y))

        # otherwise, we need to find the best split (i.e. best question to ask)
        # randomly sample self.n_features from the available features
        # on first call this would be all features, but later calls (recursive) will slowly reduce the set
        features_indices = cast(  # random.choice can give a number or an array, in this case it's always an array
            np.ndarray, np.random.choice(n_features, self.n_features, replace=False)
        )
        # given the features, and the dataset, find the best "question" to ask
        best_feature_index, best_threshold = self._find_best_split(
            X, y, features_indices
        )

        # not able to find a split that helps, so there is nothing better to do here
        if best_feature_index is None or best_threshold is None:
            return Node(value=self._calculate_leaf_value(y))

        # otherwise, we have found a question that helps split the data, so now we split it
        left_indices, right_indices = self._split_dataset(
            X[:, best_feature_index], best_threshold
        )

        # then we recursively build the left and right subtrees
        left_subtree = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(
            X[right_indices, :], y[right_indices], depth + 1
        )

        return Node(
            feature=best_feature_index,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _calculate_information_gain(
        self, y: np.ndarray, X_column: np.ndarray, threshold: float
    ) -> float:
        # calculate impurity of the parent (before splitting)
        parent_impurity = self._calculate_impurity(y)

        # simulate split
        left_indices, right_indices = self._split_dataset(X_column, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0.0

        # calculate the weighted average impurity of children
        n_total_samples = len(y)
        n_left, n_right = len(left_indices), len(right_indices)

        impurity_left = self._calculate_impurity(y[left_indices])
        impurity_right = self._calculate_impurity(y[right_indices])

        weighted_avg_child_impurity = (n_left / n_total_samples) * impurity_left + (
            n_right / n_total_samples
        ) * impurity_right

        # what we started with - what we ended up with
        information_gain = parent_impurity - weighted_avg_child_impurity
        return information_gain

    def _split_dataset(
        self, X_column: np.ndarray, threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_indices = np.argwhere(X_column <= threshold).flatten()
        right_indices = np.argwhere(X_column > threshold).flatten()
        return left_indices, right_indices

    # this tries to find the best question that splits the dataset
    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray, features_indices: np.ndarray
    ) -> Tuple[Optional[int], Optional[float]]:
        # keep track of the feature that gives us best gain and which index/threshold it is
        best_gain = -1.0
        split_index, split_threshold = None, None

        # iterate through features and thresholds to find the best split
        for feature_index in features_indices:
            X_column = X[:, feature_index]
            possible_thresholds = np.unique(X_column)
            for threshold in possible_thresholds:
                # calculate how good the split would be
                gain = self._calculate_information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

    def _traverse_tree(self, x: np.ndarray, node: Node) -> TreeValue:
        if node.is_leaf_node():
            if node.value is None:
                raise ValueError("Leaf node has no value.")
            return node.value

        if node.feature is not None and node.threshold is not None:
            if x[node.feature] <= node.threshold:
                if node.left:
                    return self._traverse_tree(x, node.left)
            else:
                if node.right:
                    return self._traverse_tree(x, node.right)
        return node.value if node.value is not None else 0


class DecisionTreeClassifier(BaseDecisionTree):
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        gini impurity
        gini = 1 - sum(probabability of each class squared)
        """
        labels, n_samples = np.unique(y), len(y)
        impurity = 1.0
        for label in labels:
            prob = len(y[y == label]) / n_samples
            impurity -= prob**2
        return impurity

    def _calculate_leaf_value(self, y: np.ndarray) -> TreeValue:
        """
        for a classifier, the leaf value is the majority class
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]


class DecisionTreeRegressor(BaseDecisionTree):
    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        variance impurity
        want to minimize variance in the children nodes
        """
        if len(y) == 0:
            return 0.0
        return float(np.var(y))

    def _calculate_leaf_value(self, y: np.ndarray) -> TreeValue:
        """
        for a regressor, the leaf value is the average of the targets
        """
        return float(np.mean(y))
