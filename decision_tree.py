import numpy as np


class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, labels):
        """
        Calculate the Gini coefficient
        """
        classes, counts = np.unique(labels, return_counts=True)
        probabilities = counts / len(labels)
        return 1 - np.sum(probabilities**2)

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split the data using Gini impurity.
        """
        n_samples, n_features = X.shape
        best_gini = float("inf")
        best_split = None

        for feature_index in range(n_features):
            # Sort the data by this feature for efficient threshold testing
            sorted_indices = np.argsort(X[:, feature_index])
            X_sorted, y_sorted = X[sorted_indices], y[sorted_indices]

            for i in range(1, n_samples):
                if X_sorted[i, feature_index] == X_sorted[i - 1, feature_index]:
                    continue

                threshold = (
                    X_sorted[i, feature_index] + X_sorted[i - 1, feature_index]
                ) / 2
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                y_left, y_right = y[left_mask], y[right_mask]

                # Skip invalid splits
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Compute Gini impurity for the split
                gini_left = self._gini(y_left)
                gini_right = self._gini(y_right)
                gini_split = (len(y_left) / n_samples) * gini_left + (
                    len(y_right) / n_samples
                ) * gini_right

                if gini_split < best_gini:
                    best_gini = gini_split
                    best_split = (feature_index, threshold)

        return best_split  # This should be None if no valid split is found

    def _split(self, X, y, feature_index, threshold):
        """
        Split the dataset into left and right branches based on the feature and threshold.
        """
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _build_tree(self, train_features, labels, depth):
        """
        Recursively build the decision tree.
        """
        # Stopping criteria
        if depth >= self.max_depth or len(np.unique(labels)) == 1 or len(labels) <= 1:
            return int(
                np.bincount(labels).argmax()
            )  # Ensure a plain integer is returned

        # Find the best split
        split = self._best_split(train_features, labels)
        if not split:  # If split == None then return the majority class
            return int(np.bincount(labels).argmax())

        feature_index, threshold = split
        X_left, X_right, y_left, y_right = self._split(
            train_features, labels, feature_index, threshold
        )

        # Handle cases where the split fails: one side is empty
        if len(y_left) == 0 or len(y_right) == 0:
            return int(np.bincount(labels).argmax())

        # Recursively build left and right subtrees
        left_tree = self._build_tree(X_left, y_left, depth + 1)
        right_tree = self._build_tree(X_right, y_right, depth + 1)

        # Return the node as a tuple: (feature_index, threshold, left_tree, right_tree)
        return (feature_index, threshold, left_tree, right_tree)

    def fit(self, train_features, labels):
        """
        Train the decision tree on the dataset.
        """
        self.tree = self._build_tree(train_features, labels, depth=0)

    def _predict_single(self, x, tree):
        """
        Predict the class for a single sample using the decision tree.
        """
        # Check if the node is a leaf node (integer)
        if isinstance(tree, int):
            return tree

        feature_index, threshold, left_tree, right_tree = tree
        if x[feature_index] <= threshold:
            return self._predict_single(x, left_tree)
        else:
            return self._predict_single(x, right_tree)

    def predict(self, test_features):
        """
        Predict the class labels for the input data.
        """
        return np.array(
            [self._predict_single(feature, self.tree) for feature in test_features]
        )
