from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        """
        Initialize the Model with a RandomForestClassifier.

        Parameters:
        - n_estimators: The number of trees in the forest.
        - max_depth: The maximum depth of the tree.
        - min_samples_split: The minimum number of samples required to split an internal node.
        """
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )

    def fit(self, X, y):
        """
        Fit the classifier to the training data.

        Parameters:
        - X: The feature data.
        - y: The target variable.
        """
        self.classifier.fit(X, y)

    def predict(self, X):
        """
        Predict the target variable based on the input features.

        Parameters:
        - X: The feature data.

        Returns:
        - Predicted targets.
        """
        return self.classifier.predict(X)

