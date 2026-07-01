from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from typing import Tuple

class Model:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the model with a RandomForestClassifier.

        Args:
        - n_estimators (int): The number of trees in the forest. Defaults to 100.
        - random_state (int): The seed used to shuffle the data before training. Defaults to 42.
        """
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    def _tune_hyperparameters(self, X: Tuple, y: Tuple) -> dict:
        """
        Perform grid search to find the optimal hyperparameters for the classifier.

        Args:
        - X (Tuple): The feature data.
        - y (Tuple): The target data.

        Returns:
        - dict: A dictionary containing the best parameters found.
        """
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
           'max_depth': [None, 5, 10, 15]
        }
        grid_search = GridSearchCV(estimator=self.classifier, param_grid=param_grid, cv=5)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def fit(self, X: Tuple, y: Tuple) -> None:
        """
        Train the model using the provided data.

        Args:
        - X (Tuple): The feature data.
        - y (Tuple): The target data.
        """
        try:
            # Tune hyperparameters if desired
            # best_params = self._tune_hyperparameters(X, y)
            # self.classifier.set_params(**best_params)
            self.classifier.fit(X, y)
        except Exception as e:
            print(f"An error occurred during training: {e}")

    def predict(self, X: Tuple) -> Tuple:
        """
        Make predictions on the provided data.

        Args:
        - X (Tuple): The feature data.

        Returns:
        - Tuple: The predicted targets.
        """
        try:
            return self.classifier.predict(X)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return None

