from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from typing import Tuple, Dict

class Model:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the model with a RandomForestClassifier.

        Args:
        - n_estimators (int): The number of trees in the forest. Defaults to 100.
        - random_state (int): The seed used to shuffle the data before training. Defaults to 42.
        """
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Perform grid search to find the optimal hyperparameters for the classifier.

        Args:
        - X (np.ndarray): The feature data.
        - y (np.ndarray): The target data.

        Returns:
        - Dict: A dictionary containing the best parameters found.
        """
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
           'max_depth': [None, 5, 10, 15]
        }
        grid_search = GridSearchCV(estimator=self.classifier, param_grid=param_grid, cv=5)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def _handle_error(self, error_message: str, exception: Exception) -> None:
        """
        Handle an error by printing an error message and logging the exception.

        Args:
        - error_message (str): The error message to print.
        - exception (Exception): The exception that occurred.
        """
        print(error_message)
        raise exception

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate the inputs to the fit and predict methods.

        Args:
        - X (np.ndarray): The feature data.
        - y (np.ndarray): The target data.
        """
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            self._handle_error("Invalid input type for X", TypeError("X must be a numpy array or pandas DataFrame"))
        if not isinstance(y, (np.ndarray, pd.Series)):
            self._handle_error("Invalid input type for y", TypeError("y must be a numpy array or pandas Series"))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using the provided data.

        Args:
        - X (np.ndarray): The feature data.
        - y (np.ndarray): The target data.
        """
        try:
            self._validate_inputs(X, y)
