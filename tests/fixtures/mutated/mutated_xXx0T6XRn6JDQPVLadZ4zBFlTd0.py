from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Model:
    """
    A class representing a machine learning model with hyperparameter tuning.
    
    Attributes:
    param_grid (dict): Dictionary containing hyperparameters and their possible values.
    classifier (GridSearchCV): An instance of GridSearchCV for hyperparameter tuning.
    """

    def __init__(self):
        """
        Initializes the classifier with hyperparameter tuning using GridSearchCV.
        """
        # Define the hyperparameter grid
        self.param_grid = {
            'n_estimators': [100, 200, 300],
           'max_depth': [None, 5, 10],
           'min_samples_split': [2, 5, 10],
           'min_samples_leaf': [1, 5, 10]
        }
        # Initialize the classifier with hyperparameter tuning
        self.classifier = GridSearchCV(RandomForestClassifier(), self.param_grid, cv=5)

    def _validate_input(self, X, y):
        """
        Validates the input data.
        
        Args:
        X (array-like): Feature data.
        y (array-like): Target data.
        
        Raises:
        TypeError: If X or y is not array-like.
        ValueError: If X and y have different lengths.
        """
        if not hasattr(X, '__len__') or not hasattr(y, '__len__'):
            raise TypeError("Both X and y must be array-like.")
        if len(X)!= len(y):
            raise ValueError("X and y must have the same length.")

    def fit(self, X, y):
        """
        Performs hyperparameter tuning during training.
        
        Args:
        X (array-like): Feature data.
        y (array-like): Target data.
        
        Raises:
        Exception: Any exception that occurs during fitting.
        """
        try:
            # Validate the input data
            self._validate_input(X, y)
            # Perform hyperparameter tuning during training
            self.classifier.fit(X, y)
        except Exception as e:
            raise Exception(f"An error occurred during fitting: {e}")

    def _make_prediction(self, X):
        """
        Uses the best-performing model from hyperparameter tuning for predictions.
        
        Args:
        X (array-like): Feature data.
        
        Returns:
        array-like: Predictions made by the best estimator.
        """
        return self.classifier.best_estimator_.predict(X)

    def predict(self, X):
        """
        Makes predictions on the given feature data.
        
        Args:
        X (array-like): Feature data.
        
        Returns:
        array-like: Predictions made by the model.
        
        Raises:
        Exception: Any exception that occurs during prediction.
        """
        try:
            # Make predictions using the best estimator
            return self._make_prediction(X)
        except Exception as e:
            raise Exception(f"An error occurred during prediction: {e}")

    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculates various metrics such as accuracy, classification report, and confusion matrix.
        
        Args:
        y_true (array-like): True target values.
