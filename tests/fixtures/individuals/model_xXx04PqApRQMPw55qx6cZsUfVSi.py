
# ========== Start: GeneCrossed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# ========== End:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np

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
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)

    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        - X (array-like): The feature dataset.
        - y (array-like): The target variable.
        """
        # Apply SMOTE to handle class imbalance if present
        X_res, y_res = self.smote.fit_resample(X, y)
        
        # Scale features to have zero mean and unit variance
        X_scaled = self.scaler.fit_transform(X_res)
        
        # Define hyperparameters for tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
           'max_depth': [None, 5, 10],
           'min_samples_split': [2, 5, 10],
           'min_samples_leaf': [1, 5, 10]
        }
        
        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(estimator=self.classifier, param_grid=param_grid, cv=5)
        grid_search.fit(X_scaled, y_res)
        
        # Update the classifier with the best parameters found
        self.classifier = grid_search.best_estimator_

    def predict(self, X):
        """
        Predict the labels of the test set.

        Parameters:
        - X (array-like): The feature dataset to predict.

        Returns:
        - predictions (array-like): The predicted labels.
        """
        # Scale the test features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions using the tuned classifier
        return self.classifier.predict(X_scaled)

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

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
        self.scaler = StandardScaler()

    def _scale_features(self, X):
        """
        Scale the features using StandardScaler.

        Parameters:
        - X: The feature data.

        Returns:
        - Scaled feature data.
        """
        return self.scaler.fit_transform(X)

    def _tune_hyperparameters(self, X, y):
        """
        Perform grid search to tune hyperparameters.

        Parameters:
        - X: The feature data.
        - y: The target variable.

        Returns:
        - Best parameters found during grid search.
        """
        param_grid = {
            'n_estimators': [10, 50, 100],
           'max_depth': [None, 5, 10],
           'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(estimator=self.classifier, param_grid=param_grid, cv=5)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def _handle_imbalance(self, X, y):
        """
        Compute class weights to handle imbalanced datasets.

        Parameters:
        - X: The feature data.
        - y: The target variable.

        Returns:
        - Class weights.
        """
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        return dict(enumerate(class_weights))

    def fit(self, X, y):
        """
        Fit the classifier to the training data.

        Parameters:
        - X: The feature data.
        - y: The target variable.
        """
        X_scaled = self._scale_features(X)
        best_params = self._tune_hyperparameters(X_scaled, y)
        self.classifier.set_params(**best_params)
        class_weights = self._handle_imbalance(X_scaled, y)
        self.classifier.fit(X_scaled, y, **class_weights)

    def predict(self, X):
        """
        Predict the target variable based on the input features.

        Parameters:
        - X: The feature data.

        Returns:
        - Predicted targets.
        """
        X_scaled = self._scale_features(X)
        return self.classifier.predict(X_scaled)

    def evaluate(self, X, y):
        """
        Evaluate the model's performance.

        Parameters:
        - X: The feature data.
        - y: The target variable.

        Returns:
        - Accuracy score, classification report, and confusion matrix.
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        matrix = confusion_matrix(y, predictions)
        return accuracy, report, matrix

