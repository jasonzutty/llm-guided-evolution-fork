
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# ========== End:

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

class Model(BaseEstimator, ClassifierMixin):
    def __init__(self):
        """
        Initialize the model with four base classifiers and a meta-classifier.
        """
        self.random_forest = RandomForestClassifier()
        self.gradient_boosting = GradientBoostingClassifier()
        self.support_vector_machine = SVC(probability=True)
        self.logistic_regression = LogisticRegression(max_iter=1000)
        self.meta_voting = VotingClassifier(estimators=[
            ('random_forest', self.random_forest), 
            ('gradient_boosting', self.gradient_boosting),
            ('support_vector_machine', self.support_vector_machine),
            ('logistic_regression', self.logistic_regression)], voting='soft')

    def fit(self, X, y):
        """
        Fit all base classifiers and then the meta-classifier.
        
        Parameters:
        - X: Training features
        - y: Target variable
        """
        # Train the base classifiers
        self.random_forest.fit(X, y)
        self.gradient_boosting.fit(X, y)
        self.support_vector_machine.fit(X, y)
        self.logistic_regression.fit(X, y)
        
        # Train the meta-classifier
        self.meta_voting.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        - X: Features to predict
        
        Returns:
        - Predictions made by the model
        """
        # Use the meta-classifier to make the final predictions
        return self.meta_voting.predict(X)
# ========== End:

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class Model:
    def __init__(self):
        """
        Initialize the model with three base classifiers and a meta-classifier.
        """
        self.random_forest = RandomForestClassifier()
        self.gradient_boosting = GradientBoostingClassifier()
        self.support_vector_machine = SVC(probability=True)
        self.meta_voting = VotingClassifier(estimators=[
            ('random_forest', self.random_forest), 
            ('gradient_boosting', self.gradient_boosting),
            ('support_vector_machine', self.support_vector_machine)], voting='soft')
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
        Perform grid search to tune hyperparameters for each base classifier.

        Parameters:
        - X: The feature data.
        - y: The target variable.

        Returns:
        - Best parameters found during grid search for each classifier.
        """
        param_grids = {
            'random_forest': {
                'n_estimators': [10, 50, 100],
               'max_depth': [None, 5, 10],
               'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [10, 50, 100],
                'learning_rate': [0.1, 0.05, 0.01],
               'max_depth': [3, 5, 10]
            },
           'support_vector_machine': {
                'C': [1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly']
            }
        }
        
        best_params = {}
        for name, classifier in [('random_forest', self.random_forest), 
                                 ('gradient_boosting', self.gradient_boosting),
                                 ('support_vector_machine', self.support_vector_machine)]:
            grid_search = GridSearchCV(estimator=classifier, param_grid=param_grids[name], cv=5)
            grid_search.fit(X, y)
            best_params[name] = grid_search.best_params_
            
        return best_params

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
        Fit all base classifiers and then the
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

