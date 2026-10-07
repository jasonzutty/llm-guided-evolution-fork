
# ========== Start: GeneCrossed
# ========== Start: GeneCrossed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# ========== End:

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self, use_ensemble=False):
        """
        Initialize the model.
        
        :param use_ensemble: Whether to use an ensemble of classifiers.
        """
        if use_ensemble:
            # Using BaggingClassifier as a simple form of ensemble
            self.classifier = BaggingClassifier(
                base_estimator=RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42
                ),
                random_state=42
            )
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42
            )
        self.scaler = StandardScaler()

    def _scale_data(self, X):
        """
        Scale the data using StandardScaler.
        
        :param X: Data to be scaled.
        :return: Scaled data.
        """
        return self.scaler.fit_transform(X)

    def _tune_hyperparameters(self, X, y):
        """
        Simplified hyperparameter tuning using GridSearchCV.
        
        :param X: Training data.
        :param y: Target values.
        """
        param_grid = {
            'base_estimator__max_depth': [None, 5],
            'base_estimator__n_estimators': [10, 50, 100]
        } if isinstance(self.classifier, BaggingClassifier) else {
          'max_depth': [None, 5],
           'n_estimators': [10, 50, 100]
        }
        grid_search = GridSearchCV(estimator=self.classifier, param_grid=param_grid, cv=3)
        grid_search.fit(X, y)
        self.classifier = grid_search.best_estimator_

    def fit(self, X, y):
        """
        Fit the model.
        
        :param X: Training data.
        :param y: Target values.
        """
        X_scaled = self._scale_data(X)
        self._tune_hyperparameters(X_scaled, y)
        self.classifier.fit(X_scaled, y)

    def predict(self, X):
        """
        Make predictions.
        
        :param X: Data to make predictions on.
        :return: Predictions.
        """
        X_scaled = self._scale_data(X)
        return self.classifier.predict(X_scaled)
# ========== End:

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class Model:
    def __init__(self, use_ensemble=False):
        """
        Initialize the model.
        
        :param use_ensemble: Whether to use an ensemble of classifiers.
        """
        if use_ensemble:
            # Using BaggingClassifier as a simple form of ensemble
            classifier = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=10))
        else:
            classifier = RandomForestClassifier(n_estimators=10)
        
        # Define pipeline with StandardScaler, PCA, and classifier
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA()),
            ('classifier', classifier)
        ])
        
        # Initialize hyperparameter tuning spaces
        self.param_grid = {
            'pca__n_components': [0.95, 0.99],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10]
        }
        
        # For ensemble case, adjust param_grid accordingly
        if use_ensemble:
            self.param_grid = {
                'pca__n_components': [0.95, 0.99],
                'classifier__base_estimator__n_estimators': [100, 200, 300],
                'classifier__base_estimator__max_depth': [None, 5, 10]
            }
        
        # Initialize refined hyperparameter tuning space
        self.refined_param_grid = None

    ### Helper Function to Perform Hyperparameter Tuning
    def _perform_hyperparameter_tuning(self, X, y, param_grid):
        """Perform randomized search for hyperparameter tuning."""
        randomized_search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_grid,
            cv=5,
            n_iter=10,
            random_state=42
        )
        
        # Fit the model using randomized search
        randomized_search.fit(X, y)
        
        return randomized_search
    
    ### Helper Function to Refine Hyperparameters
    def _refine_hyperparameters(self, X, y, best_params):
        """Refine hyperparameters using grid search."""
        # Create refined hyperparameter tuning space
        if hasattr(self.pipeline.named_steps['classifier'], 'base_estimator'):
            self.refined_param_grid = {
                'pca__n_components': [best_params['pca__n_components'] - 0.01, best_params['pca__n_components'], best_params['pca__n_components'] + 0.01],
                'classifier__base_estimator__n_estimators': [best_params['classifier__base_estimator__n_estimators'] - 50, best_params['classifier__base_estimator__n_estimators'], best_params['classifier__base_estimator__n_estimators'] + 50],
                'classifier__base_estimator__max_depth': [best_params['classifier__base_estimator__max_depth'] - 2, best_params['classifier__base_estimator__max_depth'], best_params['classifier__base_estimator__max_depth'] + 2] if best_params['classifier__base_estimator__max_depth'] is not None else [None, 5, 10]
            }
        else:
            self.refined_param_grid = {
                'pca__n_components': [best_params['pca__n_components'] - 0.01, best_params['pca__n_components'], best_params['pca__n_components'] + 0
# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import numpy as np

class Model:
    def __init__(self):
        # Define pipeline with PCA and RandomForestClassifier
        self.pipeline = Pipeline([
            ('pca', PCA()),
            ('classifier', RandomForestClassifier())
        ])
        
        # Initialize hyperparameter tuning spaces
        self.param_grid = {
            'pca__n_components': [0.95, 0.99],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10]
        }
        
        # Initialize refined hyperparameter tuning space
        self.refined_param_grid = None

    ### Helper Function to Perform Hyperparameter Tuning
    def _perform_hyperparameter_tuning(self, X, y, param_grid):
        """Perform randomized search for hyperparameter tuning."""
        randomized_search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_grid,
            cv=5,
            n_iter=10,
            random_state=42
        )
        
        # Fit the model using randomized search
        randomized_search.fit(X, y)
        
        return randomized_search
    
    ### Helper Function to Refine Hyperparameters
    def _refine_hyperparameters(self, X, y, best_params):
        """Refine hyperparameters using grid search."""
        # Create refined hyperparameter tuning space
        self.refined_param_grid = {
            'pca__n_components': [best_params['pca__n_components'] - 0.01, best_params['pca__n_components'], best_params['pca__n_components'] + 0.01],
            'classifier__n_estimators': [best_params['classifier__n_estimators'] - 50, best_params['classifier__n_estimators'], best_params['classifier__n_estimators'] + 50],
            'classifier__max_depth': [best_params['classifier__max_depth'] - 2, best_params['classifier__max_depth'], best_params['classifier__max_depth'] + 2] if best_params['classifier__max_depth'] is not None else [None, 5, 10]
        }
        
        # Perform grid search for refined hyperparameters
        grid_search = GridSearchCV(
            estimator=self.pipeline,
            param_grid=self.refined_param_grid,
            cv=5,
            scoring='accuracy'
        )
        
        # Fit the model using grid search
        grid_search.fit(X, y)
        
        return grid_search
    
    def fit(self, X, y):
        # Perform initial hyperparameter tuning
        randomized_search = self._perform_hyperparameter_tuning(X, y, self.param_grid)
        
        # Get best parameters from randomized search
        best_params = randomized_search.best_params_
        
        # Refine hyperparameters using grid search
        grid_search = self._refine_hyperparameters(X, y, best_params)
        
        # Update the pipeline with the best parameters
        self.pipeline.set_params(**grid_search.best_params_)
        self.pipeline.fit(X, y)

    def predict(self, X):
        # Make predictions using the optimized pipeline
        return self.pipeline.predict(X)

