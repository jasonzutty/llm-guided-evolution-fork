
# ========== Start: GeneCrossed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# ========== End:

import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

class Model:
    def __init__(self, use_ensemble=False):
        """
        Initialize the Model class.
        
        Attributes:
            classifier (RandomForestClassifier or BaggingClassifier): The classifier.
            pipeline (Pipeline): The pipeline containing the scaler, pca, and classifier.
        """
        if use_ensemble:
            self.classifier = BaggingClassifier(base_estimator=RandomForestClassifier())
        else:
            self.classifier = RandomForestClassifier()
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('classifier', self.classifier)
        ])

    def _hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
            X_train (array-like): The training features.
            y_train (array-like): The training target variable.
        
        Returns:
            dict: The best parameters found by GridSearchCV.
        """
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10]
        }
        if isinstance(self.classifier, BaggingClassifier):
            param_grid['classifier__base_estimator__n_estimators'] = [10, 50, 100]
            param_grid['classifier__base_estimator__max_depth'] = [None, 5, 10]
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_

    def fit(self, X, y):
        """
        Fit the model to the training data.
        
        Parameters:
            X (array-like): The feature data.
            y (array-like): The target variable.
        """
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        best_params = self._hyperparameter_tuning(X_train, y_train)
        self.pipeline.set_params(**best_params)
        self.pipeline.fit(X, y)

    def predict(self, X):
        """
        Make predictions on the given data.
        
        Parameters:
            X (array-like): The feature data.
        
        Returns:
            array-like: The predicted classes.
        """
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        """
        Evaluate the model's performance on the given data.
        
        Parameters:
            X (array-like): The feature data.
            y (array-like): The target variable.
        
        Returns:
            tuple: A tuple containing the accuracy score, classification report, and confusion matrix.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification
# --OPTION--
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

class Model:
    def __init__(self):
        """
        Initialize the Model class.
        
        Attributes:
            classifier (RandomForestClassifier): The random forest classifier.
            pipeline (Pipeline): The pipeline containing the scaler, pca, and classifier.
        """
        self.classifier = RandomForestClassifier()
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)),
            ('classifier', self.classifier)
        ])

    def _split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets.
        
        Parameters:
            X (array-like): The feature data.
            y (array-like): The target variable.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): The seed used to shuffle the data before splitting. Defaults to 42.
        
        Returns:
            tuple: A tuple containing the training features, testing features, training target variable, and testing target variable.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def _hyperparameter_tuning(self, X_train, y_train):
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
            X_train (array-like): The training features.
            y_train (array-like): The training target variable.
        
        Returns:
            dict: The best parameters found by GridSearchCV.
        """
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_

    def _fit_pipeline(self, X, y, params):
        """
        Fit the pipeline with the specified parameters.
        
        Parameters:
            X (array-like): The feature data.
            y (
