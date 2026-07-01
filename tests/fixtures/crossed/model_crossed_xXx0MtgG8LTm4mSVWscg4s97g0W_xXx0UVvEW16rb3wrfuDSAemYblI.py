
# ========== Start: GeneCrossed
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
# ========== End:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

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
        report = classification_report(y, y_pred)
        matrix = confusion_matrix(y, y_pred)
        return accuracy, report, matrix

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np

class Model:
    def __init__(self):
        """
        Initialize the model with a random forest classifier and a standard scaler.
        """
        self.classifier = RandomForestClassifier()
        self.scaler = StandardScaler()

    def _preprocess_data(self, X):
        """
        Preprocess the input data using standard scaling.
        
        Args:
            X (numpy array): Input features.
        
        Returns:
            numpy array: Scaled input features.
        """
        return self.scaler.fit_transform(X)

    def _evaluate_model(self, X_test, y_test):
        """
        Evaluate the model's performance on the test set.
        
        Args:
            X_test (numpy array): Test features.
            y_test (numpy array): Test labels.
        
        Returns:
            float: Model accuracy.
        """
        y_pred = self.classifier.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def fit(self, X, y):
        """
        Train the model on the given data.
        
        Args:
            X (numpy array): Input features.
            y (numpy array): Target labels.
        """
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Preprocess training data
        X_train_scaled = self._preprocess_data(X_train)
        
        # Train the model
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate the model on the validation set
        X_val_scaled = self._preprocess_data(X_val)
        accuracy = self._evaluate_model(X_val_scaled, y_val)
        print(f"Model accuracy on validation set: {accuracy:.3f}")

    def predict(self, X):
        """
        Make predictions on the given input data.
        
        Args:
            X (numpy array): Input features.
        
        Returns:
            numpy array: Predicted labels.
        """
        # Preprocess input data
        X_scaled = self._preprocess_data(X)
        
        # Make predictions
        return self.classifier.predict(X_scaled)

