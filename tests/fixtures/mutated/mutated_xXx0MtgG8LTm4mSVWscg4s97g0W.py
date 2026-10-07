from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
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

    def _split_data(self, X, y):
        """
        Split the data into training and validation sets.
        
        Args:
            X (numpy array): Input features.
            y (numpy array): Target labels.
        
        Returns:
            tuple: Training features, validation features, training labels, validation labels.
        """
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def _scale_data(self, X):
        """
        Scale the input data using standard scaling.
        
        Args:
            X (numpy array): Input features.
        
        Returns:
            numpy array: Scaled input features.
        """
        return self.scaler.fit_transform(X)

    def _tune_hyperparameters(self, X_train, y_train):
        """
        Tune the hyperparameters of the random forest classifier.
        
        Args:
            X_train (numpy array): Training features.
            y_train (numpy array): Training labels.
        
        Returns:
            dict: Best hyperparameters.
        """
        param_grid = {
            'n_estimators': [100, 200, 300],
           'max_depth': [None, 5, 10]
        }
        grid_search = GridSearchCV(estimator=self.classifier, param_grid=param_grid, cv=3)
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_

    def _train_model(self, X_train, y_train, best_params):
        """
        Train the model with the best hyperparameters.
        
        Args:
            X_train (numpy array): Training features.
            y_train (numpy array): Training labels.
            best_params (dict): Best hyperparameters.
        """
        self.classifier.set_params(**best_params)
        self.classifier.fit(X_train, y_train)

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
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        # Scale the data
        X_train_scaled = self._scale_data(X_train)
