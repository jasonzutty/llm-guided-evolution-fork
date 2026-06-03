
# ========== Start: GeneCrossed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# ========== End:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

class Model:
    def __init__(self):
        # Initialize a StandardScaler for feature scaling
        self.scaler = StandardScaler()
        
        # Initialize a RandomForestClassifier with default parameters
        self.classifier = RandomForestClassifier(n_jobs=-1)
        
        # Define a parameter grid for hyperparameter tuning
        self.param_grid = {
            'n_estimators': [100, 200, 300],
           'max_depth': [None, 5, 10]
        }

    def fit(self, X, y):
        # Scale the features using StandardScaler
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform hyperparameter tuning using GridSearchCV
        grid_search = GridSearchCV(estimator=self.classifier, param_grid=self.param_grid, cv=5)
        grid_search.fit(X_scaled, y)
        
        # Update the classifier with the best-performing hyperparameters
        self.classifier = grid_search.best_estimator_

    def predict(self, X):
        # Scale the features using StandardScaler
        X_scaled = self.scaler.transform(X)
        
        # Make predictions using the trained classifier
        return self.classifier.predict(X_scaled)

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class Model:
    def __init__(self):
        # Define hyperparameter tuning space for RandomForestClassifier
        self.param_grid = {
            'n_estimators': [100, 200, 300],
           'max_depth': [None, 5, 10],
           'min_samples_split': [2, 5, 10],
           'min_samples_leaf': [1, 5, 10]
        }

        # Initialize a pipeline with feature selection and RandomForestClassifier
        self.pipeline = Pipeline([
            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=100))),
            ('selector', SelectKBest(mutual_info_classif, k=10)),
            ('classifier', GridSearchCV(RandomForestClassifier(), self.param_grid, cv=5))
        ])

    def _random_search(self, X, y):
        """Perform random search for hyperparameters."""
        randomized_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(),
            param_distributions=self.param_grid,
            n_iter=10,
            cv=5
        )
        randomized_search.fit(X, y)
        return randomized_search.best_params_

    def _early_stopping(self, X, y):
        """Implement early stopping using a validation set."""
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        best_accuracy = 0
        best_model = None
        for params in self._grid_search_generator():
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        return best_model

    def _grid_search_generator(self):
        """Generate all possible combinations of hyperparameters."""
        import itertools
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def fit(self, X, y):
        # Perform random search for hyperparameters
        best_params = self._random_search(X, y)

        # Update the pipeline with the best hyperparameters
        self.pipeline.set_params(classifier__param_grid=[best_params])

        # Implement early stopping
        # best_model = self._early_stopping(X, y)

        # Train the pipeline on the data
        self.pipeline.fit(X, y)

    def predict(self, X):
        # Use the trained pipeline to make predictions
        return self.pipeline.predict(X)

    def evaluate(self, X, y):
        """Evaluate the performance of the model."""
        y_pred = self.predict(X)
        print("Accuracy:", accuracy_score(y, y_pred))
        print("Classification Report:")
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

