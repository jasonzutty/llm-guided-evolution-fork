from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Model:
    def __init__(self):
        # Initialize the classifier with hyperparameter tuning using GridSearchCV
        self.param_grid = {
            'n_estimators': [100, 200, 300],
           'max_depth': [None, 5, 10],
           'min_samples_split': [2, 5, 10],
           'min_samples_leaf': [1, 5, 10]
        }
        self.classifier = GridSearchCV(RandomForestClassifier(), self.param_grid, cv=5)

    def fit(self, X, y):
        # Perform hyperparameter tuning during training
        self.classifier.fit(X, y)
        
    def predict(self, X):
        # Use the best-performing model from hyperparameter tuning for predictions
        return self.classifier.best_estimator_.predict(X)

    def evaluate(self, X, y):
        # Evaluate the model's performance using various metrics
        predictions = self.predict(X)
        print("Accuracy:", accuracy_score(y, predictions))
        print("Classification Report:\n", classification_report(y, predictions))
        print("Confusion Matrix:\n", confusion_matrix(y, predictions))

