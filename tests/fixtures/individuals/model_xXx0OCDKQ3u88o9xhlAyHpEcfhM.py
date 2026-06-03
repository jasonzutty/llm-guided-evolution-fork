from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self):
        # Initialize the classifier with optimized parameters
        self.classifier = RandomForestClassifier(
            n_estimators=5,  # Further reduced to minimize parameters
            max_depth=3,     # Decreased to prevent overfitting
            min_samples_split=10,  # Increased for better generalization
            min_samples_leaf=5,    # Adjusted upwards for stability
            bootstrap=False       # Disable bootstrapping to reduce variance
        )

    def fit(self, X, y):
        # Fit the classifier to the training data
        self.classifier.fit(X, y)

    def predict(self, X):
        # Make predictions on new data
        return self.classifier.predict(X)

