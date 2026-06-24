
# ========== Start: GeneCrossed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# ========== End:

from sklearn.ensemble import RandomForestClassifier

class Model:
    """
    A simple wrapper around RandomForestClassifier.
    
    Attributes:
        classifier (RandomForestClassifier): The underlying classifier.
    """

    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        """
        Initializes the Model with a new RandomForestClassifier.
        
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

    def fit(self, X, y):
        """
        Fits the model to the training data.
        
        Parameters:
            X (array-like): The feature data.
            y (array-like): The target data.
        """
        self.classifier.fit(X, y)

    def predict(self, X):
        """
        Makes predictions on the given data.
        
        Parameters:
            X (array-like): The data to predict.
        
        Returns:
            array-like: The predicted values.
        """
        return self.classifier.predict(X)

# --OPTION--
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier

class Model:
    """
    A simple wrapper around RandomForestClassifier.
    
    Attributes:
        classifier (RandomForestClassifier): The underlying classifier.
    """

    def __init__(self):
        """
        Initializes the Model with a new RandomForestClassifier.
        """
        self.classifier = RandomForestClassifier()

    def fit(self, X, y):
        """
        Fits the model to the training data.
        
        Parameters:
            X (array-like): The feature data.
            y (array-like): The target data.
        """
        self.classifier.fit(X, y)

    def predict(self, X):
        """
        Makes predictions on the given data.
        
        Parameters:
            X (array-like): The data to predict.
        
        Returns:
            array-like: The predicted values.
        """
        return self.classifier.predict(X)

# Example usage
if __name__ == "__main__":
    # Assuming some data
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    model = Model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(predictions)

