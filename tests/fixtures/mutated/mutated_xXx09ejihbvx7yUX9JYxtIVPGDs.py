from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Model:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        """
        ### Initialize the Model with a RandomForestClassifier.

        Parameters:
        - **n_estimators**: The number of trees in the forest.
        - **max_depth**: The maximum depth of the tree.
        - **min_samples_split**: The minimum number of samples required to split an internal node.
        """
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def _split_data(self, X, y, test_size=0.2, random_state=42):
        """
        ### Helper Function to Split Data into Training and Testing Sets.

        Parameters:
        - **X**: The feature data.
        - **y**: The target variable.
        - **test_size**: The proportion of the dataset to include in the test split.
        - **random_state**: Controls the shuffling applied to the data before applying the split.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def fit(self, X, y):
        """
        ### Fit the Classifier to the Training Data.

        Parameters:
        - **X**: The feature data.
        - **y**: The target variable.
        """
        self._split_data(X, y)
        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, X):
        """
        ### Predict the Target Variable Based on the Input Features.

        Parameters:
        - **X**: The feature data.

        Returns:
        - **Predicted Targets**.
        """
        if X is not None:
            return self.classifier.predict(X)
        else:
            return self.classifier.predict(self.X_test)

    def evaluate(self):
        """
        ### Evaluate the Performance of the Classifier.

        Returns:
        - **Accuracy Score**.
        - **Classification Report**.
        - **Confusion Matrix**.
        """
        predictions = self.predict(None)
        accuracy = accuracy_score(self.y_test, predictions)
        report = classification_report(self.y_test, predictions)
        matrix = confusion_matrix(self.y_test, predictions)
        return accuracy, report, matrix

