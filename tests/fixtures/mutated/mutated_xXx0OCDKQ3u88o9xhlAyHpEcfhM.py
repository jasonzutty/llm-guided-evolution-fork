from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Model:
    def __init__(self, 
                 n_estimators=5,  
                 max_depth=3,     
                 min_samples_split=10,  
                 min_samples_leaf=5,    
                 bootstrap=False,
                 test_size=0.2,
                 random_state=42):
        """
        Initializes the Model class.

        Args:
        - n_estimators (int): The number of trees in the forest.
        - max_depth (int): The maximum depth of each tree.
        - min_samples_split (int): The minimum number of samples required to split an internal node.
        - min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        - bootstrap (bool): Whether to use bootstrap sampling when constructing trees.
        - test_size (float): The proportion of the dataset to include in the test split.
        - random_state (int): The seed used to shuffle the dataset before splitting it into training and test sets.
        """

        # Initialize the classifier with optimized parameters
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,  
            max_depth=max_depth,     
            min_samples_split=min_samples_split,  
            min_samples_leaf=min_samples_leaf,    
            bootstrap=bootstrap       
        )
        
        # Store other parameters for potential future use
        self.test_size = test_size
        self.random_state = random_state

    def _split_data(self, X, y):
        """
        Splits the data into training and test sets.

        Args:
        - X (array-like): The feature data.
        - y (array-like): The target data.

        Returns:
        - tuple: A tuple containing the training features, test features, training targets, and test targets.
        """
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def fit(self, X, y):
        """
        Fits the classifier to the training data.

        Args:
        - X (array-like): The feature data.
        - y (array-like): The target data.
        """
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Fit the classifier to the training data
        self.classifier.fit(X_train, y_train)

        # Evaluate the model's performance on the test set
        y_pred = self.classifier.predict(X_test)
        print("Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

    def predict(self, X):
        """
        Makes predictions on new data.

        Args:
        - X (array-like): The feature data.

        Returns:
        - array-like: The predicted targets.
        """
        return self.classifier.predict(X)

