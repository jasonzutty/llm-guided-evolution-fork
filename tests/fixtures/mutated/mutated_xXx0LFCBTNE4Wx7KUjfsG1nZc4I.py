from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Any

class Model:
    """
    A simple wrapper around RandomForestClassifier.
    
    Attributes:
        classifier (RandomForestClassifier): The underlying classifier.
    """

    def __init__(self) -> None:
        """
        Initializes the Model with a new RandomForestClassifier.
        """
        self.classifier = RandomForestClassifier()

    def _check_input_data(self, X: Any, y: Any) -> None:
        """
        Checks if the input data is valid.
        
        Args:
            X (Any): The feature data.
            y (Any): The target data.
        
        Raises:
            ValueError: If the input data is invalid.
        """
        if not isinstance(X, (list, tuple, dict)) or not isinstance(y, (list, tuple)):
            raise ValueError("Invalid input data")

    def fit(self, X: Any, y: Any) -> None:
        """
        Fits the model to the training data.
        
        Args:
            X (Any): The feature data.
            y (Any): The target data.
        """
        try:
            self._check_input_data(X, y)
            self.classifier.fit(X, y)
        except Exception as e:
            print(f"An error occurred during fitting: {e}")

    def predict(self, X: Any) -> Any:
        """
        Makes predictions on the given data.
        
        Args:
            X (Any): The data to predict.
        
        Returns:
            Any: The predicted values.
        """
        try:
            return self.classifier.predict(X)
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            return None

    @staticmethod
    def split_data(X: Any, y: Any, test_size: float = 0.2, random_state: int = 42) -> Tuple[Any, Any, Any, Any]:
        """
        Splits the data into training and testing sets.
        
        Args:
            X (Any): The feature data.
            y (Any): The target data.
            test_size (float, optional): The proportion of the data to include in the test set. Defaults to 0.2.
            random_state (int, optional): The seed used to shuffle the data before splitting. Defaults to 42.
        
        Returns:
            Tuple[Any, Any, Any, Any]: The training and testing data.
        """
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

