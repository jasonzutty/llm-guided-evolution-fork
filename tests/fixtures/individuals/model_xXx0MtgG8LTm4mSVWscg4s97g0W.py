from sklearn.ensemble import RandomForestClassifier
import numpy as np

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

