
# ========== Start: GeneCrossed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# ========== End:

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

class Model:
    def __init__(self):
        """
        Initialize the Model class with a RandomForestClassifier.
        
        Attributes:
            classifier (RandomForestClassifier): The classifier used for predictions.
            best_params_ (dict): Stores the optimal parameters found during grid search.
        """
        self.classifier = RandomForestClassifier()
        self.best_params_ = None
    
    def _create_pipeline(self, X):
        """
        Create a pipeline with preprocessing steps and the classifier.
        
        Parameters:
            X (array-like): Features of the data.
        
        Returns:
            Pipeline: A pipeline containing preprocessing steps and the classifier.
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features)])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', self.classifier)])
        
        return pipeline
    
    def fit(self, X, y):
        """
        Fit the model to the training data using randomized search for hyperparameter tuning.
        
        Parameters:
            X (array-like): Features of the training data.
            y (array-like): Target variable of the training data.
        """
        # Create a pipeline with preprocessing steps and the classifier
        pipeline = self._create_pipeline(X)
        
        # Define hyperparameters for randomized search
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 5, 10]
        }
        
        # Perform randomized search to find optimal hyperparameters
        random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, cv=5, n_iter=10)
        random_search.fit(X, y)
        
        # Update the classifier with the best parameters
        self.classifier = random_search.best_estimator_.named_steps['classifier']
        self.best_params_ = random_search.best_params_
    
    def predict(self, X):
        """
        Make predictions on new data using the trained model.
        
        Parameters:
            X (array-like): Features of the data to make predictions on.
        
        Returns:
            array-like: Predicted classes.
        """
        return self.classifier.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model's performance on test data.
        
        Parameters:
            X (array-like): Features of the test data.
            y (array-like): Actual target variable of the test data.
        
        Returns:
            tuple: Accuracy score, classification report, and confusion matrix.
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        matrix = confusion_matrix(y, y_pred)
        return accuracy, report, matrix

# --OPTION--
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

class Model:
    def __init__(self):
        """
        Initialize the Model class with a RandomForestClassifier.
        
        Attributes:
            classifier (RandomForestClassifier): The classifier used for predictions.
            best_params_ (dict): Stores the optimal parameters found during grid search.
        """
        self.classifier = RandomForestClassifier()
        self.best_params_ = None
    
    def _create_pipeline(self, X):
        """
        Create a pipeline with preprocessing steps and the classifier.
        
        Parameters:
            X (array-like): Features of the data.
        
        Returns:
            Pipeline: A pipeline containing preprocessing steps and the classifier.
        """
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features)])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', self.classifier)])
        
        return pipeline
    
    def _validate_input(self, X, y):
        """Validate input data."""
        if not isinstance(X, (np.ndarray,)):
            raise ValueError("X must be a numpy array")
        if not isinstance(y, (np.ndarray,)):
            raise ValueError("y must be a numpy array")
        if len(X)!= len(y):
            raise ValueError("X and y must have the same number of samples")

    def fit(self, X, y):
        """
        Fit the model to the training data using randomized search for hyperparameter tuning.
        
        Parameters:
            X (array-like): Features of the training data.
            y (array-like): Target variable of the training data.
        """
        self._validate_input(X, y)
        
        # Create a pipeline with preprocessing steps and the classifier
        pipeline = self._create_pipeline(X)
        
        # Define hyperparameters for randomized search
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 5, 10]
        }
        
        # Perform randomized search to find optimal hyperparameters
        random_search = RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, cv=5, n_iter=10)
        random_search.fit(X, y)
        
        # Update the classifier with the best parameters
        self.classifier = random_search.best_estimator_.named_steps
