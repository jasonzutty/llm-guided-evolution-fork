
# ========== Start: GeneCrossed
from sklearn.ensemble import RandomForestClassifier
import numpy as np
# ========== End:

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin

class Model(BaseEstimator, ClassifierMixin):
    def __init__(self):
        """
        Initialize the model with four base classifiers and a meta-classifier.
        """
        self.random_forest = RandomForestClassifier()
        self.gradient_boosting = GradientBoostingClassifier()
        self.support_vector_machine = SVC(probability=True)
        self.logistic_regression = LogisticRegression(max_iter=1000)
        self.meta_voting = VotingClassifier(estimators=[
            ('random_forest', self.random_forest), 
            ('gradient_boosting', self.gradient_boosting),
            ('support_vector_machine', self.support_vector_machine),
            ('logistic_regression', self.logistic_regression)], voting='soft')

    def fit(self, X, y):
        """
        Fit all base classifiers and then the meta-classifier.
        
        Parameters:
        - X: Training features
        - y: Target variable
        """
        # Train the base classifiers
        self.random_forest.fit(X, y)
        self.gradient_boosting.fit(X, y)
        self.support_vector_machine.fit(X, y)
        self.logistic_regression.fit(X, y)
        
        # Train the meta-classifier
        self.meta_voting.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        - X: Features to predict
        
        Returns:
        - Predictions made by the model
        """
        # Use the meta-classifier to make the final predictions
        return self.meta_voting.predict(X)

# --OPTION--
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin

class Model(BaseEstimator, ClassifierMixin):
    def __init__(self):
        """
        Initialize the model with three base classifiers and a meta-classifier.
        """
        self.random_forest = RandomForestClassifier()
        self.gradient_boosting = GradientBoostingClassifier()
        self.support_vector_machine = SVC(probability=True)
        self.meta_voting = VotingClassifier(estimators=[
            ('random_forest', self.random_forest), 
            ('gradient_boosting', self.gradient_boosting),
            ('support_vector_machine', self.support_vector_machine)], voting='soft')

    def fit(self, X, y):
        """
        Fit all base classifiers and then the meta-classifier.
        
        Parameters:
        - X: Training features
        - y: Target variable
        """
        # Train the base classifiers
        self.random_forest.fit(X, y)
        self.gradient_boosting.fit(X, y)
        self.support_vector_machine.fit(X, y)
        
        # Train the meta-classifier
        self.meta_voting.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        - X: Features to predict
        
        Returns:
        - Predictions made by the model
        """
        # Use the meta-classifier to make the final predictions
        return self.meta_voting.predict(X)

