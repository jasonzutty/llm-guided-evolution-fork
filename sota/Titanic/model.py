# --OPTION--
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class Model:
    def __init__(self):
      self.classifier = RandomForestClassifier() 
    
    def fit(self, X, y):
      self.classifier.fit(X, y)

    def predict(self, X):
       return self.classifier.predict(X)
