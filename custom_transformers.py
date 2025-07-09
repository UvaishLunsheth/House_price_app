import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # Nothing to fit

    def transform(self, X):
        X = X.copy()
        rooms_per_household = X['total_rooms'] / X['households']
        population_per_household = X['population'] / X['households']
        
        result = X.copy()
        result['rooms_per_household'] = rooms_per_household
        result['population_per_household'] = population_per_household
        
        if self.add_bedrooms_per_room:
            result['bedrooms_per_rooms'] = X['total_bedrooms'] / X['total_rooms']
        
        return result
        
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            X[col + '_log'] = np.log1p(X[col])
        return X
