#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/29
  
"""
import numpy as np
from regression import Regression

class RidgeRegression(Regression):
    """Ridge regression
    w = argmin(|t - X*w| + a * |w| ^ 2)
    """
    def __init__(self, alpha):
        self.alpha = alpha
        
    def _fit(self, X, y):
        eye = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(self.alpha * eye + X.T.dot(X), X.T.dot(y))
        
    def _predict(self, X):
        y = X.dot(self.w)
        return y
    
        
    
