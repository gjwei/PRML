#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/29
  
"""
from regression import Regression
import numpy as np


class LinearRegression(Regression):
    """Linear regression model
    y = X * w
    t ~ N(t|X@w, var)
    """
    
    def _fit(self, X, y):
        self.w = np.linalg.pinv(X).dot(y)
        self.var = np.mean(np.square(np.dot(X, self.w) - y))
    
    def _predict(self, X, return_std=False):
        y_ = X.dot(self.w)
        if return_std:
            y_std = np.sqrt(self.var) + np.zeros_like(y_)
            return y_, y_std
        return y_
