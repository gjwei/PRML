#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/29
  
"""
import numpy as np


class Regression(object):
    """
    Base class of regression
    """
    
    def fit(self, X, y, **kwargs):
        """
        estimates parameters given
        Parameters
        ----------
        X : (sample_size, n_features) np.ndarray
            training data input
        y : (sample_size,) np.ndarray
            training data target
        """
        self._check_input(X)
        self._check_target(y)
        
        if hasattr(self, '_fit'):
            self._fit(X, y, **kwargs)
        else:
            raise NotImplementedError
    
    def predict(self, X):
        """
        predict outputs of the model
        :param X: (sample_size, n_features), array
        :return: (sample_size,) ndarray, prediction of each sample
        """
        self._check_input(X)
        if hasattr(self, '_predict'):
            return self._predict(X)
        else:
            raise NotImplementedError
    
    def _check_input(self, X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X(input) is not np.ndarry")
        if X.ndim != 2:
            raise ValueError("X(Input) is not two dimensional arrya")
        if hasattr(self, 'n_features') and self.n_features != np.size(X, 1):
            raise ValueError(
                    "mismatch in dimension 1 of X(input) "
                    "(size {} is different from {})"
                        .format(np.size(X, 1), self.n_features)
            )
    
    def _check_target(self, y):
        if not isinstance(y, np.ndarray):
            raise ValueError("target must be np.ndarray")
        if y.ndim != 1:
            raise ValueError("target must be one dimenional array")
