#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/29
  
"""
import itertools
import functools
import numpy as np


class PolynomialFeatures(object):
    """
    polynomial features
    
    transform input array with polynomial features
    Example
    =======
    x =
    [[a, b],
    [c, d]]
    y = PolynomialFeatures(degree=2).transform(x)
    y =
    [[1, a, b, a^2, a * b, b^2],
    [1, c, d, c^2, c * d, d^2]]
    """
    
    def __init__(self, degree=2):
        assert isinstance(degree, int)
        self.degree = degree
    
    def transform(self, x):
        """
        transform input array with polynomial features
        :param x: (sample_size, n) array
        :return: (sample_size, 1+nC1+....+nCd) ndarray
        """
        if x.ndim == 1:
            x = x[:, None]
        x_t = x.transpose()
        features = [np.ones(len(x))]
        for degree in range(1, self.degree + 1):
            for item in itertools.combinations_with_replacement(x_t, degree):
                features.append(functools.reduce(lambda i, y: i * y, item))
        
        return np.asarray(features).transpose()
