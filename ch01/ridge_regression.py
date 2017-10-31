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

if __name__ == '__main__':
    """对ridge regression 进行评估"""
    from .polynormial import PolynomialFeatures
    import matplotlib.pyplot as plt

    def create_toy_data(func, sample_size, std):
        X = np.linspace(0, 1, sample_size)
        y = func(X) + np.random.normal(scale=std, size=X.shape)
        return X, y

    def rmse(a, b):
        return np.sqrt(np.mean(np.square(a - b)))

    def func(x):
        return np.sin(2 * np.pi * x)


    x_train, y_train = create_toy_data(func, 100, 0.2)
    x_test = np.linspace(0, 1, 100)
    y_test = func(x_test)
    
    error_train = []
    error_test = []
    for i in range(10):
        feature = PolynomialFeatures(degree=i)
        X_train = feature.transform(x_train)
        X_test = feature.transform(x_test)
        
        model = RidgeRegression(alpha=0.1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        error_train.append(rmse(y_train, model.predict(X_train)))
        error_test.append(rmse(y_test, y_pred))
        
    # 对错误进行可视化
    plt.plot(error_train, 'o-', mfc=None, mec='b', ms=10, c='b', label='training')
    plt.plot(error_test, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
    plt.legend()
    plt.xlabel("degree")
    plt.ylabel("RMSE")
    plt.show()

        
        