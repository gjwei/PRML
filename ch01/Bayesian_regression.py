#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/29
 
 看了半天还是没有看懂什么事贝叶斯回归
 Orz
 留下代码，今后慢慢琢磨吧
 
  
"""
import numpy as np
from ch01.regression import Regression


class BayesianRegressor(Regression):
    """
    Bayesian Regression
    w ~ N(w|0, alpha^(-1)I)
    y = X @ w
    t ~ N(t|X @ w, beta^(-1))
    """
    
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_precision = None
    
    def _fit(self, X, t):
        if self.w_mean is not None:
            mean_prev = self.w_mean
        else:
            mean_prev = np.zeros(np.size(X, 1))
        
        if self.w_precision is not None:
            precision_prev = self.w_precision
        else:
            precision_prev = self.alpha * np.eye(np.size(X, 1))
        
        w_precision = precision_prev + self.beta * np.dot(X.T, X)
        w_mean = np.linalg.solve(
                w_precision,
                precision_prev.dot(mean_prev) + self.beta * X.T.dot(t)
        )
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(self.w_precision)
    
    def _predict(self, X, return_std=False, sample_size=None):
        if isinstance(sample_size, int):
            w_sample = np.random.multivariate_normal(
                    self.w_mean, self.w_cov, size=sample_size
            )
            y = X.dot(w_sample.T)
            return y
        
        y = X.dot(self.w_mean)
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y


class EmpiricalBayesRegressor(Regression):
    def __init__(self, alpha=1., beta=1.):
        self.alpha = alpha
        self.beta = beta
    
    def _fit(self, X, t, max_iter=100):
        M = X.T @ X
        eigenvalues = np.linalg.eigvalsh(M)
        eye = np.eye(np.size(X, 1))
        N = len(t)
        for _ in range(max_iter):
            params = [self.alpha, self.beta]
            
            w_precision = self.alpha * eye + self.beta * X.T @ X
            w_mean = self.beta * np.linalg.solve(w_precision, X.T @ t)
            
            gamma = np.sum(eigenvalues / (self.alpha + eigenvalues))
            self.alpha = float(gamma / np.sum(w_mean ** 2).clip(min=1e-10))
            self.beta = float(
                    (N - gamma) / np.sum(np.square(t - X @ w_mean))
            )
            if np.allclose(params, [self.alpha, self.beta]):
                break
        self.w_mean = w_mean
        self.w_precision = w_precision
        self.w_cov = np.linalg.inv(w_precision)
    
    def log_evidence(self, X, t):
        """
        log evidence function
        Parameters
        ----------
        X : ndarray (sample_size, n_features)
            input data
        t : ndarray (sample_size,)
            target data
        Returns
        -------
        output : float
            log evidence
        """
        M = X.T @ X
        return 0.5 * (
            len(M) * np.log(self.alpha)
            + len(t) * np.log(self.beta)
            - self.beta * np.square(t - X @ self.w_mean).sum()
            - self.alpha * np.sum(self.w_mean ** 2)
            - np.linalg.slogdet(self.w_precision)[1]
            - len(t) * np.log(2 * np.pi)
        )
    
    def _predict(self, X, return_std=False, sample_size=None):
        if isinstance(sample_size, int):
            w_sample = np.random.multivariate_normal(
                    self.w_mean, self.w_cov, size=sample_size
            )
            y = X @ w_sample.T
            return y
        y = X @ self.w_mean
        if return_std:
            y_var = 1 / self.beta + np.sum(X @ self.w_cov * X, axis=1)
            y_std = np.sqrt(y_var)
            return y, y_std
        return y


if __name__ == '__main__':
    from ch01.polynormial import PolynomialFeatures
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
    
    feature = PolynomialFeatures(degree=2)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)
    
    model = BayesianRegressor(alpha=2e-3, beta=2)
    model.fit(X_train, y_train)
    
    y, y_err = model.predict(X_test, return_std=True)
    plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
    plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
    plt.plot(x_test, y, c="r", label="mean")
    plt.fill_between(x_test, y - y_err, y + y_err, color="pink", label="std.", alpha=0.5)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1.5, 1.5)
    plt.annotate("M=9", xy=(0.8, 1))
    plt.legend(bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0.)
    plt.show()
