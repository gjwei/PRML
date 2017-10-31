#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/30
  
"""
import numpy as np


def cov(X):
    """
    Covariance matrix
    note: specifically for mean-centered data
    note: numpy's `cov` uses N-1 as normalization
    """
    return np.dot(X.T, X) / X.shape[0]


def pca(X, k):
    average = np.mean(X, axis=0)
    m, n = np.shape(X)
    X_new = X - average
    X_new /= np.std(X_new, axis=0)
    C = cov(X_new)
    E, V = np.linalg.eig(C)  # 求解协方差矩阵的特征值和特征向量
    key = np.argsort(E)[::-1][:k]  # 按照E进行从大到小排序
    E, V = E[key], V[key]
    U = np.dot(X_new, V)
    return U, E, V


def pca_svd(X):
    X_new = X - np.mean(X, axis=0)
    X_new = X / np.std(X, axis=0)
    
    U, E, V = np.linalg.svd(X_new)
    
    
