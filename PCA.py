#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/30
  
"""
import numpy as np
from scipy.linalg import svd
import logging
from BaseEstimator import BaseEstimator

class PCA(BaseEstimator):
    y_required = False
    
    def __init__(self, n_components, solver='svd'):
        """Principal component analysis (PCA) implementation.

        Transforms a dataset of possibly correlated values into n linearly
        uncorrelated components. The components are ordered such that the first
        has the largest possible variance and each following component as the
        largest possible variance given the previous components. This causes
        the early components to contain most of the variability in the dataset.

        Parameters
        ----------
        n_components : int
        solver : str, default 'svd'
            {'svd', 'eigen'}
        """
        self.solver = solver
        self.n_components = n_components
        self.components = None
        self.mean = None
        
    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self._decompose(X)
        
    def _decompose(self, X):
        # mean centering
        X = X.copy()
        X -= self.mean
        
        if self.solver == 'svd':
            _, s, Vh = svd(X)
        elif self.solver == 'eigen':
            s, Vh = np.linalg.eigh(np.cov(X.T))
            Vh = Vh.T
        else:
            raise ValueError("Wrong solver")
            
        s_squared = s ** 2
        variance_ratio = s_squared / s_squared.sum()
        logging.info('Explained variance ratio: %s' % (variance_ratio[0:self.n_components]))
        self.components = Vh[:self.n_components]

    def transform(self, X):
        X = X.copy()
        X -= self.mean
        return np.dot(X, self.components.T)

    def _predict(self, X=None):
        return self.transform(X)


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    
    X, y = make_classification(n_samples=10000, n_features=100, n_informative=75,
                               random_state=111, n_classes=2, class_sep=2.5)
    print(X.shape)
    X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    for s in ['svd', 'eigen']:
        p = PCA(n_components=25, solver=s)
        
        p.fit(X_train, y_train)
        
        X_train_reduce = p.transform(X_train)
        X_test_reduce = p.transform(X_test)
        
        model = LogisticRegression(C=0.1,  max_iter=1000)
        model.fit(X_train_reduce, y_train)
        
        print("Classification accuracy for %s pca: %s"
               %(s, accuracy_score(y_test, model.predict(X_test_reduce))))
        
        
        
    