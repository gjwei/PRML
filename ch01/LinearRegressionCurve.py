#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/29
  
"""

import numpy as np
import matplotlib.pyplot as plt

from polynormial import PolynomialFeatures
from LinearRegression import LinearRegression


def create_toy_data(func, sample_size, std):
    X = np.linspace(0, 1, sample_size)
    y = func(X) + np.random.normal(scale=std, size=X.shape)
    return X, y


def func(x):
    return np.sin(2 * np.pi * x)


x_train, y_train = create_toy_data(func, 100, 0.2)
x_test = np.linspace(0, 1, 100)
y_test = func(x_test)

# plt.scatter(x_train, y_train, facecolor='None', edgecolors='b', s=50, label='training data')
# plt.plot(x_test, y_test, c='g', label='$\sin(2\pi x)$')
# plt.legend()
# plt.show()

for i, degree in enumerate([0, 1, 3, 9]):
    plt.subplot(2, 2, i + 1)
    features = PolynomialFeatures(degree)
    X_train = features.transform(x_train)
    X_test = features.transform(x_test)
    print("X_train shape", X_train.shape)
    print("X_test shape", X_test.shape)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y = model.predict(X_test)
    
    plt.scatter(x_train, y_train, edgecolors='b', s=50, label='training data')
    plt.plot(x_test, y_test, c='g', label='sin(2*pi*x)')
    plt.plot(x_test, y, c='r', label='fitting')
    plt.ylim(-1.5, 1.5)
plt.legend()

plt.show()

# rmse curse
def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))

train_errors = []
test_errors = []

for i in range(50):
    feature = PolynomialFeatures(degree=i)
    X_train = feature.transform(x_train)
    X_test = feature.transform(x_test)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    train_errors.append(rmse(model.predict(X_train), y_train))
    test_errors.append(rmse(y_test, y_pred))
    
plt.plot(train_errors, 'o-', mfc=None, mec='b', ms=10, c='b', label='training')
plt.plot(test_errors, 'o-', mfc="none", mec="r", ms=10, c="r", label="Test")
plt.legend()
plt.xlabel("degree")
plt.ylabel("RMSE")
plt.show()


