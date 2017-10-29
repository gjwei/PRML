#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/29

"""
import numpy as np
from pylab import *
import sys

N = 100
M = 35


def generate_data():
    x = np.linspace(0, 1, N)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, x.size)
    return x, y


def y_pred(x, wlist):
    """计算M阶累加结果"""
    result = 0
    for i in range(M + 1):
        result += wlist[i] * (x ** i)
    
    return result


def estimate(xlist, ylist, lam):
    """根据现有的数据，求的W"""
    A = []
    for i in range(M + 1):
        for j in range(M + 1):
            temp = (xlist ** (i + j)).sum()
            if i == j:
                temp += lam
            A.append(temp)
    A = array(A).reshape(M + 1, M + 1)

    T = []
    for i in range(M + 1):
        T.append(((xlist ** i) * ylist).sum())
    T = array(T)

    wlist = np.linalg.solve(A, T)

    return wlist


def main():
    xlist, ylist = generate_data()
    wlist = estimate(xlist, ylist, np.exp(-7))
    print(wlist)
    
    
    xs = np.linspace(0, 1, 500)
    ideal = np.sin(2 * np.pi * xs)  # 理想曲線
    model = [y_pred(x, wlist) for x in xs]  # 推定モデル
    
    plot(xlist, ylist, 'bo')
    plot(xs, ideal, 'g-')
    plot(xs, model, 'r-')
    show()


if __name__ == '__main__':
    main()
