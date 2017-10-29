#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
 created by gjwei on 2017/10/29
  
"""
import numpy as np
from pylab import *


# 训练数目
N = 10

xlist = np.linspace(0, 1, N)
print(xlist)
ylist = np.sin(2 * np.pi * xlist) + np.random.normal(0, 0.2, xlist.size)
print(ylist)

xs = np.linspace(0, 1, 1000)
ideal = np.sin(2 * np.pi * xs)

plot(xlist, ylist, 'bo')
plot(xs, ideal, 'g-')

xlim(0, 1)
ylim(-1.5, 1.5)
show()

