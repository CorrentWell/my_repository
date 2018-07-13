# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:33:22 2018

@author: yg2018
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
n = 40
omega = np.random.randn(1)
noise = 0.8 * np.random.randn(n, 1)
x = np.random.randn(n, 2)
tf = (omega * x[:, 0] + x[:, 1]).reshape(n,1) + noise > 0
y = 2 * tf - 1

P = np.array([X for X, Y in zip (x, y) if Y > 0])
N = np.array([X for X, Y in zip (x, y) if Y < 0])

plt.plot(P[:,0], P[:,1], "r.", label="Positive")
plt.plot(N[:,0], N[:,1], "b.", label="Negative")
plt.legend()