# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:33:22 2018

@author: yg2018
"""
import numpy as np
import matplotlib as plt

n = 40
omega = np.random.randn(1, n)
noise = 0.8 * np.random.randn(n, 1)
x = np.random.randn(n, 2)
y = 2 * (omega * x[:, 1] + x[:, 2] + noise > 0) - 1