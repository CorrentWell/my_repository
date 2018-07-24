# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 16:35:31 2018

@author: yg2018
"""
import numpy as np
import cvxopt
from cvxopt import matrix

mu = np.array([1, 2]).reshape(2, -1)
A = np.array([3, 0.5, 0.5, 1]).reshape(2, -1)
inv_A = np.linalg.inv(2*A)
eival, eivec = np.linalg.eig(inv_A)
ita = 1/np.max(eival)
r = [2, 4, 6]

def f(w):
    return (w - mu).T * A * (w - mu) - r*np.linalg.norm(w, 1)

w = np.random.rand(2)