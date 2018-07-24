# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:33:22 2018

@author: yg2018
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_data2(map_plot=True):
    np.random.seed(100)
    #seed値固定
    n = 40
    #データサイズ
    omega = np.random.randn(1)
    noise = 0.8 * np.random.randn(n, 1)
    #列ベクトルにする
    x = np.random.randn(n, 2)
    #入力の座標データ
    tf = (omega * x[:, 0] + x[:, 1]).reshape(n,1) + noise > 0 
    '''reshapeを入れないとnoiseが加わった段階で二次元配列化する。
    numpyでは列ベクトルはn*1の二次元配列として扱われているため
    reshapeを挟まない列ベクトルとreshapeした列ベクトルの足し算が二次元化してしまう。'''
    y = 2 * tf - 1
    y = np.reshape(y, -1)
    #教師データ、1がPositive, 0がNegative
    
    P = np.array([X for X, Y in zip (x, y) if Y > 0])
    N = np.array([X for X, Y in zip (x, y) if Y < 0])
    '''x,yをインデクスをそろえてfor構文で回してYが0より大きいときにPositiveとした
    このように書くことで教師データごとに集合を分ける'''
    #df = pd.DataFrame("P":P, "N":N))
    
    if map_plot:
        plt.plot(P[:,0], P[:,1], "r.", label="Positive")
        plt.plot(N[:,0], N[:,1], "b.", label="Negative")
        plt.legend()
    
    return x, y
    '''出力に注釈をつけた'''

if __name__ == "__main__":
    x, y = make_data2(map_plot=True)