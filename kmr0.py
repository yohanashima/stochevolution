# -*- coding: utf-8 -*-
from __future__ import division  
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from discrete_rv import DiscreteRV
from mc_tools.py import mc_compute_stationary, mc_sample_path

#変数
N = 15
T = 1000
ep = 0.2  #epsilon

#マルコフ連鎖の遷移行列を定義
P = np.zeros((N, N))
P[N-1, N-2], P[N-1, N-1] = ep/2, 1-ep+ep/2  #N行は直接確率入力
for i in range(N-1):
    if i <= (N-1)/3:    #期待利得が等しい場合０を選ぶ
        P[i, i-1] = i/N*((1-ep)+ep/2)
        P[i, i] = (1-i/N)*((1-ep)+ep/2)+i/N*ep/2
        P[i, i+1] = (1-i/N)*ep/2
    elif (N-1)/3 < i <= (N+2)/3:
        P[i, i-1] = i/N*((1-ep)+ep/2)
        P[i, i] = (1-i/N)*ep/2+i/N*ep/2
        P[i, i+1] = (1-i/N)*((1-ep)+ep/2)
    else:
        P[i, i-1] = i/N*ep/2
        P[i, i] = (1-i/N)*ep/2+i/N*((1-ep)+ep/2)
        P[i, i+1] = (1-i/N)*((1-ep)+ep/2)

X_t = mc_sample_path(P, init=randint(1, N), sample_size=T)

plt.plot(X_t)
plt.show()

barX = []
for i in N:
    count = sum([r == i for r in X_t])
    barX.append(count)
plt.plot(barX)
plt.show()

SD = mc_compute_stationary(P)
plt.hist(SD)
plt.show()
		
		
		
		
		