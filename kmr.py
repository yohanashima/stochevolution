# -*- coding: utf-8 -*-
from __future__ import division  
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from scipy.stats import binom
from discrete_rv import DiscreteRV
from mc_tools.py import mc_compute_stationary, mc_sample_path

#マルコフ連鎖の遷移行列を定義

def mk_matrix0(N, ep, p):  #逐次改訂
    P = np.zeros((N+1, N+1))
    P[N, N-1], P[N, N] = ep/2, 1-ep+ep/2  #N行は直接確率入力
    for i in range(N):
        if i <= (N-1)*p:    #期待利得が等しい場合０を選ぶ
            P[i, i-1] = i/N*((1-ep)+ep/2)
            P[i, i] = (1-i/N)*((1-ep)+ep/2)+i/N*ep/2
            P[i, i+1] = (1-i/N)*ep/2
        elif (N-1)*p < i <= (N+1)*p+1:
            P[i, i-1] = i/N*((1-ep)+ep/2)
            P[i, i] = (1-i/N)*ep/2+i/N*ep/2
            P[i, i+1] = (1-i/N)*((1-ep)+ep/2)
        else:
            P[i, i-1] = i/N*ep/2
            P[i, i] = (1-i/N)*ep/2+i/N*((1-ep)+ep/2)
            P[i, i+1] = (1-i/N)*((1-ep)+ep/2)
    return P

def mk_matrix1(N, ep, p):   #同時改訂
    P = np.empty((N+1, N+1))
    for i in range(N):
        if i/N < p:
		    pro = ep/2
        elif i/N == p:
            pro = 1/2
        else:
            pro = 1-ep/2
        P[i] = binom.pmf(range(N+1), N, pro)
    return P

#変数
N = 15    #人数
T = 1000  #試行回数
ep = 0.2  #epsilon
p = 1/3   #期待利得が等しくなるときの確率

#P = mk_matrix0(N, ep, p)
P = mk_matrix1(N, ep, p)

X_t = mc_sample_path(P, init=randint(1, N), sample_size=T)

plt.plot(X_t)
plt.savefig('kmr0_distribution.png')
plt.show()

barX = []
for i in range(N+1):
    count = sum([r == i for r in X_t])
    barX.append(count)
plt.plot(barX)
plt.savefig('kmr0_frequency.png')
plt.show()

SD = mc_compute_stationary(P)
plt.hist(SD)
plt.savefig('kmr0_hist.png')
plt.show()
		
		
		
		
		