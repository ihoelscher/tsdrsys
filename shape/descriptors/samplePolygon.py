# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import distance

def uniSample(v, M):
    N = len(v)
    delta = (1/float(M))*pathLength(v)
    
    g = np.empty(M, dtype=complex)
    
    g[0] = cplex(v[0])
    
    i = 0
    k = 1
    dp = 0
    
    while (i < N) and (k < M):
        va = v[i]
        vb = v[(i+1) % N]
        
        gama = distance.euclidean(va,vb)
        if ((delta - dp) <= gama):
            x = va + ((delta-dp)/float(gama)) * (vb-va)
            g[k] = cplex(x)
            dp = dp - delta
            k = k + 1
        else:
            dp = dp + gama
            i = i + 1
    
    return g
    
def pathLength(v):
    N = len(v)
    L = 0
    
    for i in range(N):
        va = v[i]
        vb = v[(i+1) % N]
        L = L + distance.euclidean(va, vb)
    
    return L
    
def cplex(x):
    return complex(x[0], x[1])