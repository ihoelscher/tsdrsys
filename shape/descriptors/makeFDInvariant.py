# -*- coding: utf-8 -*-

import numpy as np

def makeFDInvariant(G):
    print(makeScaleInvariant(G))
    Ga, Gb = makeStartPointInvariant(G)
    print(makeRotationInvariant(Ga))
    print(makeRotationInvariant(Gb))
    
    return Ga, Gb

def makeScaleInvariant(G):
    M = len(G)
    
    s = np.sum([np.square(np.abs(G[-m])) + np.square(np.abs(G[m])) for m in range(1, M/2+1)])
    
    v = 1/float(np.sqrt(s))
    
    G[1:] = v*G[1:]
    
    return v
    
def makeRotationInvariant(G):
    z = 0 + 0j

    M = len(G)    
    
    z = np.sum([(1/float(m))*(G[-m] + G[m]) for m in range(1, M/2+1)])
    
    beta = np.angle(z)
    
    G[1:] = np.exp(-beta*1j)*G[1:]
    
    return beta
    
def makeStartPointInvariant(G):
    fi = getStartPointPhase(G)
    Ga = shiftStartPointPhase(G, fi)
    Gb = shiftStartPointPhase(G, fi + np.pi)
    
    return Ga, Gb
    
def getStartPointPhase(G):
    cmax = -float('inf')
    fimax = 0
    K = 400

    for k in range(K):
        fi = np.pi*k/float(K)
        c = fp(G, fi)
        if c > cmax:
            cmax = c
            fimax = fi
    
    return fimax

def fp(G, fi):
    s = 0
    M = len(G)
    
    for m in range(1, M/2+1):
        z1 = G[-m]*np.exp(-m*fi*1j)
        z2 = G[m]*np.exp(m*fi*1j)
        
        s += np.real(z1) * np.imag(z2) - np.imag(z1) * np.real(z2)
    
    return s
    
def shiftStartPointPhase(G, fi):
    GG = np.copy(G)
    M = len(G)
    
    for m in range(1, M/2+1):
        GG[-m] = G[-m]*np.exp(-m*fi*1j)
        GG[m] = G[m]*np.exp(m*fi*1j)
    
    return GG