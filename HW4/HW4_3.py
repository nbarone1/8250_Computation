# HW 4 Problem 3

import random
import numpy as np
from scipy import stats as st

def rand():
    return random.randint(1,4)

# a

def sample_signal(p,m):
    s = np.zeros(p)
    s[0] = rand()
    i = 1
    for i in range(p):
        if random.random(0,1) < m:
            s[i] = s[i-1]
        else:
            s[i] = rand()
        i = i+1
    return s


# B

def sample_data(n,p,m):
    sd = np.zeros((n,p))
    i = 0
    for i in range(k):
        sd[i] = sample_signal(p,m)
        i = i+1
    return sample_data

# C

def row_centering(X):
    centers = st.tmean(X, axis  = 1)
    X = X - centers
    return X

def M_squared(X):
    C = X.T @ X
    C = C/(len(X)-1)
    M = np.linalg.cholesky(C)
    return M

def z_transform(X):
    Y = row_centering(X)
    M = M_squared(Y)
    Z = np.linalg.inv(M) @ Y