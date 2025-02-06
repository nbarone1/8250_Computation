# HW 4 Problem 3

import random
import numpy as np

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

