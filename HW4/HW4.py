# HW 4

import os
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.stats as st
import scipy
import pandas as pd
import matplotlib.pyplot as plt



def load_MNIST(N=None, s=1):
    print("Loading MNIST dataset...")
    data = pd.read_csv('mnist.csv', header=None)
    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()
    #X = X/255


    # subsample N
    if N is not None:
        idx = np.random.choice(X.shape[0], N, replace=False)
        X = X[idx]
        y = y[idx]


    return X, y









