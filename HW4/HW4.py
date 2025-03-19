# HW 4 Problem 2

import os
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.stats as st
import scipy
import pandas as pd
import matplotlib.pyplot as plt

cwd = os.getcwd()
print(cwd)

def load_MNIST(N=None, s=1):
    print("Loading MNIST dataset...")
    data = pd.read_csv("C:\Users\\npb28.la\Downloads\mnist.csv", header=None)
    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()
    X = X/255


    # subsample N

    for i in range(len(y)):
        if y[i] == 3:
            y[i] == 1
        else:
            y[i] == 0

    if N is not None:
        # idx = np.random.choice(X.shape[0], N, replace=False)
        X = X[:N,:]
        # X = X[idx]
        y = y[:N,:]


    return X, y


def show_image(x):
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()

def norm_eig(X,k):
    cov = X.T @ X
    # cov_norm = cov / scipy.linalg.norm(cov, axis = 0)
    # eval, evec = scipy.linalg.eig(cov_norm)
    eval, evec = scipy.linalg.eig(cov)
    idx = eval.argsort()[::-1]
    eval = eval[idx]
    evec = evec[:,idx]
    V = evec[:,:k]
    return V

X,Y = load_MNIST(1000)

show_image(X[0])

# k = 10 has the shape but it blurry
# k = 25 starts gets worse
# k = 15 things are still worse than 10
# k = 8 yielded mild results

V = norm_eig(X,8)

X_p = V @ V.T @ X[0]

show_image(X_p.astype('float64'))
