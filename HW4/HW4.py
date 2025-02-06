# HW 4

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
    data = pd.read_csv(cwd+'\HW4\mnist.csv', header=None)
    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()
    #X = X/255


    # subsample N
    if N is not None:
        # idx = np.random.choice(X.shape[0], N, replace=False)
        X = X[N]
        # X = X[idx]
        # y = y[idx]


    return X, y


def show_image(x):
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.show()

def us(X,k):
    C = X.T @ X
    C = C/(len(X)-1)
    eval, evec = scipy.linalg.eig(C)
    u_s = X @ evec
    return u_s.astype('float64')

X,Y = load_MNIST()

centers = st.tmean(X, axis  = 0)
X = X - centers

show_image(X[1])

X_p = us(X,1000)

show_image(X_p[1])

