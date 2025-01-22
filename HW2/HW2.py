# HW2 MATH 8250 Problem 4

import os
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.stats as st
import scipy
import pandas as pd
import matplotlib.pyplot as plt

# import data
curdir = os.getcwd()
data = pd.read_csv(curdir+'\HW2\senators.csv')
sen_party = data.iloc[:,0].str
data = data.iloc[:,1:]
sen_party = sen_party[-1]
sen_party = np.reshape(sen_party, (-1, 1))
data = pd.DataFrame(data.values)

# Part B

# dimension of PCA
k = 2

# Centering PCA
centers = st.tmean(data, axis  = 0)
data = data - centers
dt = data.T

cov = data @ data.T
cov_norm = cov / scipy.linalg.norm(cov, axis = 0)
eval, evec = scipy.linalg.eig(cov_norm)

idx = eval.argsort()[::-1]
eval = eval[idx]
evec = evec[:,idx]

V = evec[:,:k]

proj_2d = (V @ V.T) @ data

results = np.hstack((sen_party,V))

plt.scatter(V[:,0],V[:,1])
plt.show()

# Part C

dem = []


