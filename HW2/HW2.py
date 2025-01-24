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

def norm_eig(X,k):
    cov = X.T @ X
    cov_norm = cov / scipy.linalg.norm(cov, axis = 0)
    eval, evec = scipy.linalg.eig(cov_norm)
    idx = eval.argsort()[::-1]
    eval = eval[idx]
    evec = evec[:,idx]
    V = evec[:,:k]
    return V


# Part B

# dimension of PCA
k = 2

# Centering PCA
centers = st.tmean(data, axis  = 0)
data = data - centers

V = norm_eig(data,k)

proj_2d = data @ V

cdict = {'R': 'red', 'D': 'blue', 'I': 'yellow'}

fig, ax = plt.subplots()
px = proj_2d[0]
py = proj_2d[1]
for g in np.unique(sen_party):
    ixp = np.where(sen_party == g)
    ax.scatter(px[ixp[0]],py[ixp[0]],c = cdict[g], label = g)
ax.legend()
plt.show()

# Part C

for g in np.unique(sen_party):
    ixp = np.where(sen_party == g)
    if g == "R":
        rep = data.iloc[ixp[0],:]
    if g == "D":
        dem = data.iloc[ixp[0],:]
    if g == "I":
        ind = data.iloc[ixp[0],:]

center_d = st.tmean(dem, axis  = 0)
center_r = st.tmean(rep, axis  = 0)

diff = center_d - center_r

proj_diff = data @ diff

V2 = norm_eig(data,1)

proj_1d = data @ V2

fig2, ax2 = plt.subplots()
for g in np.unique(sen_party):
    ixp = np.where(sen_party == g)
    ax2.scatter(proj_diff[ixp[0]],np.zeros_like(proj_diff[ixp[0]]),c = cdict[g], label = g)
ax2.legend()
plt.show()

fig3, ax3 = plt.subplots()
for g in np.unique(sen_party):
    ixp = np.where(sen_party == g)
    ax3.scatter(proj_1d.iloc[ixp[0]],np.zeros_like(proj_1d.iloc[ixp[0]]),c = cdict[g], label = g)
ax3.legend()
plt.show()