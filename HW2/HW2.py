# HW2 MATH 8250 Problem 4

import os
import numpy as np
import scipy.linalg
import scipy.stats as st
import scipy
import pandas as pd

# import data
curdir = os.getcwd()
data = pd.read_csv(curdir+'\HW2\senators.csv')
sen_party = data.iloc[:,0].str
data = data.iloc[:,1:]
sen_party = sen_party[-1]
data = pd.DataFrame(data.values)

# dimension of PCA
k = 2

# Centering PCA
centers = st.tmean(data, axis  = 0)
data = data - centers
dt = data.T

cov = data.T @ data
cov_norm = cov / scipy.linalg.norm(cov, axis = 0)
eval, evec = scipy.linalg.eig(cov_norm)


