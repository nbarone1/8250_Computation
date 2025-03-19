# Homework 9

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Problem 1 (a)

data = pd.read_csv(r"C:\\Users\\npb28.la\\8250_Computation\\HW9\\dataset.csv")
data = data.to_numpy()

scatter_x = np.array(data[:,0])
scatter_y = np.array(data[:,1])
group = np.array(data[:,2])
cdict = {0: 'red', 1: 'blue'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 10)
ax.legend()
plt.show()

# Problem 1 (b)



