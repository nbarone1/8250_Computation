# HW6

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Problem 1

X1 = np.random.uniform(0, 10, 1000)
X2 = np.ones(1000)
X = np.hstack((X2.reshape(-1,1),X1.reshape(-1,1)))

E = np.random.uniform(0,1,1000)

Y = np.ones(1000)+2*X1+E

a = np.linalg.inv(X.T @ X) @ X.T @ Y

y_bar = a[0] + a[1]*X1

loss = np.linalg.norm(Y-y_bar)

fig, ax = plt.subplots()
ax.scatter(X1,Y, c= "red")
ax.plot(X1,y_bar, c = "blue")
plt.title(loss)
plt.show()

# Problem 2
# Choose 1000 discrete points in order to discretize

x = np.linspace(1,2,1000)

