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

a1 = np.exp(x)
a2 = np.sin(x)
a3 = gamma(x)
A = np.hstack((a1.reshape(-1,1),a2.reshape(-1,1),a3.reshape(-1,1)))

Y2 = 1/x

alpha = np.linalg.inv(A.T @ A) @ A.T @ Y2

est_a = alpha[0]*a1 + alpha[1]*a2 + alpha[2]*a3

loss2 = np.linalg.norm(Y2-est_a)

fig2, ax2 = plt.subplots()
ax2.scatter(x,Y2, c= "red")
ax2.plot(x,est_a, c = "blue")
plt.title(loss2)
plt.show()

# Problem 3