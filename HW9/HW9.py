# Homework 9

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# Problem 1 (a)

data = pd.read_csv(r"C:\\Users\\npb28.la\\8250_Computation\\HW9\\dataset.csv")
data = data.to_numpy()

data_x = data[:, [0, 1]]

scatter_x = np.array(data[:,0])
scatter_y = np.array(data[:,1])
group = np.array(data[:,2])
cdict = {0: 'red', 1: 'blue'}

plt.scatter(data_x[:, 0], data_x[:, 1], c=group, cmap='bwr', edgecolors='k', alpha=0.7)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter Plot of Dataset')
plt.savefig(r"C:\\Users\\npb28.la\\8250_Computation\\HW9\\ScatterPlot.png", dpi=300, bbox_inches='tight')
plt.show()

# Problem 1 (b)

def prep(x):
    ones_column = np.ones((x.shape[0], 1))
    return np.hstack((ones_column, x))

def sigmoid(x):
     y = 1/(1+(np.exp((x))))
     return y

def ll(W,X,Y):
    m = X.shape[1]
    z = X @ W
    A = sigmoid(z)
    cost = -1.0/m*np.sum(Y*np.log(A)+(1.0-Y)*np.log(1.0-A))   
    return cost

def min(x,y):
    alpha_init = np.zeros(3)
    res = minimize(ll,alpha_init, args = (x,y),method = "BFGS")
    print(f'Optimal Alpha: {res}')
    return res

X = prep(data_x)

alpha_opt = min(X,group).x

# Problem 1 (c)

def pred(x,alpha,y):
    probs = sigmoid(x @ alpha)
    y_pred = (probs >= 0.5).astype(int)
    accuracy = np.mean(y_pred == y)
    print(f'Accuracy: {accuracy * 100:.2f}%')





xx = np.linspace(data_x[:, 0].min(), data_x[:, 0].max(), 100)
yy = -(alpha_opt[0] + alpha_opt[1] * xx) / alpha_opt[2]  

plt.scatter(data_x[:, 0], data_x[:, 1], c=group, cmap='bwr', edgecolors='k', alpha=0.7)
plt.plot(xx, yy, 'k-', label='Decision Boundary')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.savefig(r"C:\\Users\\npb28.la\\8250_Computation\\HW9\\logistic_regression_output.png", dpi=300, bbox_inches='tight')
plt.show()

pred(X,alpha_opt)