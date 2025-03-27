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


# Problem 3 (b)

def kernel(b,z,d):
    p = b*z.T@z + 1
    r = pow(p,d)
    return r

def sigmoid(b,z,d):
     y = 1/(1+(np.exp((kernel(b,z,d)))))
     return y

def ll_2(b,z,y,d):
    m = X.shape[1]
    A = sigmoid(b,z,d)
    cost = -1.0/m*np.sum(y*np.log(A)+(1.0-y)*np.log(1.0-A))   
    return cost

def ll_grad(alpha, x, y,d):
    grad = np.zeros_like(alpha)
    for i in range(len(alpha)):
        grad += (sigmoid(-alpha @ x[i],x,d) - (1 - y[i])) * x[i]
    return grad


def ll_hessian(alpha, x,d,y, rho):
    h = np.zeros((x.shape[1], x.shape[1]))
    for i in range(len(alpha)):
        sig = sigmoid(alpha,x,d)
        h -= sig * (1 - sig) * np.outer(x[i], x[i])
    return h + rho * np.eye(h.shape[0])


def newton_method(x, y, d, lr, rho, num_iter, v=False):
    alpha = 0
    for i in range(num_iter):
        alpha = alpha + lr * (np.linalg.inv(ll_hessian(alpha, x,y,d, rho)) @ ll_grad(alpha, x, y,d))
        if v:
            print(ll_2(alpha, x, y,d))
        rho *= 0.99
    return alpha

def min_2(x,y,d):
    alpha_init = np.zeros(3)
    res = minimize(ll,alpha_init, args = (x,y),method = "BFGS")
    print(f'Optimal Alpha: {res}')
    return res

X = prep(data_x)

d1 = 5

d2 = 50

beta5 = min_2(X,group,d1).x

beta50 = min_2(X,group,d2).x

yy_25 = -(beta5[0] + beta5[1] * xx) / beta5[2]
yy_250 = -(beta50[0] + beta50[1] * xx) / beta50[2]