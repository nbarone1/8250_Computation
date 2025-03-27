# Homework 10 Problem 4

import numpy as np
import matplotlib.pyplot as plt

tol = 1e-6

def loss_fun(y,A,x):
    l = y - A @ x
    loss = np.linalg.norm(l)**2
    return loss 

def grad(y,A,x):
    g = 2 * A.T @ ((A @ x) - y)
    return g

def new_x(y,A,x,alpha):
    g = grad(y,A,x)
    x_new = x - alpha*g
    for i in range(len(x)):
        x_new[i] = max(0,x_new[i])
    return x_new, g



def run(tol):
    dx = 1
    loss = []
    lr = .01
    y = np.random.uniform(size=(5,1))
    A = np.random.uniform(size=(5,10))
    x = np.ones(shape = (10,1))
    x_n = np.zeros(shape = (10,1))
    while dx > tol:
        loss.append(loss_fun(y,A,x))
        print(loss_fun(y,A,x))
        x_n, g = new_x(y,A,x,lr)
        dx = np.linalg.norm(x-x_n)
        x = x_n
    return x_n, loss
    
    
x, loss = run(tol)

plt.plot(np.array(loss),'red')
plt.show()

# Given y,A are random result is obtained within 1000,2500 iterations with the given learning rate.