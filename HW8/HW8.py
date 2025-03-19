# Homework 8 

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats as norm
import os

def load_MNIST(N=None, s=1):
    print("Loading MNIST dataset...")
    data = pd.read_csv(r"C:\\Users\\npb28.la\Downloads\\mnist.csv", header=None)
    y = data.iloc[:, 0].to_numpy()
    X = data.iloc[:, 1:].to_numpy()
    X = X/255

    for i in range(len(y)):
        if y[i] == 3:
            y[i] = 1
        else:
            a = y[i]
            y[i] = 0

    ones_column = np.ones((X.shape[0], 1))

    X = np.hstack((ones_column, X))

    # subsample N
    if N is not None:
        # idx = np.random.choice(X.shape[0], N, replace=False)
        X = X[:N,:].transpose()
        # X = X[idx]
        y = y[:N,:].transpose()

        XT = X[N:N+100,:].transpose()
        YT = y[N:N+1,:].transpose()


    return X, y, XT, YT

X,y = load_MNIST(1000)

# Problem 1

x0 = np.array([4,4])
l_rate = .0002  
tol = 1e-12
tol2 = 1e-7

def cost_fun(x0):
    x = x0[0]
    y = x0[1]
    return 100*(y-x**2)**2+(1-x)**2

def grad(x0):
    x = x0[0]
    y = x0[1]
    return np.array([-2.0*(1 - x) - 4.0*100*(y - x**2)*x,
                     2.0*100*(y - x**2)])

def grad_desc(x0,l_rate,tol):
    dx = 1
    x_old = x0.copy()
    g = grad(x_old)
    iter = 0

    while dx > tol:
        xp = x_old - l_rate*g

        dx = np.linalg.norm(xp-x_old)

        g_new = grad(xp)

        # l_rate_new = np.abs(np.transpose(xp) @ (g_new - g)) / np.linalg.norm(g_new - g)**2
        # l_rate = min(10*l_rate, l_rate_new)
        
        g = g_new
        
        x_old = xp

        iter += 1

    print(iter)
    print(x_old)
    print(np.linalg.norm(np.ones(2)-x_old))
    print(cost_fun(x_old))

    #252882 iterations. unsure what is going on.

    return x_old

def gradgrad(x0):
    x=x0[0]
    y=x0[1]
    g11 = -400.0 * (y - x**2) + 800.0 * x**2 + 2
    g12 = - 400.0 * x
    g21 = - 400.0 * x
    g22 = 200.0
    return np.array([[g11, g12], [g21, g22]])

def newt(x0,tol):
    dx = 1
    x_old = x0.copy()
    g = grad(x_old)
    iter = 0

    while dx > tol:
        s = np.linalg.matmul(np.linalg.inv(gradgrad(x_old)),grad(x_old))
        xp = x_old - s
        dx = np.linalg.norm(xp-x_old)
        x_old = xp

        iter += 1

    print(iter)
    print(x_old)
    print(np.linalg.norm(np.ones(2)-x_old))
    print(cost_fun(x_old))    

    # 6 iterations

    return

grad_desc(x0,l_rate,tol)

newt(x0,tol2)

# Problem 2

X,Y, XT, YT = load_MNIST(1000)

def sigmoid(x):
     y = 1/(1+(np.exp((x))))
     return y

def ll(W,X,Y):
    m = X.shape[1]
    z = np.dot(W.T,X)
    A = sigmoid(z)
    cost = -1.0/m*np.sum(Y*np.log(A)+(1.0-Y)*np.log(1.0-A))   
    return cost

def propagate(W,X,Y):
    m = X.shape[1]

    z = np.dot(W.T,X)
    A = sigmoid(z)
    cost = -1.0/m*np.sum(Y*np.log(A)+(1.0-Y)*np.log(1.0-A))  
    
    dw = 1.0/m*np.dot(X, (A-Y).T)
    db = 1.0/m*np.sum(A-Y)
    
    assert (dw.shape == W.shape)
    assert (db.dtype == float)
    
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    
    grads = {"dw": dw, 
             "db":db}
    
    return grads, cost

def optimize(w, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w, X, Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w - learning_rate*dw
        
        if i % 100 == 0:
            costs.append(cost)
            
        if print_cost and i % 100 == 0:
            print ("Cost (iteration %i) = %f" %(i, cost))
            
    grads = {"dw": dw, "db": db}
    params = {"w": w, "b": b}
        
    return params, grads, costs

def predict (w, X):  
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    A = sigmoid (np.dot(w.T, X))
    
    for i in range(A.shape[1]):
        if (A[:,i] > 0.5): 
            Y_prediction[:, i] = 1
        elif (A[:,i] <= 0.5):
            Y_prediction[:, i] = 0
            
    assert (Y_prediction.shape == (1,m))
    
    return Y_prediction

def model (X_train, Y_train, X_test, Y_test, num_iterations = 1000, learning_rate = 0.5, print_cost = False):
    W = np.zeros(X.shape[1],1)
    
    parameters, grads, costs = optimize(w, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]

    Y_prediction_test = predict (w, X_test)
    Y_prediction_train = predict (w, X_train)
    
    train_accuracy = 100.0 - np.mean(np.abs(Y_prediction_train-Y_train)*100.0)
    test_accuracy = 100.0 - np.mean(np.abs(Y_prediction_test-Y_test)*100.0)
    
    d = {"costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
         "w": w,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    
    print ("Accuarcy Test: ",  test_accuracy)
    print ("Accuracy Train: ", train_accuracy)
    
    return d

def ll_grad(alpha, x, y):
    grad = np.zeros_like(alpha)
    for i in range(len(alpha)):
        grad += (sigmoid(-alpha @ x[i]) - (1 - y[i])) * x[i]
    return grad


def ll_hessian(alpha, x, rho):
    h = np.zeros((x.shape[1], x.shape[1]))
    for i in range(len(alpha)):
        sig = sigmoid(alpha @ x[i])
        h -= sig * (1 - sig) * np.outer(x[i], x[i])
    return h + rho * np.eye(h.shape[0])


def newton_method(x, y, lr, rho, num_iter, v=False):
    alpha = 0
    for i in range(num_iter):
        alpha = alpha + lr * (np.linalg.inv(ll_hessian(alpha, x, rho)) @ ll_grad(alpha, x, y))
        if v:
            print(ll(alpha, x, y))
        rho *= 0.99
    return alpha

res = model(X,Y,XT,YT,True)

optimal_alpha = newton_method(XT, YT, 0.01, 1, 20, True)

