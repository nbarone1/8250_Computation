# MATH 8250 HW5
import scipy.io as sio
import numpy as np


# Problem 4

# Read the MTX file
X = sio.mmread('X.mtx')

A = X @ X.T

tol = 1e-6

n = A.shape

b = np.random.rand(n)

max_iter = 100

k = 2

def normalize(x):
    """
    The function `normalize` takes an input array `x`, calculates the maximum absolute value in `x`, and
    returns a tuple containing this maximum value and `x` normalized by dividing by the maximum value.
    
    :param x: The parameter `x` is a numpy array that you want to normalize
    :return: The `normalize` function returns two values: `fac`, which is the maximum absolute value of
    the input array `x`, and `x_n`, which is the input array `x` normalized by dividing it by its
    maximum value.
    """
    fac = (x.T @ A @ x) / (x.T @ x) 
    x_n = x / x.max()
    return fac, x_n

# (a) Power Iteration for Eigenvalues

def pow_iter(A,x):
    """
    The function takes in matrix A and potential eigenvector x, calculates the dot product (potential new eigenvector).
    
    :param A: The matrix the power iteration is being performed on.
    :param x: The purposed eigenvector.
    :return: Returns the resulting purposed eigenvalue and eigenvector pair.
    """
    x = np.dot(A,x)
    val, vec = normalize(x)
    return val, vec

def find_ev(A,x,tol,max_iter):
    vec_curr = x
    A = A
    val_curr = 0
    for i in range(max_iter):
        val, vec = pow_iter(A,vec_curr)
        if (val - val_curr) < tol:
            break
        vec_curr = vec
        val_curr = val
    return vec_curr, val_curr


def new_A(A,vec,val):
    A = A - val*(vec @ vec.T)/(np.linalg.norm(vec, ord=1)**2)
    return A

def power_method(A,b,k,tol,max_iter):
    for i in range(k):
        vec, val = find_ev(A,b,tol,max_iter)
        print(vec)
        A = new_A(A,vec,val)


power_method(A,b,k,tol,max_iter)


# (b) Dense Choice

Xdense = X.toarray()

Adense = Xdense @ Xdense.T

power_method(Adense,b,k,tol,max_iter)