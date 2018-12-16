import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

'''
Implementation of Gaussian and Wendland Kernels

If this script is exectued, it produces example plots of kernels.
'''

def gausskernel(shape = 1):
    return lambda x,y : np.exp(-shape * np.linalg.norm(x-y)**2)

def wendlandkernel(d,k, shape = 1):
    if d == 1:
        if k == 0:
            p = lambda r: np.maximum(1-r,0)
        elif k == 1:
            p = lambda r: np.maximum((1-r), 0)**3 * (3*r + 1)
        elif k == 2:
            p = lambda r: np.maximum((1-r), 0)**5 * (8*r**2 + 5*r + 1)
    elif d <= 3:
        if k == 0:
            p = lambda r: np.maximum((1-r), 0)**2
        elif k == 1:
            p = lambda r: np.maximum((1-r), 0)**4 * (4*r + 1)
        elif k == 2:
            p = lambda r: np.maximum((1-r), 0)**6 * (35*r**2 + 18*r + 3)
        elif k == 3:
            p = lambda r: np.maximum((1-r), 0)**8 * (32*r**3 + 25*r**2 + 8*r + 1)
    elif d <= 5:
        if k == 0:
            p = lambda r: np.maximum((1-r), 0)**3
        elif k == 1:
            p = lambda r: np.maximum((1-r), 0)**5 * (5*r + 1)
        elif k == 2:
            p = lambda r: np.maximum((1-r), 0)**7 * (16*r**2 + 7*r + 1)

    return lambda x,y: p(shape * np.linalg.norm(x-y))


if __name__ == "__main__":
    x = 1
    X = np.linspace(-2,4,100)
    kernelvectorize1 = np.vectorize(lambda y: gausskernel()(x,y))
    kernelvectorize2 = np.vectorize(lambda y: gausskernel(1/2)(x,y))
    kernelvectorize3 = np.vectorize(lambda y: gausskernel(2)(x,y))
    Y1 = kernelvectorize1(X)
    Y2 = kernelvectorize2(X)
    Y3 = kernelvectorize3(X)

    fig, ax = plt.subplots()
    ax.plot(X, Y2, label = 'a = 0.5')
    ax.plot(X, Y1, label = 'a = 1')
    ax.plot(X, Y3, label = 'a = 2')
    legend = ax.legend(loc='upper right')
    for label in legend.get_lines():
        label.set_linewidth(1.5)
    plt.title("Gaussian Kernels at x = 1")
    plt.show()

    n = 50
    x = np.array([1,1])
    X1 = X = np.linspace(-1.5,3.5,n)
    X2 = X = np.linspace(-1.5,3.5,n)
    X1, X2 = np.meshgrid(X1, X2)
    X = np.array([X1.ravel(), X2.ravel()]).T
    kernelvectorize = np.vectorize(lambda i: gausskernel()(x,X[i,:]))
    Z = kernelvectorize(range(len(X))).reshape(n,n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z)
    plt.title("Gaussian Kernel at x = (1,1)")
    plt.show()

    x = 1
    X = np.linspace(-0.5,2.5,100)
    kernelvectorize = np.vectorize(lambda y: wendlandkernel(1,0)(x,y))
    Y1 = kernelvectorize(X)
    kernelvectorize = np.vectorize(lambda y: wendlandkernel(1,1)(x,y))
    Y2 = kernelvectorize(X)
    kernelvectorize = np.vectorize(lambda y: wendlandkernel(1,2)(x,y))
    Y3 = kernelvectorize(X)

    fig, ax = plt.subplots()
    ax.plot(X, Y1, label = 'k = 0')
    ax.plot(X, Y2, label = 'k = 1')
    ax.plot(X, Y3, label = 'k = 2')
    legend = ax.legend(loc='upper right')
    for label in legend.get_lines():
        label.set_linewidth(1.5)
    plt.title("Wendland Kernels for d = 1 and k = 0, 1, 2 at x = 1")
    plt.show()

    n = 50
    x = np.array([1,1])
    X1 = X = np.linspace(0,2,n)
    X2 = X = np.linspace(0,2,n)
    X1, X2 = np.meshgrid(X1, X2)
    X = np.array([X1.ravel(), X2.ravel()]).T
    kernelvectorize = np.vectorize(lambda i: wendlandkernel(2,1)(x,X[i,:]))
    Z = kernelvectorize(range(len(X))).reshape(n,n)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z)
    plt.title("Wendland Kernel for d = 2 and k = 1 at x = (1,1)")
    plt.show()
