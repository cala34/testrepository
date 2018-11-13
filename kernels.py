import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def gausskernel(shape = 1):
    # return lambda x,y : np.exp(-shape**2 * np.linalg.norm(x-y, axis = 1)**2)
    return lambda x,y : np.exp(-shape**2 * np.linalg.norm(x-y)**2)

def wendlandkernel(d,k):
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

    return lambda x,y: p(np.linalg.norm(x-y))


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    x = 1
    X = np.linspace(-2,4,100)
    kernelvectorize1 = np.vectorize(lambda y: gausskernel()(x,y))
    kernelvectorize2 = np.vectorize(lambda y: gausskernel(1/2)(x,y))
    kernelvectorize3 = np.vectorize(lambda y: gausskernel(2)(x,y))
    Y1 = kernelvectorize1(X)
    Y2 = kernelvectorize2(X)
    Y3 = kernelvectorize3(X)

    plt.figure()
    plt.plot(X, Y1, X, Y2, X, Y3)
    plt.show()
    #
    # import pdb; pdb.set_trace()
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
    plt.show()

    x = 1
    X = np.linspace(-0.5,2.5,100)
    kernelvectorize = np.vectorize(lambda y: wendlandkernel(1,0)(x,y))
    Y1 = kernelvectorize(X)
    kernelvectorize = np.vectorize(lambda y: wendlandkernel(1,1)(x,y))
    Y2 = kernelvectorize(X)
    kernelvectorize = np.vectorize(lambda y: wendlandkernel(1,2)(x,y))
    Y3 = kernelvectorize(X)

    plt.figure()
    plt.plot(X, Y1, X, Y2, X, Y3)
    plt.show()

    # import pdb; pdb.set_trace()
    n = 50
    x = np.array([1,1])
    X1 = X = np.linspace(0,2,n)
    X2 = X = np.linspace(0,2,n)
    X1, X2 = np.meshgrid(X1, X2)
    X = np.array([X1.ravel(), X2.ravel()]).T
    kernelvectorize = np.vectorize(lambda i: wendlandkernel(2,1)(x,X[i,:]))
    Z = kernelvectorize(range(len(X))).reshape(n,n)

    fig = plt.figure()
    plt.title("kernel translate")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z)
    plt.show()
