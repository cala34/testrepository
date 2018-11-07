import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pgreedy
import kernels

def grid():
    # discrete_omega = np.random.rand(20, 2)
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    # discrete_omega is of shape (len(X)**2, 2)
    print("number of data sites: ", len(X)**2)
    discrete_omega = np.zeros((len(X)**2, 2))
    for i in range(len(X)):
    	for j in range(len(Y)):
            discrete_omega[i*len(Y)+j, 0] = X[i]
            discrete_omega[i*len(Y)+j, 1] = Y[j]
    X, Y = np.meshgrid(X, Y)
    # test function from R^2 to R: sin(2norm(x,y))
    R = np.sqrt(X**2 + Y**2)
    # in mesh grid form
    Z = np.sin(R)

    # in shape (len(X)**2, 1)
    f = np.array([])
    for i in range(len(X)):
    	for j in range(len(X)):
    		f = np.append(f, Z[i,j])
    f = f.reshape((-1,1))

    return discrete_omega, f, X, Y, Z

# Test
if __name__ == "__main__":
    # training parameter
    kernel = kernels.gausskernel()
    discrete_omega, f, X, Y, Z = grid()
    max_iterations = len(discrete_omega)
    p_tolerance = math.pow(10, -7)

    data_dependent = f is not None

    # computing the basis
    # todo: training dict
    selected, surrogate, kernel_coeff, residual_eval, power_eval, rkhs_error = pgreedy.train(discrete_omega, kernel, max_iterations, p_tolerance, f)


    num_iterations = len(selected)
    pmax = np.max(power_eval, axis=0)
    if data_dependent:
         rmax = np.max(np.sum(residual_eval, axis=2), axis=1)

    # print training results
    print("Training Results:")
    print("number of iterations: ", num_iterations)

    if data_dependent:
        fig = plt.figure()
        plt.title("max power function (-), \n max residual on omega (-.)")
        plt.semilogy(range(0, num_iterations), pmax, '-', range(0, num_iterations), rmax, '-.')
        fig = plt.figure()
        fig.suptitle("surrogate model and selected values (blue)")
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(discrete_omega[:,0], discrete_omega[:,1], surrogate.ravel(), c='g', marker='o')
        ax.scatter(discrete_omega[selected,0], discrete_omega[selected,1], surrogate[selected, :].ravel(), c='b', marker='o')
        fig = plt.figure()
        fig.suptitle("original model")
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z)

    fig = plt.figure()
    plt.title("sample points (selected red)")
    plt.plot(discrete_omega[:,0], discrete_omega[:,1], 'b.', discrete_omega[selected,0], discrete_omega[selected,1], 'r.')
    fig = plt.figure()
    plt.title("max power function (-)")
    plt.semilogy(range(0, num_iterations), pmax, '-')


    plt.show()
