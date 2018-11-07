import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pgreedy
import kernels

def grid(num_data_sites):
    data = 10 * np.random.rand(num_data_sites) - 5
    f = np.sin(data)
    X = np.arange(-5,5, .1)
    Y = np.sin(X)
    return data.reshape((-1,1)), f.reshape((-1,1)), X, Y

# Test
if __name__ == "__main__":
    # training parameter
    num_data_sites = 150
    kernel = kernels.gausskernel()
    discrete_omega, f, X, Y = grid(num_data_sites)
    max_iterations = num_data_sites
    p_tolerance = math.pow(10, -7)

    data_dependent = f is not None

    # computing the basis
    # todo: training dict
    selected, surrogate, kernel_coeff, residual_eval, power_eval, rkhs_error = pgreedy.train(discrete_omega, kernel, max_iterations, p_tolerance, f)

    num_iterations = len(selected)
    pmax = np.max(power_eval, axis=0)
    if data_dependent:
         rmax = np.max(np.sum(residual_eval, axis=2), axis=1)
    # 
    # def eval(x):
    #     kernelvector = np.vectorize(lambda y: kernel(discrete_omega[y, :], x))
    #     return np.sum(kernelvector(selected) @ kernel_coeff, axis=0)
    #
    # evalvector = np.vectorize(lambda y: eval[discrete_omega[y,:]])
    # print('eval:', eval(np.arange(num_data_sites)))
    # print('surrogate', surrogate)


    if data_dependent:
        plt.figure()
        plt.title("surrogate model (blue points) and selected points (red)")
        plt.plot(X, Y, '-', discrete_omega, surrogate, 'b.', discrete_omega[selected], surrogate[selected], 'r.')
        plt.figure()
        plt.title("max value of the power function on discrete_omega and \n quadrature of residual at each iteration")
        plt.semilogy(range(num_iterations), pmax, '-') #, np.log(np.absolute(residual_quad)), 'go')
    else:
        plt.figure()
        plt.title("max value of the power function on discrete_omega")
        plt.semilogy(range(num_iterations), pmax, '-')
        plt.figure()
        plt.title("data and selected points (red)")
        plt.plot(discrete_omega, np.ones(len(discrete_omega)), '.', discrete_omega[selected], np.ones(len(discrete_omega[selected])), 'r.')

    plt.show()
