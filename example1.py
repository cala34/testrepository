import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import math
import pgreedy
import kernels


# rebuild example from Haasdonk, Santin
def grid(num_translates, kernel):
    # set number of data sites
    num_data = 2000
    # initialize the random number generator with the current system time
    np.random.seed()
    # generate data from [0,1]^2 to interpolate on
    data = 2 * np.random.rand(num_data, 2) - 1
    # generate coefficients and centers for target function
    centers = 2 * np.random.rand(num_translates, 2) - 1
    alpha = 10 * np.random.rand(num_translates) - 5 # coefficients in [-5,5]

    # compute values of target function
    def model(x):
        kernelvector = np.vectorize(lambda i: kernel(x, centers[i,:]))
        K = kernelvector(range(num_translates))
        return np.inner(alpha, K)
    modelvectordata = np.vectorize(lambda i: model(data[i,:]))
    f = modelvectordata(range(num_data))

    # compute rkhs norm of target function
    T = np.zeros((num_translates, num_translates))
    for k in range(num_translates):
        kernelcenters = np.vectorize(lambda i: kernel(centers[k,:], centers[i,:]))
        T[k,:] = kernelcenters(range(num_translates))
    rkhs_norm = alpha @ T @ alpha

    # save interpolation data
    interpolation_data = {'data': data, 'f': f.reshape((-1,1)), 'rkhs_norm_2': rkhs_norm}

    # mesh grid for plotting the target function
    X = np.arange(-1.5,1.5,.1)
    Y = np.arange(-1.5,1.5,.1)
    Z = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            Z[i,j] = model(np.array([X[i], Y[j]]))
    X, Y = np.meshgrid(X, Y)
    plot_f = {"X": X, "Y": Y, "Z": Z.T}

    return interpolation_data, plot_f

def plot_results(interpolation_data, results, plot_f, train_param):
    # load variables
    data = interpolation_data['data']
    num_data = len(data)
    rkhs_norm = interpolation_data['rkhs_norm_2']

    kernel = train_param['kernel']

    num_iterations = results['num_iterations']
    selected = results['selected']
    surrogate = results['surrogate']
    kernel_coeff = results['kernel_coeff']
    pmax = results['max_power_fct']
    rmax = results['max_residual']
    rkhs_error = results['rkhs_error']

    X = plot_f['X']
    Y = plot_f['Y']
    Z = plot_f['Z']


    fig, ax = plt.subplots()
    ax.plot(data[:,0], data[:,1], 'b.', label = 'training data')
    ax.plot(data[selected,0], data[selected,1], 'r.', label = 'selection')
    legend = ax.legend(loc='upper right')
    for label in legend.get_lines():
        label.set_linewidth(1.5)

    # plot max residual on data and its upper bound
    fig, ax = plt.subplots()
    plt.xlabel('iteration')
    ax.semilogy(range(num_iterations), pmax*np.sqrt(rkhs_norm), '-', label = 'upper bound')
    ax.semilogy(range(num_iterations), rmax, '-.', label = 'max residual')
    legend = ax.legend(loc='upper right')
    for label in legend.get_lines():
        label.set_linewidth(1.5)

    fig, ax = plt.subplots()
    plt.xlabel('iteration')
    ax.semilogy(range(num_iterations), rkhs_error, '--', label = 'rkhs error')
    legend = ax.legend(loc='upper right')
    for label in legend.get_lines():
        label.set_linewidth(1.5)

    # # plot surrogate values
    # def surrogatefunction(x):
    #     selection = data[selected,:]
    #     kernelvector = np.vectorize(lambda i: kernel(x, selection[i,:]))
    #     K = kernelvector(range(len(selected)))
    #     return np.inner(kernel_coeff.ravel(), K)
    # X1 = np.arange(-1.5,1.5,.1)
    # Y1 = np.arange(-1.5,1.5,.1)
    # Z1 = np.zeros((len(X1),len(X1)))
    # for i in range(len(X1)):
    #     for j in range(len(X1)):
    #         Z1[i,j] = surrogatefunction(np.array([X1[i], Y1[j]]))
    # X1, Y1 = np.meshgrid(X1, Y1)
    # fig = plt.figure()
    # # fig.suptitle("surrogate model")
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(X1, Y1, Z1.T)
    # ax.scatter(data[selected,0], data[selected,1], surrogate[selected, :].ravel(), c='b', marker='o')

    # plot target function
    fig = plt.figure()
    # fig.suptitle("target function")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, color='#052b68')


# Test
if __name__ == "__main__":
    # expansion size of target function
    num_translates = 100

    # load data
    interpolation_data, plot_f = grid(num_translates, kernels.gausskernel())

    # training parameters
    # kernel = kernels.gausskernel()
    kernel = kernels.wendlandkernel(2,2)
    train_param = {}
    train_param['kernel'] = kernel
    train_param['max_iterations'] = 100
    train_param['p_tolerance'] = math.pow(10, -13)
    train_param['r_tolerance'] = math.pow(10, -13)

    # training
    results = pgreedy.train(interpolation_data, train_param)

    # plot training results
    plot_results(interpolation_data, results, plot_f, train_param)

    plt.show()
