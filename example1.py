import numpy as np
import matplotlib.pyplot as plt
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
    # generate data to interpolate on
    data = 2 * np.random.rand(num_data, 2) - 1
    # generate coefficients and centers for target function
    alpha = 30 * np.random.rand(num_translates) - 15
    centers = 2 * np.random.rand(num_translates, 2) - 1

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
    interpolation_data = {'data': data, 'f': f.reshape((-1,1)), 'rkhs_norm_f_2': rkhs_norm}

    # mesh grid for plotting the target function
    X = np.arange(-1,1,.1)
    Y = np.arange(-1,1,.1)
    Z = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            Z[i,j] = model(np.array([X[i], Y[j]]))
    X, Y = np.meshgrid(X, Y)
    plot_f = {"X": X, "Y": Y, "Z": Z.T}

    return interpolation_data, plot_f

def plot_results(interpolation_data, results, plot_f):
    # load variables
    data = interpolation_data['data']
    num_data = len(data)
    rkhs_norm = interpolation_data['rkhs_norm_f_2']

    num_iterations = results['num_iterations']
    selected = results['selected']
    surrogate = results['surrogate']
    pmax = results['max_power_fct']
    rmax = results['max_residual']
    rkhs_error = results['rkhs_error']

    X = plot_f['X']
    Y = plot_f['Y']
    Z = plot_f['Z']

    # plot max power function of each iteration
    fig = plt.figure()
    plt.title("sample points (selected red)")
    plt.plot(data[:,0], data[:,1], 'b.', data[selected,0], data[selected,1], 'r.')
    fig = plt.figure()
    plt.title("max power function (-)")
    plt.semilogy(range(0, num_iterations), pmax, '-',)

    # plot max residual on data and its upper bound
    fig = plt.figure()
    plt.title("norm(f) * max power function (-) (upper bound), \n max residual on training data (-.)")
    plt.semilogy(range(0, num_iterations), pmax*np.sqrt(rkhs_norm), '-', range(0, num_iterations), rmax, '-.')

    # plot rkhs error at each iteration
    fig = plt.figure()
    plt.title("rkhs error on omega at each iteration ")
    plt.semilogy(range(0, num_iterations), rkhs_error, '--')

    # plot surrogate values
    fig = plt.figure()
    fig.suptitle("surrogate model and selected values (blue)")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], surrogate.ravel(), c='g', marker='o')
    ax.scatter(data[selected,0], data[selected,1], surrogate[selected, :].ravel(), c='b', marker='o')

    # plot target function
    fig = plt.figure()
    fig.suptitle("target function")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)

    plt.show()


# Test
if __name__ == "__main__":
    # training parameters
    kernel = kernels.gausskernel(1/2)
    num_translates = 100 # number of kernel translates for target function
    train_param = {}
    train_param['kernel'] = kernel
    train_param['max_iterations'] = 170
    train_param['p_tolerance'] = math.pow(10, -7)
    train_param['r_tolerance'] = math.pow(10, -7)

    # load data
    interpolation_data, plot_f = grid(num_translates, kernel)

    # training
    results = pgreedy.train(interpolation_data, train_param)

    # plot training results
    plot_results(interpolation_data, results, plot_f)
