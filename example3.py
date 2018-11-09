import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pgreedy
import kernels

def grid(num_data):
    interpolation_data = {}
    np.random.seed()
    data = 10 * np.random.rand(num_data, 2) - 5
    f = np.sin(np.linalg.norm(data, axis=1))
    interpolation_data['data'] = data
    interpolation_data['f'] = f.reshape((-1,1))

    X = np.arange(-5, 5, 0.5)
    Y = np.arange(-5, 5, 0.5)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    plot_f = {}
    plot_f['X'] = X
    plot_f['Y'] = Y
    plot_f['Z'] = Z

    return interpolation_data, plot_f


def plot_results(interpolation_data, results, plot_f):
    # load variables
    data = interpolation_data['data']
    num_iterations = results['num_iterations']
    selected = results['selected']
    surrogate = results['surrogate']
    pmax = results['max_power_fct']
    rmax = results['max_residual']

    fig = plt.figure()
    plt.title("max power function (-), \n max residual on data (-.)")
    plt.semilogy(range(0, num_iterations), pmax, '-', range(0, num_iterations), rmax, '-.')

    fig = plt.figure()
    fig.suptitle("surrogate model and selected values (blue)")
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], surrogate.ravel(), c='g', marker='o')
    ax.scatter(data[selected,0], data[selected,1], surrogate[selected, :].ravel(), c='b', marker='o')

    fig = plt.figure()
    fig.suptitle("target function")
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(plot_f['X'], plot_f['Y'], plot_f['Z'])

    fig = plt.figure()
    plt.title("sample points (selected red)")
    plt.plot(data[:,0], data[:,1], 'b.', data[selected,0], data[selected,1], 'r.')

    plt.show()


# Test
if __name__ == "__main__":
    # training parameters
    kernel = kernels.gausskernel(1/2)
    num_data = 2000
    train_param = {}
    train_param['kernel'] = kernel
    train_param['max_iterations'] = num_data
    train_param['p_tolerance'] = math.pow(10, -7)
    train_param['r_tolerance'] = math.pow(10, -7)

    # load data
    interpolation_data, plot_f = grid(num_data)

    # training
    results = pgreedy.train(interpolation_data, train_param)

    # plot training results
    plot_results(interpolation_data, results, plot_f)
