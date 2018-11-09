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
    data = 10 * np.random.rand(num_data) - 5
    interpolation_data['data'] = data
    interpolation_data['f'] = np.sin(data).reshape((-1,1))

    plot_f = {}
    X = np.arange(-5,5, .1)
    plot_f['X'] = X
    plot_f['Y'] = np.sin(X)

    return interpolation_data, plot_f

def plot_results(interpolation_data, results, plot_f):

    data = interpolation_data['data']
    num_iterations = results['num_iterations']
    surrogate = results['surrogate']
    selected = results['selected']
    pmax = results['max_power_fct']
    rmax = results['max_residual']
    X = plot_f['X']
    Y = plot_f['Y']

    plt.figure()
    plt.title("surrogate model (blue points) and selected points (red)")
    plt.plot(X, Y, '-', data, surrogate, 'b.', data[selected], surrogate[selected], 'r.')
    plt.figure()
    plt.title("max power function on data (-) \n and max residual at each iteration (-.)")
    plt.semilogy(range(num_iterations), pmax, '-', range(num_iterations), rmax, '-.')
    plt.show()


# Test
if __name__ == "__main__":
    # training parameter
    num_data = 150
    kernel = kernels.gausskernel()

    train_param = {}
    train_param['kernel'] = kernel
    train_param['max_iterations'] = num_data
    train_param['p_tolerance'] = math.pow(10,-7)

    interpolation_data, plot_f = grid(num_data)

    # training
    results = pgreedy.train(interpolation_data, train_param)

    plot_results(interpolation_data, results, plot_f)
