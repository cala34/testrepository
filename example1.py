import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pgreedy
import kernels


# rebuild example from Haasdonk, Santin
def grid(num_translates, kernel):
    num_data_sites = 2000
    # initialize the random number generator with the current system time
    np.random.seed()
    data = 2 * np.random.rand(num_data_sites,2) - 1
    alpha = 30 * np.random.rand(num_translates) - 15
    centers = 2 * np.random.rand(num_translates, 2) - 1
    f = np.zeros(num_data_sites)
    for i in range(num_data_sites):
        K = kernel(data[i, :], centers)
        f[i] = np.inner(alpha, K)
    T = np.zeros((num_translates, num_translates))
    for k in range(num_translates):
        T[k,:] = kernel(centers[k,:], centers)
    norm_squarred = alpha @ T @ alpha

    # mesh grid for plotting the model
    X = np.arange(-1,1,.1)
    Y = np.arange(-1,1,.1)
    Z = np.zeros((len(X),len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            Z[i,j] = np.inner(alpha, kernel(np.array([X[i], Y[j]]), centers))
    X, Y = np.meshgrid(X, Y)
    print("norm of f:", np.sqrt(norm_squarred))
    return data, f.reshape((-1,1)), X, Y, Z.T, norm_squarred

# Test
if __name__ == "__main__":
    # training parameter
    kernel = kernels.gausskernel()
    discrete_omega, f, X, Y, Z, norm_squarred = grid(100, kernel)
    max_iterations = 170
    p_tolerance = 0 #math.pow(10, -7)

    f_is_rkhs = norm_squarred is not None
    data_dependent = f is not None

    # computing the basis
    # todo: training dict
    selected, surrogate, kernel_coeff, residual_eval, power_eval, rkhs_error = pgreedy.train(discrete_omega, kernel, max_iterations, p_tolerance, f, norm_squarred)

    num_iterations = len(selected)
    pmax = np.max(power_eval, axis=0)
    if data_dependent:
         rmax = np.max(np.sum(residual_eval, axis=2), axis=1)

    if data_dependent:
        fig = plt.figure()
        if f_is_rkhs:
            plt.title("norm(f) * max power function (-), \n max residual on omega (-.)")
            plt.semilogy(range(0, num_iterations), pmax*np.sqrt(norm_squarred), '-', range(0, num_iterations), rmax, '-.')
            fig = plt.figure()
            plt.title("rkhs error on omega at each iteration ")
            plt.semilogy(range(0, num_iterations), rkhs_error, '--')
        else:
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
    # else:
    fig = plt.figure()
    plt.title("sample points (selected red)")
    plt.plot(discrete_omega[:,0], discrete_omega[:,1], 'b.', discrete_omega[selected,0], discrete_omega[selected,1], 'r.')
    fig = plt.figure()
    plt.title("max power function (-)")
    plt.semilogy(range(0, num_iterations), pmax, '-',)

    # fig = plt.figure()
    # fig.suptitle("residuals")
    # ax = fig.add_subplot(111, projection='3d')
    # x = discrete_omega[:,0]
    # y = discrete_omega[:, 1]
    # ax.scatter(x, y, residual_eval.ravel(), cmap=cm.coolwarm, marker='o')

    plt.show()
