import numpy as np
import math
import kernels
import pgreedy
import matplotlib.pyplot as plt

# Test
if __name__ == "__main__":
    # training parameter
    kernel = kernels.gausskernel()
    output = np.load("output.npy")

    num_data_sites_train = 2000
    num_data_sites_test = 1000
    max_iterations = 20
    p_tolerance = math.pow(10, -2)

    discrete_omega_train = output[0][:num_data_sites_train,:]
    discrete_omega_test = output[1][:num_data_sites_test,:]
    f_train = output[2][:num_data_sites_train,:]
    f_test = output[3][:num_data_sites_test,:]
    labels_train = output[4][:num_data_sites_train]
    labels_test = output[5][:num_data_sites_test]

    # computing the basis
    # todo: training dict
    selected, surrogate, kernel_coeff, residual_eval, power_eval, rkhs_error = pgreedy.train(discrete_omega_train, kernel, max_iterations, p_tolerance, f_train)

    num_iterations = len(selected)
    pmax = np.max(power_eval, axis=0)
    rmax = np.max(np.sum(residual_eval, axis=2), axis=1)

    def eval(x):
        kernelvector = np.vectorize(lambda y: kernel(discrete_omega_train[y, :], x))
        return np.sum(kernelvector(selected) @ kernel_coeff, axis=0)

    test = np.zeros((len(discrete_omega_test), 10))
    for i in range(len(discrete_omega_test)):
        test[i,:] = eval(discrete_omega_test[i,:])

    # compute the predicted class
    predictions_train = np.argmax(surrogate, axis=1)
    predictions_test = np.argmax(test, axis=1)
    # compute number of correct predictions / total number of predictions
    accuracy_train = np.sum(predictions_train == labels_train) / len(predictions_train)
    accuracy_test = np.sum(predictions_test == labels_test) / len(predictions_test)
    print("train accuracy: ", accuracy_train)
    print("test accuracy: ", accuracy_test)

    print("test: ", (pmax == np.ones(len(pmax))).all())
    fig = plt.figure()
    plt.title("max power function (-), \n max residual on omega (-.)")
    plt.semilogy(range(0, num_iterations), pmax, '-', range(0, num_iterations), rmax, '-.')
    plt.show()
