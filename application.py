import numpy as np
import math
import kernels
import pgreedy
import matplotlib.pyplot as plt

'''
This script loads data from output.npy and performs kernel interpolation
on the entire set of all feature vectors

(trains ~1,5 hours)
'''

def print_accuracy(data_train, data_test, labels_train, labels_test, results):

    num_iterations = results['num_iterations']

    # function that computes the approximation value at x
    def eval(x):
        selected = results['selected']
        kernel_coeff = results['kernel_coeff']
        kernelvector = np.vectorize(lambda i: kernel(x, data_train[i, :]))
        return kernelvector(selected) @ kernel_coeff

    # array of approximation values for test data
    test = np.zeros((len(data_test),10))
    for i in range(len(data_test)):
        test[i,:] = eval(data_test[i,:])

    # compute the predicted class
    surrogate = results['surrogate']
    predictions_train = np.argmax(surrogate, axis=1)
    predictions_test = np.argmax(test, axis=1)
    # compute number of correct predictions / total number of predictions
    accuracy_train = np.sum(predictions_train == labels_train) / len(predictions_train)
    accuracy_test = np.sum(predictions_test == labels_test) / len(predictions_test)
    print("train accuracy: ", accuracy_train)
    print("test accuracy: ", accuracy_test)


def plot_results(interpolation_data, results):

    num_iterations = results['num_iterations']
    pmax = results['max_power_fct']
    rmean = results['mean_residual']
    rmax = results['max_residual']

    fig, ax = plt.subplots()
    plt.xlabel('iteration')
    ax.semilogy(range(num_iterations), pmax, '-', label = 'max power function')
    ax.semilogy(range(num_iterations), rmax, '-.', label = 'max residual')
    ax.semilogy(range(num_iterations), rmean, ':', label = 'mean residual')
    legend = ax.legend(loc='center right')
    for label in legend.get_lines():
        label.set_linewidth(1.5)

    plt.show()


if __name__ == "__main__":

    kernel = kernels.gausskernel(1/100)
    num_data_train = 60000
    num_data_test = 10000

    # training parameters
    train_param = {}
    train_param['kernel'] = kernel
    train_param['max_iterations'] = 4000
    train_param['p_tolerance'] = math.pow(10, -7)
    train_param['r_tolerance'] = math.pow(10, -7)

    # load data and sample randomly
    output = np.load("output.npy")
    np.random.seed()
    len_train = output[0].shape[0]
    len_test = output[1].shape[0]
    rand_train_indeces = np.random.randint(len_train, size = num_data_train)
    rand_test_indeces = np.random.randint(len_test, size = num_data_test)
    data_train = output[0][rand_train_indeces,:]
    data_test = output[1][rand_test_indeces,:]
    f_train = output[2][rand_train_indeces,:]
    f_test = output[3][rand_test_indeces,:]
    labels_train = output[4][rand_train_indeces]
    labels_test = output[5][rand_test_indeces]

    # interpolation data
    interpolation_data = {}
    interpolation_data['data'] = data_train
    interpolation_data['f'] = f_train

    # training
    results = pgreedy.train(interpolation_data, train_param)

    k = results['num_iterations'] - 1
    residual = results['residual']
    surrogate = results['surrogate']
    f = interpolation_data['f']

    print_accuracy(data_train, data_test, labels_train, labels_test, results)

    plot_results(interpolation_data, results)
