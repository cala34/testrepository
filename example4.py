import numpy as np
import math
import kernels
import pgreedy
import matplotlib.pyplot as plt

def print_accuracy(data_train, data_test, labels_train, labels_test, results):

    num_iterations = results['num_iterations']

    def eval(x):
        selected = results['selected']
        kernel_coeff = results['kernel_coeff']
        kernelvector = np.vectorize(lambda i: kernel(x, data_train[i, :]))
        return kernelvector(selected) @ kernel_coeff

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

    fig = plt.figure()
    plt.title("max power function (-), \n max residual (:) \n mean residual (-.)")
    plt.semilogy(range(num_iterations), pmax, '-', range(num_iterations), rmean, '-.', range(num_iterations), rmax, ':')
    plt.show()

    return


if __name__ == "__main__":

    kernel = kernels.gausskernel(1/100)
    num_data_train = 2000
    num_data_test = 100

    # training parameters
    train_param = {}
    train_param['kernel'] = kernel
    train_param['max_iterations'] = 1000
    train_param['p_tolerance'] = math.pow(10, -7)
    train_param['r_tolerance'] = math.pow(10, -7)

    # load data
    output = np.load("output.npy")
    data_train = output[0][:num_data_train,:]
    data_test = output[1][:num_data_test,:]
    f_train = output[2][:num_data_train,:]
    f_test = output[3][:num_data_test,:]
    labels_train = output[4][:num_data_train]
    labels_test = output[5][:num_data_test]

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

    print("Ziffern mit Fehler > .8:", np.sum(f[np.linalg.norm(residual[k,:,:],axis=1) > .8], axis=0))

    print_accuracy(data_train, data_test, labels_train, labels_test, results)

    plot_results(interpolation_data, results)
