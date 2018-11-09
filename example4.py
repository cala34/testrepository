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

    test = np.zeros((len(data_test), 10))
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
    rmax = results['max_residual']

    fig = plt.figure()
    plt.title("max power function (-), \n max residual on omega (-.)")
    plt.semilogy(range(num_iterations), pmax, '-', range(num_iterations), rmax, '-.')
    plt.show()

    return

# Test
if __name__ == "__main__":

    kernel = kernels.gausskernel(1/100)
    num_data_train = 10000
    num_data_test = 1000

    # training parameters
    train_param = {}
    train_param['kernel'] = kernel
    train_param['max_iterations'] = 500
    train_param['p_tolerance'] = math.pow(10, -2)
    train_param['r_tolerance'] = math.pow(10, -2)

    # load data
    output = np.load("output.npy")
    data_train = output[0][:num_data_train,:]
    data_test = output[1][:num_data_test,:]
    f_train = output[2][:num_data_train,:]
    f_test = output[3][:num_data_test,:]
    labels_train = output[4][:num_data_train]
    labels_test = output[5][:num_data_test]

    # similarity = np.zeros((num_data_train,num_data_train))
    # for i in range(num_data_train):
    #     x = np.vectorize(lambda j: np.linalg.norm(data_train[i,:] - data_train[j,:]))
    #     similarity[i,:] = x(range(num_data_train))
    # print(similarity)
    # print(similarity.shape)

    # interpolation data
    interpolation_data = {}
    interpolation_data['data'] = data_train
    interpolation_data['f'] = f_train

    # training
    results = pgreedy.train(interpolation_data, train_param)

    print_accuracy(data_train, data_test, labels_train, labels_test, results)

    plot_results(interpolation_data, results)
