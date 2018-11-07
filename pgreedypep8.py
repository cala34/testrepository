import numpy as np
import math
from numbers import Number

"""
To interpolate a given model f: Omega -> R^output_dim with the use of a kernel
basis, the p-greedy algorithm chooses kernel translates iteratively and computes
the interpolant in each iteration. Once a tolerance is met or the max number of
iterations is reached, the interpolant is found and the algorithm stops.

We are assuming:
	- data is an 2d-np.array of shape (num_data_sites, d) containing all
		available data sites in Omega \subset R^d
	- kernel is a strictly positive definite kernel from Omega x Omega to R
	- f is an 2d-np.array of shape (len(data), output_dim) containing the values
		of f evaluated on data
"""


def train(data,
          kernel,
          max_iterations,
          p_tolerance,
          f=None,
          norm_squarred=None):

    print("Started training...")
    data_dependent = f is not None
    num_data_sites = data.shape[0]

    # kernel matrix A
    A = np.zeros((num_data_sites, num_data_sites))
    for k in range(num_data_sites):
        A[:, k] = kernel(data, data[k, :])

    # initializing needed variables

    # selected indices
    selected = []
    # not selected indices
    notselected = list(range(num_data_sites))
    # a 2-d array of the Newton basis evaluated on data
    # axis 0 = data value, axis 1 = iteration
    basis_eval = np.zeros((num_data_sites, max_iterations))
    # the power function evaluated on data at each iteration
    # axis 0 = data value, axis 1 = iteration
    power_eval = np.empty((num_data_sites, max_iterations))
    power_eval.fill(np.nan)
    # number of iterations
    num_iterations = max_iterations

    output_dim = None
    f_is_rkhs = None
    change_of_basis = None
    newton_coeff = None
    residual_eval = None
    residual_quad = None
    rkhs_error = None
    surrogate = None
    kernel_coeff = None

    if data_dependent:
        output_dim = f.shape[1]
        # indicates whether f is element of the kernel's underlying RKHS
        f_is_rkhs = norm_squarred is not None
        # the basis transition matrix
        change_of_basis = np.zeros((max_iterations, max_iterations))
        # an array of the coefficients wrt the Newton basis
        newton_coeff = np.zeros((max_iterations, output_dim))
        # the residual evaluated on data at each iteration
        # axis 0 = iteration, axis 1 = data value, axis 2 = component
        residual_eval = np.zeros((max_iterations, num_data_sites, output_dim))
        # an array containing the quadrature of f-surrogate at each iteration
        residual_quad = np.array([])
        # an array storing the interpolation error in rkhs norm at each
        # iteration
        if f_is_rkhs:
            rkhs_error = np.zeros(max_iterations)

    fifty_iterations = 1

    for k in range(0, max_iterations):
        # print training status
        if k > 50 * fifty_iterations:
            print("Selected more than", 50 * fifty_iterations, "points...")
            fifty_iterations += 1

        # point selection for this iteration
        if k > 0:
            selection_index = np.argmax(power_eval[notselected, k - 1])
        else:
            selection_index = np.argmax(A.diagonal())
        selected.append(notselected[selection_index])

        # computing the kth basis function
        if k > 0:
            a = basis_eval[notselected, 0:k].T
            b = basis_eval[selected[k], 0:k]
            basis_eval[notselected, k] = A[notselected,
                                           selected[k]].ravel() - b @ a
            basis_eval[notselected, k] /= power_eval[selected[k], k - 1]
        else:
            basis_eval[notselected, k] = A[notselected, selected[k]].ravel()
            basis_eval[notselected, k] /= np.sqrt(A[selected[k], selected[k]])

        # updating the power function
        if k > 0:
            power_squared = power_eval[:, k - 1]**2
        else:
            power_squared = A.diagonal()
        basis_squared = basis_eval[:, k]**2
        power_eval[:, k] = np.sqrt(np.abs(power_squared - basis_squared))

        if data_dependent:
            # computing the kth coefficient wrt the Newton basis
            if k > 0:
                newton_coeff[k, :] = residual_eval[k - 1,
                                                   selected[k], :] / power_eval[selected[k], k - 1]
            else:
                newton_coeff[k, :] = f[selected[k], :] / \
                    A[selected[k], selected[k]]

            # updating the residual
            if k > 0:
                residual_eval[k, :, :] = residual_eval[k - 1, :, :] - \
                    np.outer(basis_eval[:, k], newton_coeff[k, :])
            else:
                residual_eval[k, :, :] = f - \
                    np.outer(basis_eval[:, k], newton_coeff[k, :])

            # updating the basis transition matrix
            change_of_basis[0:k, k] = - change_of_basis[0:k,
                                                        0:k] @ basis_eval[selected[k], 0:k].T
            change_of_basis[k, k] = 1
            if k > 0:
                # no need for copy?
                change_of_basis[:,
                                k] /= np.copy(power_eval[selected[k], k - 1])
            else:
                change_of_basis[:, k] /= A[selected[0], selected[0]]

            # computing norm(f-f_k)
            if f_is_rkhs:
                kernel_coeff = (
                    change_of_basis[0:k + 1, 0:k + 1] @ newton_coeff[0:k + 1, :]).reshape((-1, output_dim))
                sum = 0
                for i in range(output_dim):
                    sum += np.inner(kernel_coeff[:,
                                                 i],
                                    f[selected,
                                      i] - residual_eval[k,
                                                         selected,
                                                         i])
                rkhs_error[k] = np.sqrt(np.absolute(norm_squarred - sum))

        notselected.pop(selection_index)

        # break if tolerance is met
        if np.max(power_eval[:, k]) <= p_tolerance:
            num_iterations = k + 1
            # cutting of not needed space for more iterations
            basis_eval = basis_eval[:, 0:num_iterations]
            power_eval = power_eval[:, 0:num_iterations]
            if data_dependent:
                newton_coeff = newton_coeff[0:num_iterations, :]
                residual_eval = residual_eval[0:num_iterations, :, :]
                change_of_basis = change_of_basis[0:num_iterations,
                                                  0:num_iterations]
                if f_is_rkhs:
                    rkhs_error = rkhs_error[0:num_iterations]
            break

    if data_dependent:
        # resulting approximation of data
        surrogate = basis_eval @ newton_coeff
        # computing the coefficients wrt the kernel basis
        kernel_coeff = change_of_basis @ newton_coeff

    print("Completed Training.\n")
    print("Training results:\n")
    print("number of data sites:", num_data_sites)
    print("number of selected data sites/number of iterations:", num_iterations)
    print("max of power function on training data:",
          np.max(power_eval[:, num_iterations - 1]))
    if data_dependent:
        print("max residual on training data:", np.sum(
            residual_eval[num_iterations - 1, :]))
    if f_is_rkhs:
        print("max RKHS error on training data:",
              rkhs_error[num_iterations - 1])

    return selected, surrogate, kernel_coeff, residual_eval, power_eval, rkhs_error
