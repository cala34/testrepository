import numpy as np

"""
To interpolate a given model f: Omega -> R^output_dim with the use of a kernel
basis, the p-greedy algorithm chooses kernel translates iteratively and computes
the interpolant in each iteration. Once a tolerance is met or the max number of
iterations is reached, the interpolant is computed and the algorithm stops.

We are assuming:
    - data is an 2d-np.array of shape (num_data, d) containing all
        available data sites in Omega \subset R^d
    - kernel is a strictly positive definite kernel from Omega x Omega to R
    - f is an 2d-np.array of shape (len(data), output_dim) containing the values
        of f evaluated on data
"""

def train(interpolation_data, train_param):
    # load interpolation data
    data = interpolation_data['data']
    num_data = data.shape[0]
    data_dependent = 'f' in interpolation_data
    f_is_rkhs = 'rkhs_norm_2' in interpolation_data
    if data_dependent:
        f = interpolation_data['f']
    if f_is_rkhs:
        norm_squarred = interpolation_data['rkhs_norm_2']

    # load training parameters
    kernel = train_param['kernel']
    if 'max_iterations' in train_param:
        max_iterations = train_param['max_iterations']
    else:
        max_iterations = num_data
    if 'p_tolerance' in train_param:
        p_tol = train_param['p_tolerance']
    else:
        p_tol = 0
    if 'r_tolerance' in train_param:
        r_tol = train_param['r_tolerance']
    else:
        r_tol = 0

    # function returning kernel values of data for given indices
    def kernel_matrix(k,l):
        if data.ndim > 1:
            return kernel(data[k,:], data[l,:])
        else:
            return kernel(data[k], data[l])
    # function returning a submatrix of the kernel matrix on data
    kernel_matrix_vectorized = np.vectorize(kernel_matrix)
    # diagonal of kernel matrix
    kernel_matrix_diagonal = np.array([kernel_matrix(i,i) for i in range(num_data)])

    # initializing needed variables
    # selected indices
    selected = []
    # not selected indices
    notselected = list(range(num_data))
    # a 2d array of the Newton basis evaluated on data
    # axis 0 = data value, axis 1 = iteration
    newton_basis = np.zeros((num_data, max_iterations))
    # the power function evaluated on data at each iteration
    # axis 0 = data value, axis 1 = iteration
    power_fct = np.zeros((num_data, max_iterations))
    # max power function in each iteration
    max_power_fct = np.zeros(max_iterations)
    # number of iterations
    num_iterations = max_iterations
    # variable to print training status
    fifty_iterations = 1

    output_dim = None
    transition_matrix = None
    newton_coeff = None
    residual = None
    max_residual = None
    rkhs_norm_surrogate = None
    surrogate = None
    kernel_coeff = None

    if data_dependent:
        # dimension of output space of f
        output_dim = f.shape[1]
        # basis transition matrix
        transition_matrix = np.zeros((max_iterations, max_iterations))
        # coefficients wrt the Newton basis
        # axis 0 = iteration, axis 1 = component
        newton_coeff = np.zeros((max_iterations, output_dim))
        # residual evaluated on data at each iteration
        # axis 0 = iteration, axis 1 = data value, axis 2 = component
        residual = np.zeros((max_iterations, num_data, output_dim))
        # 2-norm of the max residual in each iteration
        max_residual = np.zeros(max_iterations)
        # 2-norm of the mean residual in each iteration
        mean_residual = np.zeros(max_iterations)
        if f_is_rkhs:
            # interpolation error in rkhs norm ^2 at each iteration
            rkhs_norm_surrogate = np.zeros(max_iterations)

    # preparing output
    results = {}

    # Training
    print("Started training...")
    for k in range(0, max_iterations):
        # print training status
        if k > 50 * fifty_iterations:
            print("Selected more than", 50*fifty_iterations, "points...")
            fifty_iterations += 1

        # point selection in this iteration
        if k > 0:
            selection_index = np.argmax(power_fct[notselected,k-1])
        else:
            selection_index = np.argmax(kernel_matrix_diagonal)
        selected.append(notselected[selection_index])

        # computing the kth basis function
        if k > 0:
            a = newton_basis[notselected, 0:k].T
            b = newton_basis[selected[k], 0:k]
            kernel_values = kernel_matrix_vectorized(notselected, selected[k]).ravel()
            newton_basis[notselected, k] =  kernel_values - b @ a
            newton_basis[notselected, k] /= power_fct[selected[k], k-1]
        else:
            kernel_values = kernel_matrix_vectorized(notselected, selected[k]).ravel()
            power = np.sqrt(kernel_matrix_diagonal[k])
            newton_basis[notselected, k] = kernel_values / power

        # updating the power function
        if k > 0:
            power_squared = power_fct[notselected, k-1]**2
        else:
            power_squared = kernel_matrix_diagonal[notselected]
        basis_squared = newton_basis[notselected, k]**2
        power_fct[notselected, k] = np.sqrt(np.abs(power_squared - basis_squared))
        max_power_fct[k] = np.max(power_fct[:,k])

        if data_dependent:
            # computing the kth coefficient wrt the Newton basis
            if k > 0:
                r = residual[k-1, selected[k], :]
                p = power_fct[selected[k], k-1]
                newton_coeff[k, :] = r / p
            else:
                power = np.sqrt(kernel_matrix(selected[k],selected[k]))
                newton_coeff[k, :] = f[selected[k], :] / power

            # updating the residual
            if k > 0:
                r = residual[k-1, :, :]
            else:
                r = f
            cv = np.outer(newton_basis[:, k], newton_coeff[k, :])
            residual[k, notselected, :] = r[notselected] - cv[notselected]
            mean_residual[k] = np.mean(np.linalg.norm(residual[k,:,:], axis=1))
            max_residual[k] = np.max(np.linalg.norm(residual[k,:,:], axis=1))

            # updating the transition matrix
            t = transition_matrix[0:k, 0:k]
            n = newton_basis[selected[k], 0:k].T
            transition_matrix[0:k, k] = - t @ n
            transition_matrix[k, k] = 1
            if k > 0:
                transition_matrix[:, k] /= power_fct[selected[k], k-1]
            else:
                transition_matrix[:, k] /= kernel_matrix(selected[0], selected[0])

            # compute rkhs norm^2 of surrogate
            if f_is_rkhs:
                rkhs_norm_surrogate[k] = np.sum(newton_coeff[0:k+1, :]**2)

        notselected.pop(selection_index)

        # break if tolerance is met
        if max_power_fct[k] <= p_tol or max_residual[k] <= r_tol:
            # save expansion size
            num_iterations = k+1
            # cutting of not needed space for more iterations
            newton_basis = newton_basis[:, 0:num_iterations]
            power_fct = power_fct[:, 0:num_iterations]
            max_power_fct = max_power_fct[0:num_iterations]
            if data_dependent:
                newton_coeff = newton_coeff[0:num_iterations, :]
                residual = residual[0:num_iterations,:,:]
                mean_residual = mean_residual[0:num_iterations]
                max_residual = max_residual[0:num_iterations]
                transition_matrix = transition_matrix[0:num_iterations, 0:num_iterations]
                if f_is_rkhs:
                    rkhs_norm_surrogate = rkhs_norm_surrogate[0:num_iterations]
            break

    # saving training Results
    results['selected'] = selected
    results['num_iterations'] = num_iterations
    results['max_power_fct'] = max_power_fct
    if data_dependent:
        # resulting approximation of data
        surrogate = newton_basis @ newton_coeff
        results['surrogate'] = surrogate
        # computing the coefficients wrt the kernel basis
        kernel_coeff = transition_matrix @ newton_coeff
        results['kernel_coeff'] = kernel_coeff
        results['residual'] = residual
        results['mean_residual'] = mean_residual
        results['max_residual'] = max_residual
        if f_is_rkhs:
            rkhs_error = norm_squarred - rkhs_norm_surrogate # squarred error
            results['rkhs_error'] = rkhs_error

    # print training results
    print("Completed Training.\n")
    print("Training results:\n")
    print("number of training data sites:", num_data)
    print("number of selected data sites/expansion size:", num_iterations)
    print("max of power function on training data:", max_power_fct[num_iterations-1])
    if data_dependent:
        print("mean residual on training data:", mean_residual[num_iterations-1])
        print("max residual on training data:", max_residual[num_iterations-1])
    if f_is_rkhs:
        print("RKHS error:", rkhs_error[num_iterations-1])

    return results
