import numpy as np
import math
from numbers import Number


def train(data, kernel, f, max_iterations, p_tolerance, norm_squarred=None):

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

	f_is_rkhs = norm_squarred is not None
	num_data_sites = data.shape[0]
	output_dim = f.shape[1]

	# kernel matrix A
	A = np.zeros((num_data_sites, num_data_sites))
	for k in range(num_data_sites):
		A[:, k] = kernel(data, data[k, :])

	selected, basis_eval, power_eval = pointSelection(data, A, max_iterations, p_tolerance)

	num_iterations = len(selected)

	surrogate, kernel_coeff, residual_eval, rkhs_error = evaluation(data, f, A, selected, basis_eval, power_eval, norm_squarred)

	return selected, surrogate, kernel_coeff, residual_eval, power_eval, rkhs_error



def pointSelection(data, A, max_iterations, p_tolerance):

	"""
	pointSelection selects the points independent of their values of f
	according to the p-greedy algorithm
	"""

	# initializing needed variables
	# number of available data sites
	num_data_sites = data.shape[0]
	# selected indices
	selected = []
	# not selected indices
	notselected = list(range(num_data_sites))
	# a 2-d array of the Newton basis evaluated on data, axis 0 = data value, axis 1 = iteration
	basis_eval = np.zeros((num_data_sites, max_iterations))
	# the power function evaluated on data at each iteration, axis 0 = data value, axis 1 = iteration
	power_eval = np.empty((num_data_sites, max_iterations))
	power_eval.fill(np.nan)
	# number of iterations
	num_iterations = max_iterations

	for k in range(max_iterations):
		# point selection for this iteration
		if k > 0:
			selection_index = np.argmax(power_eval[notselected,k-1])
		else:
			selection_index = np.argmax(A.diagonal())
		selected.append(notselected[selection_index])

		# computing the kth basis function
		if k > 0:
			a = basis_eval[notselected, 0:k].T
			b = basis_eval[selected[k], 0:k]
			basis_eval[notselected, k] = A[notselected, selected[k]].ravel() - b @ a
			basis_eval[notselected, k] /= power_eval[selected[k], k-1]
		else:
			basis_eval[notselected, k] = A[notselected, selected[k]].ravel()
			basis_eval[notselected, k] /= A[selected[k],selected[k]]

		#updating the power function
		if k > 0:
			power_squared = power_eval[:, k-1]**2
		else:
			power_squared = A.diagonal()
		basis_squared = basis_eval[:, k]**2
		power_eval[:, k] = np.sqrt(np.abs(power_squared - basis_squared))

		notselected.pop(selection_index)

		# break if tolerance is met
		if np.max(power_eval[:,k]) <= p_tolerance:
			num_iterations = k+1
			# cutting of not needed space for more iterations
			basis_eval = basis_eval[:, 0:num_iterations]
			power_eval = power_eval[:, 0:num_iterations]
			break

	return selected, basis_eval, power_eval


def evaluation(data, f, A, selected, basis_eval, power_eval, norm_squarred):

	"""
	computes the surrogate model and its residuals for a given model f
	"""

	#initializing needed variables
	# number of available data sites
	num_data_sites = data.shape[0]
	# output dimension of given model f
	output_dim = f.shape[1]
	# expansion size
	num_iterations = len(selected)
	# indicates whether f is element of the kernel's underlying RKHS
	f_is_rkhs = norm_squarred is not None
	# the coefficients wrt the Newton basis
	newton_coeff = np.zeros((num_iterations, output_dim))
	# the residual evaluated on data at each iteration, axis 0 = iteration, axis 1 = data value, axis 2 = component
	residual_eval = np.zeros((num_iterations, num_data_sites, output_dim))
	# the basis transition matrix
	change_of_basis = np.zeros((num_iterations, num_iterations))
	# # an array containing the quadrature of f-surrogate at each iteration
	# residual_quad = np.array([])
	# an array storing the interpolation error in rkhs norm at each iteration
	if f_is_rkhs:
		rkhs_error = np.zeros(num_iterations)
	else:
		rkhs_error = None

	for k in range(num_iterations):
		# computing the kth coefficient wrt the Newton basis
		if k > 0:
			newton_coeff[k,:] = residual_eval[k-1, selected[k], :] / power_eval[selected[k], k-1]
		else:
			newton_coeff[k,:] = f[selected[k], :] / A[selected[k],selected[k]]

		# computing the residual r_k
		if k > 0:
			residual_eval[k,:,:] = residual_eval[k-1,:,:] - np.outer(basis_eval[:,k], newton_coeff[k,:])
		else:
			residual_eval[k,:,:] = f - np.outer(basis_eval[:,k], newton_coeff[k,:])

		# updating the basis transition matrix
		change_of_basis[0:k, k] = - change_of_basis[0:k, 0:k] @ basis_eval[selected[k], 0:k].T
		change_of_basis[k, k] = 1
		if k > 0:
			change_of_basis[:, k] /= np.copy(power_eval[selected[k], k-1]) # no need for copy?
		else:
			change_of_basis[:, k] /= A[selected[0], selected[0]]

		# computing norm(f-f_k)
		if f_is_rkhs:
			kernel_coeff = (change_of_basis[0:k+1, 0:k+1] @ newton_coeff[0:k+1, :]).reshape((-1, output_dim))
			sum = 0
			for i in range(output_dim):
				sum += np.inner(kernel_coeff[:, i], f[selected[0:k+1], i] - residual_eval[k, selected[0:k+1], i])
			rkhs_error[k] = np.sqrt(np.absolute(norm_squarred - sum))

	# resulting approximation of data
	surrogate = basis_eval @ newton_coeff
	# computing the coefficients wrt the kernel basis
	kernel_coeff = change_of_basis @ newton_coeff

	return surrogate, kernel_coeff, residual_eval, rkhs_error #residual_quad
