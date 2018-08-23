import numpy as np
import random
import math
from numbers import Number
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import examples


def train(discrete_omega, kernel, f, max_iterations, p_tolerance, norm_squarred=None):
	""" For a surrogate model sf for model f: Omega \subset R^d -> R^q,
		the coefficients of the expansion wrt a kernel basis is computed.
		Assuming:
			- discrete_omega is an 2d-np.array of numbers in Omega (R^d) of shape (length, d)
			- kernel is a function from Omega x Omega to real numbers
			- f is an np.array of shape (len(discrete_omega), q),
				where q is the output dimension, containing the function
				evaluations of discrete_omega
	"""

	f_is_rkhs = norm_squarred is not None

	domega_length = discrete_omega.shape[0]
	q = f.shape[1]

	# kernel matrix A
	A = np.zeros((domega_length, domega_length))
	for k in range(domega_length):
		A[:, k] = kernel(discrete_omega, discrete_omega[k, :])

	# initializing needed variables

	# selected indices
	selected = []
	# not selected indices
	notselected = list(range(domega_length))
	# a 2-d array of the Newton basis evaluated on discrete_omega
	basis_eval = np.zeros((domega_length, max_iterations))
	# an array of the coefficients wrt the Newton basis
	coeff = np.zeros((max_iterations, q))
	# the residual evaluated on discrete_omega at each iteration
	# shape (max_iterations, domega_length, q)
	# axis 0 = iteration, axis 1 = data value, axis 2 = component
	residual_eval = np.zeros((max_iterations, domega_length, q))
	# the power function evaluated on discrete_omega at each iteration
	# axis 0 = data value, axis 1 = iteration
	power_eval = np.empty((domega_length, max_iterations))
	power_eval.fill(np.nan)
	# the basis transition matrix
	change_of_basis = np.zeros((max_iterations, max_iterations))
	# number of iterations
	num_iterations = max_iterations
	# an array containing the quadrature of f-surrogate at each iteration
	residual_quad = np.array([])
	# an array storing the interpolation error in rkhs norm at each iteration
	rkhs_error = np.zeros(max_iterations)

	for k in range(0, max_iterations):
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

		# computing the kth coefficient wrt the Newton basis
		if k > 0:
			coeff[k, :] = residual_eval[k-1, selected[k], :] / power_eval[selected[k], k-1]
		else:
			coeff[k, :] = f[selected[k], :] / A[selected[k],selected[k]]

		# updating the residual
		if k > 0:
			residual_eval[k, :, :] = residual_eval[k-1, :, :] - np.outer(basis_eval[:, k], coeff[k, :])
		else:
			residual_eval[k, :, :] = f - np.outer(basis_eval[:, k], coeff[k, :])

		# updating the basis transition matrix
		change_of_basis[0:k, k] = - change_of_basis[0:k, 0:k] @ basis_eval[selected[k], 0:k].T
		change_of_basis[k, k] = 1
		if k > 0:
			change_of_basis[:, k] /= np.copy(power_eval[selected[k], k-1]) # no need for copy?
		else:
			change_of_basis[:, k] /= A[selected[0], selected[0]]

		#updating the power function
		if k > 0:
			power_squared = power_eval[notselected, k-1]**2
		else:
			power_squared = A.diagonal()
		basis_squared = basis_eval[notselected, k]**2
		power_eval[notselected, k] = np.sqrt(np.abs(power_squared - basis_squared))
		if k > 0:
			power_eval[selected[0:k], k] = power_eval[selected[0:k], k-1]

		if f_is_rkhs:
			kernel_coeff = (change_of_basis[0:k+1, 0:k+1] @ coeff[0:k+1, :]).reshape((-1, q))
			sum = 0
			for i in range(q):
				sum += np.inner(kernel_coeff[:, i], f[selected, i] - residual_eval[k, selected, i])
			rkhs_error[k] = np.sqrt(np.absolute(norm_squarred - sum))

		notselected.pop(selection_index)

		# break if tolerance is met
		if np.max(power_eval[:,k]) <= p_tolerance:
			num_iterations = k+1
			# cutting of not needed space for more iterations
			basis_eval = basis_eval[:, 0:num_iterations]
			coeff = coeff[0:num_iterations, :]
			residual_eval = residual_eval[0:num_iterations,:,:]
			power_eval = power_eval[:, 0:num_iterations]
			change_of_basis = change_of_basis[0:num_iterations, 0:num_iterations]
			if f_is_rkhs:
				rkhs_error = rkhs_error[0:num_iterations]
			break

	# resulting approximation of discrete_omega
	surrogate = basis_eval @ coeff
	# computing the coefficients wrt the kernel basis
	kernel_coeff = change_of_basis @ coeff

	if f_is_rkhs:
		return [selected, surrogate, kernel_coeff, residual_eval, power_eval, residual_quad,num_iterations, rkhs_error]
	else:
		return [selected, surrogate, kernel_coeff, residual_eval, power_eval, residual_quad,num_iterations]
