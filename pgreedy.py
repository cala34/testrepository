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
	residual_eval = np.copy(f)
	# the power function evaluated on discrete_omega at each iteration
	power_eval = A[0,0] * np.ones(domega_length, dtype=np.longdouble)
	# the basis transition matrix
	change_of_basis = np.zeros((max_iterations, max_iterations))
	# an array storing the maximum power function at each iteration
	pmax = np.zeros(max_iterations)
	# an array storing the max residual at each iteration
	rmax = np.zeros(max_iterations)
	# number of iterations
	num_iterations = max_iterations
	# an array containing the quadrature of f-surrogate at each iteration
	residual_quad = np.array([])
	# an array storing the interpolation error in rkhs norm at each iteration
	rkhs_error = np.zeros(max_iterations)
	v = np.zeros(max_iterations)

	for k in range(0, max_iterations):
		# point selection for this iteration
		i = np.argmax(power_eval[notselected])
		selected.append(notselected[i])
		pmax[k] = np.max(power_eval[notselected])

		# computing the kth basis function
		if k > 0:
			a = basis_eval[notselected, 0:k].T
			b = basis_eval[selected[k], 0:k]
			basis_eval[notselected, k] = A[notselected, selected[k]].ravel() - b @ a
		else:
			basis_eval[notselected, k] = A[notselected, selected[k]].ravel()
		basis_eval[notselected, k] /= power_eval[selected[k]]

		# computing the kth coefficient wrt the Newton basis
		coeff[k, :] = residual_eval[selected[k], :] / power_eval[selected[k]]

		# updating the residuals
		residual_eval = residual_eval - np.outer(basis_eval[:, k], coeff[k, :])
		rmax[k] = np.max(np.sum(np.absolute(residual_eval[notselected,:]), axis=1))
		v[k] = np.max(np.sum(np.absolute(residual_eval[selected,:]), axis=1))

		# updating the basis transition matrix
		change_of_basis[0:k, k] = - change_of_basis[0:k, 0:k] @ basis_eval[selected[k], 0:k].T
		change_of_basis[k, k] = 1
		change_of_basis[:, k] /= np.copy(power_eval[selected[k]])

		#updating the power function
		power_squared = power_eval[notselected]**2
		basis_squared = basis_eval[notselected, k]**2
		power_eval[notselected] = np.sqrt(np.abs(power_squared - basis_squared))

		if f_is_rkhs:
			kernel_coeff = (change_of_basis[0:k+1, 0:k+1] @ coeff[0:k+1, :]).reshape((-1, q))
			sum = 0
			for i in range(q):
				sum += np.inner(kernel_coeff[:, i], f[selected, i] - residual_eval[selected, i])
			rkhs_error[k] = np.sqrt(np.absolute(norm_squarred - sum))

		del notselected[i]

		# break if tolerance is met
		if pmax[k] <= p_tolerance:
			num_iterations = k+1
			# cutting of not needed space for more iterations
			basis_eval = basis_eval[:, 0:num_iterations]
			coeff = coeff[0:num_iterations, :]
			change_of_basis = change_of_basis[0:num_iterations, 0:num_iterations]
			pmax = pmax[0:num_iterations]
			rmax = rmax[0:num_iterations]
			v = v[0:num_iterations]
			if f_is_rkhs:
				rkhs_error = rkhs_error[0:num_iterations]
			break

	# resulting approximation of discrete_omega
	surrogate = basis_eval @ coeff
	# computing the coefficients wrt the kernel basis
	kernel_coeff = change_of_basis @ coeff

	if f_is_rkhs:
		return [selected, surrogate, kernel_coeff, residual_eval, power_eval, pmax, residual_quad,num_iterations, rkhs_error, rmax,v]
	else:
		return [selected, surrogate, kernel_coeff, residual_eval, power_eval, pmax, residual_quad,num_iterations, rmax]
