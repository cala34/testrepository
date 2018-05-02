import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def pgreedy(discrete_omega, kernel, f, max_iterations, p_tolerance):
	""" For a surrogate model sf for model f: Omega \subset R^d -> R^q,
		the coefficients of the expansion wrt a kernel basis is computed.
		Assuming:
			- discrete_omega is an 2d-np.array of numbers in Omega (R^d) of shape (length, d)
			- kernel is a function from Omega x Omega to real numbers
			- f is an np.array of shape (len(discrete_omega), q),
				where q is the output dimension, containing the function
				evaluations of discrete_omega
	"""

	domega_length = discrete_omega.shape[0]
	q = f.shape[1]

	def A(i, j):
		if type(i) is list or type(i) is np.ndarray:
			rows = np.array([])
			for row_index in i:
				row = np.array([])
				if type(j) is list or type(j) is np.ndarray:
					for column_index in j:
						row = np.append(row, kernel(discrete_omega[row_index], discrete_omega[column_index]))
				else:
					row = np.append(row, kernel(discrete_omega[row_index], discrete_omega[j]))
				rows = np.append(rows, row)
			result = rows.reshape(len(i), -1)
		else:
			result = kernel(discrete_omega[i], discrete_omega[j])
		return result


	# initializing needed variables

	# selected indices
	selected = []
	# not selected indices
	notselected = range(domega_length)
	# a 2-d array of the Newton basis evaluated on discrete_omega
	basis_eval = np.zeros((domega_length, max_iterations))
	# an array of the coefficients wrt the Newton basis
	coeff = np.zeros((max_iterations, q))
	# the residual evaluated on discrete_omega at each iteration
	residual_eval = np.copy(f)
	# the power function evaluated on discrete_omega at each iteration
	power_eval = A(0,0) * np.ones(domega_length)
	# the basis transition matrix
	change_of_basis = np.zeros((max_iterations, max_iterations))
	# an array storing the maximum power function at each iteration
	pmax = np.zeros(max_iterations)
	num_iterations = max_iterations


	for k in range(0, max_iterations):
		# point selection of this iteration
		i = np.argmax(power_eval[notselected])
		selected.append(notselected[i])
		pmax[k] = power_eval[notselected[i]]

		# computing the kth basis function
		if k > 0:
			a = np.transpose(basis_eval[notselected, 0:k]).reshape((k, len(notselected)))
			b = basis_eval[selected[k], 0:k]
			basis_eval[notselected, k] = A(notselected, selected[k]).reshape(len(notselected)) - np.dot(b, a)
			# Problem: 2-d arrays mit shape( _ , 1) werden automatisch als Zeilenvektor gespeichert
		else:
			basis_eval[notselected, k] = A(notselected, selected[k]).reshape(len(notselected))
		basis_eval[notselected, k] /= power_eval[selected[k]]

		# computing the kth coefficient wrt the Newton basis
		coeff[k, :] = residual_eval[selected[k], :] / power_eval[selected[k]]

		# updating the residuals
		x = residual_eval[notselected, :].reshape(len(notselected), q)
		y = basis_eval[notselected, k].reshape(len(notselected), 1)
		z = coeff[k, :].reshape(1,q)
		residual_eval[notselected, :] = np.subtract(x, np.matmul(y, z))

		# updating the basis transition matrix
		change_of_basis[0:k, k] = - np.dot(np.copy(change_of_basis[0:k, 0:k]), np.copy(np.transpose(basis_eval[selected[k], 0:k])))
		change_of_basis[k, k] = 1
		change_of_basis[:, k] /= np.copy(power_eval[selected[k]])

		#updating the power function
		abs_vector = np.vectorize(math.fabs)
		power_squared = np.square(power_eval[notselected])
		basis_squared = np.square(basis_eval[notselected, k])
		power_eval[notselected] = np.sqrt(abs_vector(power_squared - basis_squared))

		del notselected[i]

		# break if tolerance is met
		if pmax[k] < p_tolerance:
			num_iterations = k+1
			# cutting of not needed space for more iterations
			basis_eval = basis_eval[:, 0:num_iterations]
			coeff = coeff[0:num_iterations, :]
			change_of_basis = change_of_basis[0:num_iterations, 0:num_iterations]
			break

	# resulting approximation of discrete_omega
	sf = np.dot(coeff.reshape(1, num_iterations), np.transpose(basis_eval[:, :])).reshape(-1)
	# computing the coefficients wrt the kernel basis
	a = np.dot(change_of_basis, coeff)
	# sfa = np.dot(A(range(0, domega_length), selected), a).reshape(-1)

	# print training results
	print "Training Results:"
	print "number of iterations: ", num_iterations
	print "maximum of power function on discrete_omega: ", pmax[num_iterations-1]
	print "maximum residual: ", max(residual_eval)

	# plotting the training
	plt.figure(1)
	plt.title("residuals")
	plt.plot(discrete_omega, residual_eval, 'bo', discrete_omega[selected], residual_eval[selected], 'ro')
	plt.figure(2)
	plt.title("maximum of the power function evaluated on discrete_omega at each iteration")
	plt.plot(range(0,max_iterations), pmax, 'r^')

	return [a, sf, discrete_omega[selected], num_iterations]

# Test
if __name__ == "__main__":
	# discrete_omega = np.random.rand(20, 2)
	X = np.arange(-5, 5, 0.25)
	Y = np.arange(-5, 5, 0.25)

	# discrete_omega is of shape (len(X)**2, 2)
	discrete_omega = np.zeros((len(X)**2, 2))
	for i in range(len(X)):
		for j in range(len(Y)):
			discrete_omega[i*len(Y)+j, 0] = X[i]
			discrete_omega[i*len(Y)+j, 1] = Y[j]

	# X, Y in meshgrid form, ie X, Y of shape (len(X), len(X))
	X, Y = np.meshgrid(X, Y)

	def kernel(x,y):
		return math.exp(-np.linalg.norm(x-y)**2)

	# test function from R^2 to R: sin(2norm(x,y))
	R = np.sqrt(X**2 + Y**2)
	# in mesh grid form
	Z = np.sin(R)

	# in shape (len(X)**2, 1)
	f = np.array([])
	for i in range(len(X)):
		for j in range(len(X)):
			f = np.append(f, Z[i,j])
	f = f.reshape((-1,1))

	# computing the basis
	[a, sf, selection, num_iterations] = pgreedy(discrete_omega, kernel, f, 600, math.pow(10, -2))

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
	ax.set_zlim(-1.01, 1.01)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	fig.colorbar(surf, shrink=0.5, aspect=5)
	ax.scatter(X, Y, sf)
	plt.show()
