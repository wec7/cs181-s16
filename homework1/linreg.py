#####################
# CS 181, Spring 2016
# Homework 1, Problem 3
#
##################

import csv
import numpy as np
import matplotlib.pyplot as plt

def create_basis_matrix(times, basis_func_type, degree):
	assert(basis_func_type in ['polynomial', 'sin'])
	ret = [np.ones(times.shape)]
	if basis_func_type == 'polynomial':
		for i in xrange(degree):
			ret.append(times**(i+1))
		return np.vstack(ret)
	else:
		for i in xrange(degree):
			ret.append(np.sin(times/(i+1)))
		return np.vstack(ret)

def plot(basis_func_type = 'polynomial', degree=1):
	csv_filename = 'congress-ages.csv'
	times  = []
	ages = []

	with open(csv_filename, 'r') as csv_fh:

	    # Parse as a CSV file.
	    reader = csv.reader(csv_fh)

	    # Skip the header line.
	    next(reader, None)

	    # Loop over the file.
	    for row in reader:

	        # Store the data.
	        times.append(float(row[0]))
	        ages.append(float(row[1]))

	# Turn the data into numpy arrays.
	times  = np.array(times)
	ages = np.array(ages)

	# Plot the data.
	plt.plot(times, ages, 'o')
	plt.xlabel("Congress age (nth Congress)")
	plt.ylabel("Average age")
	plt.show()

	# Create the simplest basis, with just the time and an offset.
	X = create_basis_matrix(times, basis_func_type, degree).T

	# Nothing fancy for outputs.
	Y = ages

	# Find the regression weights using the Moore-Penrose pseudoinverse.
	w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

	# Compute the regression line on a grid of inputs.
	# DO NOT CHANGE grid_times!!!!!
	grid_times = np.linspace(75, 120, 200)
	grid_X = create_basis_matrix(grid_times, basis_func_type, degree)
	grid_Yhat  = np.dot(grid_X.T, w)

	# Plot the data and the regression line.
	plt.plot(times, ages, 'o', grid_times, grid_Yhat, '-')
	plt.xlabel("Congress age (nth Congress)")
	plt.ylabel("Average age")
	plt.savefig("congress-ages-"+basis_func_type+'-'+str(degree))
	plt.show()

if __name__ == '__main__':
	plot(basis_func_type='polynomial', degree = 1)
	plot(basis_func_type='polynomial', degree = 7)
	plot(basis_func_type='polynomial', degree = 3)
	plot(basis_func_type='sin', degree = 4)
	plot(basis_func_type='sin', degree = 7)
	plot(basis_func_type='sin', degree = 20)