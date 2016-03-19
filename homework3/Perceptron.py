# CS 181, Harvard University
# Spring 2016
# Author: Ankit Gupta
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c

# This is the base Perceptron class. Do not modify this file. 
# You will inherit this class in your implementations of KernelPerceptron and BudgetKernelPerceptron.
class Perceptron(object):
	def __init__(self, numsamples):
		self.numsamples = numsamples
		
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		assert(X.shape[0] == Y.shape[0])

	def predict(self, X):

		# This is a temporary predict function so that the distribution code compiles
		# You should delete this and write your own
		return (X[:, 0] > .5)

	# Do not modify this method!
	def visualize(self, output_file, width=3, show_charts=False, save_fig=True, include_points=True):
		X = self.X

		# Create a grid of points
		x_min, x_max = min(X[:, 0] - width), max(X[:, 0] + width)
		y_min, y_max = min(X[:, 1] - width), max(X[:, 1] + width)
		xx,yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min,
		    y_max, .05))

		# Flatten the grid so the values match spec for self.predict
		xx_flat = xx.flatten()
		yy_flat = yy.flatten()
		X_topredict = np.vstack((xx_flat,yy_flat)).T

		# Get the class predictions
		Y_hat = self.predict(X_topredict)
		Y_hat = Y_hat.reshape((xx.shape[0], xx.shape[1]))

		cMap = c.ListedColormap(['r','b','g'])

		# Visualize them.
		plt.figure()
		plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
		if include_points:
			plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
		if save_fig:
			plt.savefig(output_file)
		if show_charts:
			plt.show()