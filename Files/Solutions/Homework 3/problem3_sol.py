# CS 181, Harvard University
# Spring 2016
# Author: Ankit Gupta

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron


# Simple kernel
def K(x_i, x_j):
	return np.dot(x_i, x_j)

class KernelPerceptron(Perceptron):
	def __init__(self, numsamples):
		self.S = {}
		self.numsamples = numsamples

	def __calculate_y_hat(self, x_t):
		total = 0
		#x_t = self.X[t]
		for i, a_i in self.S.iteritems():
			x_i = self.X[i]
			total += a_i*K(x_i, x_t)
		return total

	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		assert(X.shape[0] == Y.shape[0])

		got = 0
		saw = 0
		for t in np.random.randint(X.shape[0], size=self.numsamples):
			saw += 1
			x_t =  X[t]
			y_t = Y[t]
			if y_t*self.__calculate_y_hat(x_t) <= 0:
				self.S[t] = y_t
				got += 1
				print got, saw


	def predict_element(self, x):
		y_hat = self.__calculate_y_hat(x)
		return 1 if y_hat > 0 else 0

	def predict(self, X):
		return np.array([self.predict_element(x) for x in X])



class BudgetKernelPerceptron(Perceptron):
	def __init__(self, beta, N, numsamples):
		self.beta = beta
		self.N = N
		self.numsamples = numsamples
		self.S = {}
	def __calculate_y_hat(self, x_t):
		total = 0
		for i, a_i in self.S.iteritems():
			x_i = self.X[i]
			total += a_i*K(x_i, x_t)
		return total

	def remove_worst(self):
		argmax = None
		max_val = -1e100
		for i, a_i in self.S.iteritems():
			x_i = self.X[i]
			val = a_i*(self.__calculate_y_hat(x_i) - a_i*K(x_i, x_i))
			if val > max_val:
				argmax = i
		del self.S[argmax]
		
	def fit(self, X, Y):
		self.X = X
		self.Y = Y
		assert(X.shape[0] == Y.shape[0])

		saw = 0
		for t in np.random.randint(X.shape[0], size=self.numsamples):
			saw += 1
			x_t =  X[t]
			y_t = Y[t]
			if y_t*self.__calculate_y_hat(x_t) <= self.beta:
				self.S[t] = y_t
				print len(self.S), saw
				if len(self.S) > self.N:
					self.remove_worst()

	def predict_element(self, x):
		y_hat = self.__calculate_y_hat(x)
		return 1 if y_hat > 0 else 0

	def predict(self, X):
		return np.array([self.predict_element(x) for x in X])


# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 10
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=False, save_fig=False, include_points=True)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=False, include_points=True)