# CS 181, Harvard University
# Spring 2016
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron

# Implement this class
class KernelPerceptron(Perceptron):
	def __init__(self, numsamples):
		self.numsamples = numsamples

	# Implement this!
	# def fit(self, X, Y):

	# Implement this!
	# def predict(self, X):

# Implement this class
class BudgetKernelPerceptron(Perceptron):
	def __init__(self, beta, N, numsamples):
		self.beta = beta
		self.N = N
		self.numsamples = numsamples
		
	# Implement this!
	# def fit(self, X, Y):

	# Implement this!
	# def predict(self, X):



# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 100
numsamples = 20000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)

bk = BudgetKernelPerceptron(beta, N, numsamples)
bk.fit(X, Y)
bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=True)
