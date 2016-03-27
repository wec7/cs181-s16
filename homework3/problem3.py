# CS 181, Harvard University
# Spring 2016
import numpy as np 
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib.colors as c
from Perceptron import Perceptron

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

# Implement this class
class KernelPerceptron(Perceptron):
    def __init__(self, numsamples, kernel=linear_kernel, T=1):
        self.numsamples = numsamples
        self.kernel = kernel
        self.T = T

    def fit(self, X, y):
        self.X = X
        self.Y = y
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        for t in range(self.T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "%d support vectors out of %d points" % (len(self.alpha), n_samples)

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        return np.sign(self.project(X))

# Implement this class
class BudgetKernelPerceptron(KernelPerceptron):
    def __init__(self, beta, N, numsamples, kernel=linear_kernel, T=1):
        self.beta = beta
        self.N = N
        self.numsamples = numsamples
        self.kernel = kernel
        self.T = T

    def fit(self, X, y):
        self.X = X
        self.Y = y
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        self.alpha = np.zeros(n_samples, dtype=np.float64)
        sv = np.zeros(n_samples, dtype=bool)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        for t in range(self.T):
            for i in range(n_samples):
                if np.sum(K[:,i] * self.alpha * y) * y[i] <= self.beta:
                    self.alpha[i] += 1.0
                    sv[i] = True
                    if sum(sv) > self.N:
                    	na = (y*(y-self.alpha*K.diagonal())) * sv
                    	argmax_index = np.argmax(na)
                    	sv[argmax_index] = False

        # Support vectors
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print "%d support vectors out of %d points" % (len(self.alpha), n_samples)

# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')[:10000]
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 100
numsamples = 2000

kernel_file_name = 'smo.png'
budget_kernel_file_name = 'smo.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

# bk = BudgetKernelPerceptron(beta, N, numsamples)
# bk.fit(X, Y)
# bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
