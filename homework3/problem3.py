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

class LaSVM(KernelPerceptron):
    def __init__(self, C, kernel, tau, eps=0.001):
        self.S = []
        self.a = []
        self.g = []
        self.y = []
        self.C = C
        self.k = kernel
        self.tau = tau
        self.eps = eps
        self.b = 0
        self.delta = 0
        self.i = 0
        self.misses = 0
        
    def predict(self, v):
        return sum(self.a[i]*self.k(self.S[i],v) for i in xrange(len(self.S)))

    def A(self, i):
        return min(0, self.C*self.y[i])
    
    def B(self, i):
        return max(0, self.C*self.y[i])

    def tau_violating(self, i, j):
        return ((self.a[i] < self.B(i)) and
                (self.a[j] > self.A(j)) and
                (self.g[i] - self.g[j] > self.tau))

    def extreme_ij(self):
        S = self.S
        i = np.argmax(list((self.g[i] if self.a[i]<self.B(i) else -np.inf)
                           for i in xrange(len(S))))
        j = np.argmin(list((self.g[i] if self.a[i]>self.A(i) else np.inf)
                           for i in xrange(len(S))))
        return i,j

    def lbda(self, i, j):
        S = self.S
        l= min((self.g[i]-self.g[j])/(self.k(S[i],S[i])+self.k(S[j],S[j])-self.k(S[i],S[j])),
               self.B(i)-self.a[i],
               self.a[j]-self.A(j))
        self.a[i] += l
        self.a[j] -= l
        for s in xrange(len(S)):
            self.g[s] -= l*(self.k(S[i],S[s])-self.k(S[j],S[s]))
        return l
    
    def lasvm_process(self, v, cls, w):
        self.S.append(v)
        self.a.append(0)
        self.y.append(cls)
        self.g.append(cls - self.predict(v))
        if cls > 0:
            i = len(self.S)-1
            foo, j = self.extreme_ij()
        else:
            j = len(self.S)-1
            i, foo = self.extreme_ij()
        if not self.tau_violating(i, j): return
        S = self.S
        lbda = self.lbda(i,j)

    def lasvm_reprocess(self):
        S = self.S
        i,j = self.extreme_ij()
        if not self.tau_violating(i,j): return
        lbda = self.lbda(i,j)
        i,j = self.extreme_ij()
        to_remove = []
        for s in xrange(len(S)):
            if self.a[s] < self.eps:
                to_remove.append(s)
        for s in reversed(to_remove):
            del S[s]
            del self.a[s]
            del self.y[s]
            del self.g[s]
        i,j = self.extreme_ij()
        self.b = (self.g[i]+self.g[j])/2.
        self.delta = self.g[i]-self.g[j]

    def update(self, v, c, w):
        if len(self.S) < 10:
            self.S.append(v)
            self.y.append(c)
            self.a.append(c)
            self.g.append(0)
            for i in xrange(len(self.S)):
                self.g[i] = self.y[i]-self.predict(self.S[i])
        else:
            if c*(self.predict(v) + self.b) < 0:
                self.misses += 1
            self.i += 1
            self.lasvm_process(v,c,w)
            self.lasvm_reprocess()
            self.lasvm_reprocess()
            if self.i % 1000 == 0:
                print "m", self.misses, "s", len(self.S)
                self.misses = 0

# Do not change these three lines.
data = np.loadtxt("data.csv", delimiter=',')[:2000]
X = data[:, :2]
Y = data[:, 2]

# These are the parameters for the models. Please play with these and note your observations about speed and successful hyperplane formation.
beta = 0
N = 100
numsamples = 2000

kernel_file_name = 'k.png'
budget_kernel_file_name = 'bk.png'

# Don't change things below this in your final version. Note that you can use the parameters above to generate multiple graphs if you want to include them in your writeup.
k = KernelPerceptron(numsamples, kernel=gaussian_kernel)
k.fit(X,Y)
k.visualize(kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)

# bk = BudgetKernelPerceptron(beta, N, numsamples)
# bk.fit(X, Y)
# bk.visualize(budget_kernel_file_name, width=0, show_charts=True, save_fig=True, include_points=False)
