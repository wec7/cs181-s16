import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp

class LogisticRegression:
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter

    # Numerically stable calculation of log softmax
    # Uses the property that log(softmax(z)) = z - log(sum(exp(z)))
    def __log_softmax(self, X, W):
        res = X.dot(W.T)
        lse = logsumexp(res, axis=1)[:, np.newaxis]
        return res - lse

    def __makeOneHot(self, i):
        onehot = np.zeros(self.nClasses)
        onehot[i] = 1
        return onehot

    def __makeAllOneHot(self, Y):
        return np.array([self.__makeOneHot(y) for y in Y])

    # Note that this is calculating the gradient over the whole training set
    # Assumes that Y is given as a one-hot vector.
    # Also adds regularization
    def __gradient(self):
        log_softmax_result = self.__log_softmax(self.X, self.W)
        softmax_result = np.exp(log_softmax_result)
        diff = softmax_result - self.Y
        W_grad  =  diff.T.dot(self.X)
        return W_grad + 2 * self.lambda_parameter * self.W

    def __updateParameters(self):
        W_grad = self.__gradient()
        self.W = self.W - W_grad * self.eta
        return W_grad

    def __loss(self):
        total_loss = 0.
        log_softmax_res = self.__log_softmax(self.X, self.W)
        total_loss = -np.sum(self.Y * log_softmax_res)
        # Add regularization
        return total_loss + self.lambda_parameter * (np.power(self.W, 2).sum())

    # 'lambda' is a reserved keyword in python.
    def fit(self, X, Y):
        assert(X.shape[0] == Y.shape[0])

        # Add a column of 1s
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)

        nClasses = max(Y) + 1
        nFeatures = X.shape[1]
        N = X.shape[0]
        self.nClasses = nClasses
        self.nFeatures = nFeatures
        self.X = X
        self.C = Y
        self.Y = self.__makeAllOneHot(Y)
        self.N = N
        self.W = np.zeros((nClasses, nFeatures))

        print(self.__loss())
        epoch = 0
        while True:
            epoch += 1
            # Update the gradient
            gradient = self.__updateParameters()

            # Every 10000 times, also calculate the gradient norm and loss, and
            # display it.
            if epoch % 10000 == 0:
                norm = np.linalg.norm(gradient)
                loss = self.__loss()
                print("On epoch: {}".format(epoch))
                print("   Gradient norm is {}".format(norm))
                print("   Loss is {}".format(loss))
                print(self.W)
                if norm < .000001:
                    break

    def predict(self, X_to_predict):
        X_to_predict = np.append(X_to_predict, np.ones((X_to_predict.shape[0],
                                                        1)), axis=1)
        softmax_res = X_to_predict.dot(self.W.T)
        predictions = np.argmax(softmax_res, axis=1)
        print(np.bincount(predictions))
        return predictions

    def visualize(self, output_file, width=2, show_charts=False):
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
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
