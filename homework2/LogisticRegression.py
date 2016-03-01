import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import logsumexp
from sklearn.linear_model import LogisticRegression as BaseLogisticRegression

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression(BaseLogisticRegression):
    def __init__(self, eta, lambda_parameter):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
        super(LogisticRegression, self).__init__()
    
    # Just to show how to make 'private' methods
    def __oneHotEncoding(self, Y):
        rets = []
        for y in Y:
            ret = np.zeros(3)
            ret[y] = 1
            rets.append(ret)
        return np.array(rets)

    def __softmax(self, X, W):
        activations = X.dot(W.T)
        return activations - logsumexp(activations, axis=1)[:, np.newaxis]

    def __loss(self):
        return -np.sum(self.Y * self.__softmax(self.X, self.W)) + self.lambda_parameter * (np.power(self.W, 2).sum())

    def __gradientLoss(self):
        return (np.exp(self.__softmax(self.X, self.W)) - self.Y).T.dot(self.X) + 2 * self.lambda_parameter * self.W

    def __iteration(self):
        gradientLoss = self.__gradientLoss()
        self.W = self.W - gradientLoss * self.eta
        return gradientLoss

    # TODO: Implement this method!
    def fit(self, X, Y):
        X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
        self.nFeatures = X.shape[1]
        self.X = X
        self.Class = Y
        self.Y = self.__oneHotEncoding(Y)
        self.N = X.shape[0]
        self.W = np.zeros((3, self.nFeatures))

        epoch = 0
        while True:
            epoch += 1
            gradient = self.__iteration()
            if epoch % 1e6 == 0:
                norm = np.linalg.norm(gradient)
                if norm < 1e-6:
                    break
        # return super(LogisticRegression, self).fit(X, Y)

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        X_to_predict = np.append(X_to_predict, np.ones((X_to_predict.shape[0], 1)), axis=1)
        return np.argmax(X_to_predict.dot(self.W.T), axis=1)
        # return super(LogisticRegression, self).predict(X_to_predict)

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
        plt.scatter(X[:, 0], X[:, 1], c=self.Class, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
