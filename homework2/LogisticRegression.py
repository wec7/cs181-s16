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
    def __oneHotEncoding(self, C):
        ret = []
        for c in C:
            if c == 0:
                ret.append([1, 0, 0])
            elif c == 1:
                ret.append([0, 1, 0])
            else:
                ret.append([0, 0 ,1])
        return np.array(ret)

    def __oneHotDecoding(self, C):
        ret = []
        for c in C:
            if c == [1, 0, 0]:
                ret.append(0)
            elif c == [0, 1, 0]:
                ret.append(1)
            else:
                ret.append(2)
        return np.array(ret)

    # TODO: Implement this method!
    def fit(self, X, C):
        self.X = X
        self.Class = C
        return super(LogisticRegression, self).fit(X, C)

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        return super(LogisticRegression, self).predict(X_to_predict)

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
