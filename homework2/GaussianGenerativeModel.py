from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel(LinearDiscriminantAnalysis):
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance
        super(GaussianGenerativeModel, self).__init__()

    # Just to show how to make 'private' methods
    def __meanCovMatrix(self):
        means = []
        cov = []
        shared_cov = np.zeros((self.nFeatures, self.nFeatures))
        for c in range(self.nClasses):
            rows_in_class = self.X[self.Y == c]
            means.append(np.mean(rows_in_class, axis=0))
            if self.isSharedCovariance:
                Cov_i = np.cov(rows_in_class.T)
                shared_cov += Cov_i*rows_in_class.shape[0]
            else:
                cov.append(np.cov(rows_in_class.T))
        if self.isSharedCovariance:
            return np.array(means), shared_cov/self.X.shape[0]
        return np.array(means), cov

    def __numClasses(self):
        rets = np.zeros(self.nClasses)
        for y in self.Y:
            rets[y] += 1
        return rets

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.nClasses = 3
        self.nFeatures = X.shape[1]
        self.N = X.shape[0]

        self.class_means, self.shared_covariance = self.__meanCovMatrix()
        class_counts = self.__numClasses()
        self.b = np.log(class_counts / (class_counts.sum()))
        # return super(GaussianGenerativeModel, self).fit(X, Y)

    def __gaussianProb(self, x):
        class_probs = np.zeros(self.nClasses)
        for c in range(self.nClasses):
            cov = self.shared_covariance if self.isSharedCovariance else self.shared_covariance[c]
            class_probs[c] = multivariate_normal.pdf(x, mean=self.class_means[c], cov=cov)
        return np.log(class_probs)

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        gaussian_probs = np.zeros((X_to_predict.shape[0], self.nClasses))
        for i in range(X_to_predict.shape[0]):
            gaussian_probs[i] = self.__gaussianProb(X_to_predict[i])
        return np.argmax(gaussian_probs + self.b, axis=1)
        # return super(GaussianGenerativeModel, self).predict(X_to_predict)

    # Do not modify this method!
    def visualize(self, output_file, width=3, show_charts=False):
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
        plt.scatter(X[:, 0], X[:, 1], c=self.Y, cmap=cMap)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
