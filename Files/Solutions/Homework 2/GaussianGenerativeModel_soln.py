from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c

class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    def fit(self, X, Y):
        nClasses = max(Y) + 1
        nFeatures = X.shape[1]
        N = X.shape[0]
        self.X = X
        self.Y = Y
        self.nClasses = nClasses
        self.nFeatures = nFeatures
        self.N = N
        assert(X.shape[0] == Y.shape[0])

        class_means, shared_covariance = self.__get_mean_and_covariance_matrix(X,
                Y, nFeatures, nClasses)

        # Get p(y) by normalize the distribution of counts
        class_counts = self.__getClassCounts(Y, nClasses)
        p_y = class_counts / (class_counts.sum())

        # Taking the log of the distributions allows us to get away with
        # addition instead of multiplication later.
        b = np.log(p_y)

        # Save all learned parameters
        self.class_means = class_means
        self.shared_covariance = shared_covariance
        self.b = b

    def __get_mean_and_covariance_matrix(self, X, Y, nfeatures, nclasses):
        means = []
        cov = []

        # Only used if self.isSharedCovariance is true
        shared_cov = np.zeros((nfeatures, nfeatures))

        # Filter by class, and calculate mean and covariance of each.
        for c in range(nclasses):
            rows_in_class = X[Y == c]
            means.append(np.mean(rows_in_class, axis=0))
            if self.isSharedCovariance:
                Cov_i = np.cov(rows_in_class.T)
                shared_cov += Cov_i*rows_in_class.shape[0]
            else:
                cov.append(np.cov(rows_in_class.T))

        # Return the shared covariance matrix if set that way, else return the
        # separate matricies.
        if self.isSharedCovariance:
            return np.array(means), shared_cov/X.shape[0]
        return np.array(means), cov

    def __gaussianProb(self, x, means, covariances, nClasses):
        class_probs = np.zeros(nClasses)
        for c in range(nClasses):
            if self.isSharedCovariance:
                class_probs[c] = multivariate_normal.pdf(x, mean=means[c],
                        cov=covariances)
            else:
                class_probs[c] = multivariate_normal.pdf(x, mean=means[c],
                        cov=covariances[c])
        return np.log(class_probs)

    def __getClassCounts(self, Y, nclasses):
        counts = np.zeros(nclasses)
        for y in Y:
            counts[y] += 1
        return counts

    def predict(self, X_to_predict):
        mus = self.class_means
        Sigma = self.shared_covariance
        b = self.b
        nClasses = mus.shape[0]
        gaussian_probs = np.zeros((X_to_predict.shape[0], nClasses))
        for i in range(X_to_predict.shape[0]):
            gaussian_probs[i] = self.__gaussianProb(X_to_predict[i], mus, Sigma,
                    nClasses)
        Y_hats = gaussian_probs + b
        predictions = np.argmax(Y_hats, axis=1)
        return predictions

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
