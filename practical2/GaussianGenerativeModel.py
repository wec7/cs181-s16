from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class GaussianGenerativeModel:
    def __init__(self, isSharedCovariance=False):
        self.isSharedCovariance = isSharedCovariance

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.K = Y.max()+1
        self.D = X.shape[1]
        means = []
        covmats = []
        invmats = []
        covdets = []
        if self.isSharedCovariance:
            allcov = []
            for k in range( self.K ):
                samp = X[Y==k,:]
                n_samp = samp.shape[0]
                mean = samp.mean(axis=0)
                mean = np.reshape(mean,(1,self.D))
                means.append(mean)
                dev = samp - np.dot( np.ones((n_samp,1)), mean )
                covmat = np.dot( dev.T, dev ) 
                if( len( allcov ) == 0 ):
                    allcov = covmat
                else:
                    allcov += covmat
                    
            allcov = allcov * ( 1.0 / X.shape[0] )
            allinv = np.linalg.inv(allcov)
            alldet = np.log(np.linalg.det(allcov))
            for k in range( self.K ):
                covmats.append(allcov)
                invmats.append(allinv)
                covdets.append(alldet)
        else:
            for k in range( self.K ):
                samp = X[Y==k,:]
                n_samp = samp.shape[0]
                mean = samp.mean(axis=0)
                mean = np.reshape(mean,(1,self.D))
                means.append(mean)
                dev = samp - np.dot( np.ones((n_samp,1)), mean )
                covmat = np.dot( dev.T, dev ) * (1.0/n_samp)
                covmats.append(covmat)
                invmats.append( np.linalg.inv(covmat) )
                covdets.append( np.log(np.linalg.det(covmat)) )
        self.means = means
        self.covmats = covmats
        self.invmats = invmats
        self.covdets = covdets
        # print "Gaussian training completed."
        return
    
    # test in sample results
    def test_insample( self ):
        Y_pred = self.predict( self.X )
        incorr = 0
        for idx, pred in enumerate( Y_pred ):
            if pred != self.Y[idx]:
                incorr = incorr+1
        self.insample_err = (incorr + 0.0) / len( self.Y )

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        M = X_to_predict.shape[0]
        prob = [] # will use vstack, so output is KxM
        Y = []
        for k in range( self.K ):
            xdiff = X_to_predict - np.dot( np.ones((M,1)), self.means[ k ] )
            eng = []
            for xrow in xdiff:
                eng.append( np.dot( np.dot( xrow, self.invmats[k]), xrow.T) + self.covdets[k] ) 
            eng = np.array( eng )
            if( len( prob ) == 0 ):
                prob = eng
            else:
                prob = np.vstack((prob,eng))
                
        for ent in prob.T:
            val = np.argmin( ent )
            Y.append(val)
        return np.array(Y)

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
