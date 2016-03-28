import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from scipy.misc import logsumexp
from copy import deepcopy

# Please implement the fit and predict methods of this class. You can add additional private methods
# by beginning them with two underscores. It may look like the __dummyPrivateMethod below.
# You can feel free to change any of the class attributes, as long as you do not change any of 
# the given function headers (they must take and return the same arguments), and as long as you
# don't change anything in the .visualize() method. 
class LogisticRegression:
    def __init__(self, eta, lambda_parameter, normalize = False ):
        self.eta = eta
        self.lambda_parameter = lambda_parameter
        self.normalize = False
    
    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None
    # Convert to one hot coding, assuming class is from 0 to N-1
    def __convertonehot(self):
        T = np.zeros((self.N,self.K))
        # vectorize?
        for idx, val in enumerate(self.C):
            T[idx,val]=1
        self.T = T 
        
    # TODO: Implement this method!
    def fit(self, X, C):
        # K, number of classes, N, number of samples, D, dimention of features
        self.K = C.max()+1
        self.N = len( C )
        self.D = X.shape[1] + 1
        
        # normalize to ensure numerical stability
        mn = X.mean( axis = 0 )
        sd = X.std( axis = 0 )
        sd[ sd == 0 ] = 1.0
        self.train_mn = mn
        self.train_std = sd
        
        X_enh = deepcopy(X)
        if( self.normalize ):
            X_enh = ( X_enh - mn ) / sd

        self.X = X
        self.X_enh = np.hstack( ( X_enh, np.ones((self.N,1)) ) ) 
        self.C = C

        self.__convertonehot()
        self.W = np.zeros((self.D, self.K))
        #self.W = np.random.rand(self.D, self.K)

        max_iter = 100
        n_iter = 0
        self.err = []
        err_prev = 1e10
        
        while n_iter < max_iter:
            z = np.dot( self.X_enh, self.W )
            sig = np.exp(z) 
            sig = sig / np.dot( sig, np.ones((self.K,self.K)) )  
            self.sig = sig
            
            entr = - np.log(sig) * self.T
            err_new = entr.sum() + 0.5 * self.lambda_parameter * ( self.W * self.W ).sum()
            self.err.append( err_new )
            
            diff = abs( err_new - err_prev )
            err_prev = err_new
            
            if diff < self.eta:
                break
                
            #calculate Hessian matrix
            H = np.zeros((self.K*self.D,self.K*self.D))
            for k in range( self.K ):
                for j in range( self.K ):
                    i_kj = float( k == j )
                    s_coef = sig[:,k] * ( i_kj - sig[:,j] )  # should not have neg sign, order of (k,j)
                    H[k*self.D:(k+1)*self.D,j*self.D:(j+1)*self.D ] = np.dot( self.X_enh.T * np.reshape(s_coef,(1,self.N)), self.X_enh )
                    
            # print "Hessian", H   
            self.H = H
            H = H + self.lambda_parameter * np.eye( self.K * self.D, self.K * self.D )            
            W_vec = np.reshape( self.W.T, (self.D * self.K,1) )
            grad_mat  = np.dot( self.X_enh.T, sig - self.T )
            grad_vec = np.reshape( grad_mat.T, (self.D * self.K,1) ) + self.lambda_parameter * W_vec
            W_new = W_vec - np.linalg.solve( H, grad_vec )
            self.W = np.reshape( W_new, (self.K, self.D) ).T            
            n_iter = n_iter+1            
            
        #print "Number of iterations: %d" % n_iter
        #print "Incremental error improvement: %f" % diff
        
        return
    
    # test in sample results
    def test_insample( self ):
        Y_pred = self.predict( self.X )
        incorr = 0
        for idx, pred in enumerate( Y_pred ):
            if pred != self.C[idx]:
                incorr = incorr+1
        self.insample_err = (incorr + 0.0) / self.N

    # TODO: Implement this method!
    def predict(self, X_to_predict):
        # The code in this method should be removed and replaced! We included it just so that the distribution code
        # is runnable and produces a (currently meaningless) visualization.
        M = X_to_predict.shape[0]
        
        X_enh = deepcopy( X_to_predict )
        if( self.normalize ):
            X_enh = ( X_enh - self.train_mn ) / self.train_std        
        X_enh = np.hstack( ( X_enh, np.ones((M,1)) ) ) 
        
        z = np.dot( X_enh, self.W )
        sig = np.exp(z)
        sig = sig / np.dot( sig, np.ones((self.K,self.K)) ) 
        
        Y = []
        for ent in sig:
            val = np.argmax( ent )
            Y.append(val)
            
        return np.array(Y)

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
        
        #Y_hat = np.ones(Y_hat.shape, dtype = Y_hat.dtype)
        #self.Y_hat = Y_hat
        #self.xx = xx
        #self.yy = yy
                
        cMap = clrs.ListedColormap(['r','b','g'])

        # Visualize them.
        plt.figure()
        plt.pcolormesh(xx,yy,Y_hat, cmap=cMap)
        plt.scatter(X[:, 0], X[:, 1], c=self.C, cmap=cMap)
        plt.title( "Logistic Regression Results with Lambda %.4f" % self.lambda_parameter )
        plt.savefig(output_file)
        if show_charts:
            plt.show()
