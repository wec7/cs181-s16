# Don't change these imports. Note that the last two are the
# class implementations that you will implement in
# LogisticRegression.py and GaussianNaiveBayes.py
import matplotlib.pyplot as plt
import pandas as pd
from LogisticRegression import LogisticRegression
from GaussianGenerativeModel import GaussianGenerativeModel


## These are the hyperparameters to the classifiers. You may need to
# adjust these as you try to find the best fit for each classifier.

# Logistic Regression parameters
eta = .0001
lambda_parameter = .001
eta = .0001
lambda_parameter = 0.0
lambda_parameter = 0.1
lambda_parameter = 0.01
lambda_parameter = 0.001


# Do not change anything below this line!!
# -----------------------------------------------------------------

# Read from file and extract X and Y
df = pd.read_csv("fruit.csv")
X = df[['width', 'height']].values
Y = (df['fruit'] - 1).values

nb1 = GaussianGenerativeModel(isSharedCovariance=False)
nb1.fit(X,Y)
nb1.test_insample()
print "Gaussian model with separate covariance: in-sample error rate %.1f%%" % (nb1.insample_err * 100.0 )
nb1.visualize("generative_result_separate_covariances.png",show_charts=True)

nb2 = GaussianGenerativeModel(isSharedCovariance=True)
nb2.fit(X,Y)
nb2.test_insample()
print "Gaussian model with shared covariance: in-sample error rate %.1f%%" % (nb2.insample_err * 100.0 )
nb2.visualize("generative_result_shared_covariances.png",show_charts=True)

for idx,lamda in enumerate([ 0.0001, 0.001, 0.01, 0.1, 1, 0 ]):
    lr = LogisticRegression(eta=eta, lambda_parameter=lamda)
    lr.fit(X,Y)
    lr.test_insample()
    print "Logistic regression with lambda %.5f, in-sample error rate %.1f%%" % ( lamda, lr.insample_err * 100.0 )
    lr.visualize('logistic_regression_result_%d.png' % idx, show_charts=True)



