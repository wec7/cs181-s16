# CS 181, Spring 2016
# Homework 4: Clustering
# Name:
# Email:

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans as BasicKMeans

class KMeans(BasicKMeans):
	# K is the K in KMeans
	# useKMeansPP is a boolean. If True, you should initialize using KMeans++
	def __init__(self, K, useKMeansPP):
		super(KMeans, self).__init__(n_clusters=K)
		self.K = K
		self.useKMeansPP = useKMeansPP		

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	def fit(self, X):
		self.X = X
		return super(KMeans, self).fit(X)

	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		pass

	# This should return the arrays for D images from each cluster that are representative of the clusters.
	def get_representative_images(self, D):
		pass

	# img_array should be a 2D (square) numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. 
	# However, we do ask that any images in your writeup be grayscale images, just as in this example.
	def create_image_from_array(self, img_array):
		plt.figure()
		plt.imshow(img_array, cmap='Greys_r')
		plt.show()
		return

# This line loads the images for you. Don't change it! 
pics = np.load("images.npy", allow_pickle=False)
pics_plain = [np.reshape(pic, 28*28) for pic in pics]

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, 
# though you may add more public methods for things like the visualization if you want.
# Also, you must cluster all of the images in the provided dataset, so your code should be fast enough to do that.
K = 10
KMeansClassifier = KMeans(K=10, useKMeansPP=False)
print KMeansClassifier.fit_predict(pics_plain)
# KMeansClassifier.create_image_from_array(pics[0])




