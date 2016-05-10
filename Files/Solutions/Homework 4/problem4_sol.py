# CS 181, Spring 2016
# Homework 4: Clustering
# Solution written by Ankit Gupta, ankitgupta@college.harvard.edu

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class KMeans(object):
	# K is the K in KMeans
	# useKMeansPP is a boolean. If True, you should initialize using KMeans++
	def __init__(self, K, useKMeansPP):
		self.K = K
		self.useKMeansPP = useKMeansPP

	def __distances_to_closest_mean(self):
		distances = self.__getdistances()
		return np.amin(distances, axis=1)


	def __initialize_means_pp(self):
		# Pick a point to be the first mean
		first_mean = self.X[np.random.choice(self.X.shape[0])]
		# Set all of the means to be the first mean
		self.means = np.zeros((self.K, self.X.shape[1])) + first_mean

		# Pick the rest of the means
		for i in range(1, self.K):
			#print i
			# Get the distance to the cloest mean for each point
			closest_distances_sq = np.power(self.__distances_to_closest_mean(), 2)
			# Normalize
			normalized = closest_distances_sq/np.sum(closest_distances_sq)

			pick = np.random.choice(self.X.shape[0], p=normalized)
			self.means[i] = self.X[pick]
			#print self.means[i]

	def __objective(self):
		distances = self.__getdistances()
		elements = distances[np.arange(self.X.shape[0]), self.assignments]
		return np.sum(elements)

	def __update_means(self):
		for k in range(self.K):
			in_class_values = self.X[self.assignments == k]
			#print(len(in_class_values))
			self.means[k] = np.mean(in_class_values, axis=0)

	#def __initialize_assignments(self):
	#	self.assignments = np.random.randint(self.K, size=self.X.shape[0])

	def __initialize_means(self):
		#self.means = np.zeros((self.K, self.X.shape[1]))
		self.means = np.random.randn(self.K, self.X.shape[1])
		#self.__update_means()

	# Returns an N x K array, where Distances_nk is the distance from point n to mean of cluster k
	def __getdistances(self):
		distances = np.zeros((self.X.shape[0], self.K))
		for k in range(self.K):
			distances[:, k] = np.sum(np.power(self.X - self.means[k], 2), axis=1)
		return distances
	def __update_assignments(self):
		distances = self.__getdistances()
		self.assignments = np.argmin(distances, axis=1)

	# Get indicies into n smallest values of numpy array
	def __get_d_smallest(self, arr, n):
		return arr.argsort()[:n]

	# X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images.
	def fit(self, X):
		self.X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
		self.objective_vals = []
		if self.useKMeansPP:
			self.__initialize_means_pp()
		else:
			self.__initialize_means()
		print "Done initializing means"
		for i in range(100):
			print i
			self.__update_assignments()
			self.objective_vals.append(self.__objective())
			self.__update_means()
			self.objective_vals.append(self.__objective())
		print "Final Objective:", self.__objective()


	# This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
	def get_mean_images(self):
		return self.means

	# This should return the arrays for D images from each cluster that are representative of the clusters. 
	def get_representative_images(self, D):
		self.representatives = np.zeros((self.K, D))
		distances = self.__getdistances()
		for k in range(self.K):
			self.representatives[k] = self.__get_d_smallest(distances[:, k], D)
		return self.representatives

	# img_array should be a 28x28 numpy array.
	# Note, you are welcome to change this function (including its arguments and return values) to suit your needs. However, we do ask that your images be 28x28 grayscale images, 
	#        just as in this example.
	def create_image_from_array(self, img_array):
		plt.figure()
		plt.imshow(img_array, cmap='Greys_r')
		plt.show()
		return

	# This assumes that fit() has already been called
	def create_objective_graph(self):
		if not self.objective_vals:
			return
		plt.figure()
		plt.plot(np.arange(len(self.objective_vals)), self.objective_vals)
		plt.show()


# This line loads the images for you. Don't change it!
pics = np.load("images.npy", allow_pickle=False)

# You are welcome to change anything below this line. This is just an example of how your code may look.
# That being said, keep in mind that you should not change the constructor for the KMeans class, though you may add more public methods if you want.
K = 10
KMeansClassifier = KMeans(K=10, useKMeansPP=False)
KMeansClassifier.fit(pics)
KMeansClassifier.create_objective_graph()

# means = KMeansClassifier.get_mean_images()
# for mean in means:
# 	KMeansClassifier.create_image_from_array(mean.reshape(28,28))

# for vals in KMeansClassifier.get_representative_images(4):
# 	print vals
# 	for representative in vals:
# 		KMeansClassifier.create_image_from_array(pics[representative])



