# CS 181, Spring 2016
# Homework 5: EM
# Name:
# Email:

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class LDA(object):

	# Initializes with the number of topics
	def __init__(self, num_topics):
		self.num_topics = num_topics

	# This should run the M step of the EM algorithm
	def M_step(self):
		pass

	# This should run the E step of the EM algorithm
	def E_step(self):
		pass

	# This should print the topics that you find
	def print_topics(self):
		pass

# This line loads the text for you. Don't change it! 
text_data = np.load("text.npy", allow_pickle=False)
with open('words.txt', 'r') as f:
	word_dict_lines = f.readlines()

# Feel free to add more functions as needed for the LDA class. You are welcome to change anything below this line. 
# However, your code should be contained in the constructor for the LDA class, and should be executed in a way 
# similar to the below.
LDAClassifier = LDA(num_topics=10)
LDAClassifier.print_topics()