# CS 181, Spring 2016
# Homework 5: EM
# Name:
# Email:

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import lda

class LDA(lda.LDA):

    # This should run the M step of the EM algorithm
    def M_step(self):
        pass

    # This should run the E step of the EM algorithm
    def E_step(self):
        pass

    def Vocab(self):
    	with open('words.txt', 'r') as f:
    		word_dict_lines = f.readlines()
    	return [line.split()[1] for line in word_dict_lines]

    # This should print the topics that you find
    def print_topics(self, n_top_words=8):
        topic_word = self.topic_word_ 
        vocab = self.Vocab()
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
            print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    def plot_loglikelihood(self):
    	plt.plot(model.loglikelihoods_)
    	plt.show()

# This line loads the text for you. Don't change it! 
text_data = np.load("text.npy", allow_pickle=False)

# Preprocess
text_data = text_data.astype(int)
doc_size = max(text_data[:,0])
vocab_size = max(text_data[:,1])
X = np.zeros((doc_size, vocab_size), dtype=np.int64)
for data in text_data:
    X[data[0]-1][data[1]-1] = data[2]

# Feel free to add more functions as needed for the LDA class. You are welcome to change anything below this line. 
# However, your code should be contained in the constructor for the LDA class, and should be executed in a way 
# similar to the below.
LDAClassifier = LDA(n_topics=10, n_iter=500)
LDAClassifier.fit(X)
LDAClassifier.print_topics()