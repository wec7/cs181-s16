# CS 181, Spring 2016
# Homework 5: EM
# Name:
# Email:

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import lda
import lda._lda
import lda.utils

class LDA(object):

    def __init__(self, n_topics, n_iter=2000):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.alpha = 0.1
        self.eta = 0.01
        self.random_state = None
        self.refresh = 10

        if alpha <= 0 or eta <= 0:
            raise ValueError("alpha and eta must be greater than zero")

        # random numbers that are reused
        rng = lda.utils.check_random_state(random_state)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates

    def fit(self, X):
        random_state = self.random_state
        rands = self._rands.copy()
        self._initialize(X)
        for it in range(self.n_iter):
            random_state.shuffle(rands)
            if it % self.refresh == 0:
                ll = self.loglikelihood()
                # keep track of loglikelihoods for monitoring convergence
                self.loglikelihoods_.append(ll)
            self._sample_topics(rands)
        ll = self.loglikelihood()
        # note: numpy /= is integer division
        self.components_ = (self.nzw_ + self.eta).astype(float)
        self.components_ /= np.sum(self.components_, axis=1)[:, np.newaxis]
        self.topic_word_ = self.components_
        self.doc_topic_ = (self.ndz_ + self.alpha).astype(float)
        self.doc_topic_ /= np.sum(self.doc_topic_, axis=1)[:, np.newaxis]

        # delete attributes no longer needed after fitting to save memory and reduce clutter
        del self.WS
        del self.DS
        del self.ZS
        return self

    def _initialize(self, X):
        D, W = X.shape
        N = int(X.sum())
        n_topics = self.n_topics
        n_iter = self.n_iter

        self.nzw_ = nzw_ = np.zeros((n_topics, W), dtype=np.intc)
        self.ndz_ = ndz_ = np.zeros((D, n_topics), dtype=np.intc)
        self.nz_ = nz_ = np.zeros(n_topics, dtype=np.intc)

        self.WS, self.DS = WS, DS = lda.utils.matrix_to_lists(X)
        self.ZS = ZS = np.empty_like(self.WS, dtype=np.intc)
        np.testing.assert_equal(N, len(WS))
        for i in range(N):
            w, d = WS[i], DS[i]
            z_new = i % n_topics
            ZS[i] = z_new
            ndz_[d, z_new] += 1
            nzw_[z_new, w] += 1
            nz_[z_new] += 1
        self.loglikelihoods_ = []

    def loglikelihood(self):
        nzw, ndz, nz = self.nzw_, self.ndz_, self.nz_
        alpha = self.alpha
        eta = self.eta
        nd = np.sum(ndz, axis=1).astype(np.intc)
        return lda._lda._loglikelihood(nzw, ndz, nz, nd, alpha, eta)

    def _sample_topics(self, rands):
        n_topics, vocab_size = self.nzw_.shape
        alpha = np.repeat(self.alpha, n_topics).astype(np.float64)
        eta = np.repeat(self.eta, vocab_size).astype(np.float64)
        lda._lda._sample_topics(self.WS, self.DS, self.ZS, self.nzw_, self.ndz_, self.nz_,
                                alpha, eta, rands)

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