#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
import numpy
import pandas
import csv

###############################################################################

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02, pairs=None):
    Q = Q.T
    for step in xrange(steps):
        for i, j in pairs:
            eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
            for k in xrange(K):
                P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i, j in pairs:
            e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
            for k in xrange(K):
                e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

###############################################################################

if __name__ == "__main__":

    train_file = 'train.csv'
    test_file  = 'test.csv'
    soln_file  = 'mf.csv'

    # Load the training data.
    train_data = {}
    with open(train_file, 'r') as train_fh:
        train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
        next(train_csv, None)
        for i, row in enumerate(train_csv):
            user   = row[0]
            artist = row[1]
            plays  = row[2]
        
            if not user in train_data:
                train_data[user] = {}
            
            train_data[user][artist] = int(plays)

    # Compute df and pairs
    df = pandas.DataFrame.from_dict(train_data).fillna(0)
    all_users = list(df.columns)
    all_artists = list(df.index)
    pairs = []
    for user, user_artists in train_data.iteritems():
        for artist in user_artists.keys():
            print user, artist
            pair = (all_artists.index(artist), all_users.index(user))
            pairs.append(pair)
    print "# pairs:", len(pairs)

    # Compute the matrix factorization
    R = numpy.array(df.values)

    N = len(R)
    M = len(R[0])
    K = 2

    P = numpy.random.rand(N,K)
    Q = numpy.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K, steps=1, pairs=pairs) #FIXME: steps should be larger
    nR = numpy.dot(nP, nQ.T)

    nDf = pandas.DataFrame(data=nR, index=df.index, columns=df.columns)

    # Write out test solutions.
    with open(test_file, 'r') as test_fh:
        test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
        next(test_csv, None)

        with open(soln_file, 'w') as soln_fh:
            soln_csv = csv.writer(soln_fh,
                                  delimiter=',',
                                  quotechar='"',
                                  quoting=csv.QUOTE_MINIMAL)
            soln_csv.writerow(['Id', 'plays'])

            for row in test_csv:
                id     = row[0]
                user   = row[1]
                artist = row[2]
                soln_csv.writerow([id, nDf[user][nDf.index==artist].values[0]])
