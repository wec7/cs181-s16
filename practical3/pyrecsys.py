import recsys.algorithm
recsys.algorithm.VERBOSE = True

from recsys.algorithm.factorize import SVD
svd = SVD()
svd.load_data(filename='train.csv', sep=',', format={'col':0, 'row':1, 'value':2})

k = 100
svd.compute(k=k, pre_normalize=None, mean_center=True, post_normalize=True)

MIN_RATING = 0.0
MAX_RATING = 5000.0

import csv
test_file = 'test.csv'
soln_file = 'recsys.csv'

with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]
            res    = svd.predict(artist, user, MIN_RATING, MAX_RATING)
            soln_csv.writerow([id, res])