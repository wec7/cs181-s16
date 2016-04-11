#!/usr/bin/python
#
# Created by Albert Au Yeung (2010)
#
# An implementation of matrix factorization
#
import numpy
import pandas
import csv
from sklearn.preprocessing import Imputer

if __name__ == "__main__":

    train_file = 'train.csv'
    test_file  = 'test.csv'
    soln_file  = 'mean.csv'

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

    # Compute the matrix factorization
    df = pandas.DataFrame.from_dict(train_data).fillna(0.)
    R = numpy.array(df.values)

    model = Imputer(missing_values=0, strategy='mean', verbose=True)
    nR = model.fit_transform(R)

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
                res    = nDf[user][nDf.index==artist].values[0]
                print id, res
                soln_csv.writerow([id, res])
