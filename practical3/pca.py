import numpy as np
import csv
from sklearn.preprocessing import Imputer
from sklearn.decomposition import IncrementalPCA

# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'user_median.csv'

# Load the training data.
train_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not user in train_data:
            train_data[user] = {}
        
        train_data[user][artist] = int(plays)

# Compute per-user median.
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        user_plays.append(plays)
    user_medians[user] = np.median(np.array(user_plays))

# PCA
df = pandas.DataFrame.from_dict(train_data)
df = df.fillna(df.median())
for user in df.columns:
    df[user] = df[user] / user_medians[user] - 1.
R = numpy.array(df.values)