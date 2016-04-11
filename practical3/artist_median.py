import numpy as np
import csv

# Predict via the artist-specific median.
# If the artist has no data, use the global median.

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'artist_median.csv'

# Load the training data.
train_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not artist in train_data:
            train_data[artist] = {}
        
        train_data[artist][user] = int(plays)

# Compute the global median and per-artist median.
plays_array  = []
artist_medians = {}
for artist, artist_data in train_data.iteritems():
    artist_plays = []
    for user, plays in artist_data.iteritems():
        plays_array.append(plays)
        artist_plays.append(plays)

    artist_medians[artist] = np.median(np.array(artist_plays))
global_median = np.median(np.array(plays_array))

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

            if artist in artist_medians:
                soln_csv.writerow([id, artist_medians[artist]])
            else:
                print "User", id, "not in training data."
                soln_csv.writerow([id, global_median])
                