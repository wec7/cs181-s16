import csv
import numpy as np
import matplotlib.pyplot as plt

csv_filename = 'motorcycle.csv'
csv_filename = 'congress-ages.csv'
times  = []
forces = []
with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        times.append(float(row[0]))
        forces.append(float(row[1]))

# Turn the data into numpy arrays.
times  = np.array(times)
forces = np.array(forces)

# Plot the data.
# plt.plot(times, forces, 'o')
# plt.show()

def make_basis(xx):
    X = np.ones(xx.shape).T   
    # a
    #for j in xrange(1, 8):
    #    X = np.vstack((X, xx**j))
    # b
    #for j in xrange(1, 4):
    #    X = np.vstack((X, xx**j))
    # c
    #for j in xrange(1, 5):
    #    X = np.vstack((X, np.sin(xx/j)))
    # d
    for j in xrange(1, 8):
        X = np.vstack((X, np.sin(xx/j)))
    # e
    #for j in xrange(1, 21):
    #    X = np.vstack((X, np.sin(xx/j)))
    return X.T

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(times.shape), times)).T
X = make_basis(times)

# Nothing fancy for outputs.
Y = forces

# Find the regression weights using the Moore-Penrose pseudoinverse.
w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))

# Compute the regression line on a grid of inputs.
grid_times = np.linspace(75, 120, 200)
grid_X     = make_basis(grid_times)
grid_Yhat  = np.dot(grid_X, w)

# Plot the data and the regression line.
plt.plot(times, forces, 'o',
        grid_times, grid_Yhat, '-')
plt.xlabel("Congress age (nth Congress)")
plt.ylabel("Average age")
plt.show()
        
