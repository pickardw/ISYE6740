import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import csc_matrix, find
from itertools import combinations
import copy
from IPython.display import HTML

# change the value to 0 to run just kmeans without comparing to brute force
# method;
iscomparebruteforce = 0

dim = 2  # dimension of the data points
# change to larger number of data points and cno = 3 after comparison with brute force
# method
if iscomparebruteforce == 1:
    m = 6  # number of data points
    # fix the seed of the random number generator, so each time generate
    # the same random points;
    np.random.seed(42)
    x = np.concatenate((np.random.randn(dim, m) + np.tile(np.array([4, 1]), (m, 1)).T,
                        np.random.randn(dim, m) + np.tile(np.array([4, 4]), (m, 1)).T), axis=1)
    x = np.concatenate((x, np.random.randn(dim, m) +
                        np.tile(np.array([1, 2]), (m, 1)).T), axis=1)

    # number of clusters
    cno = 2
else:
    m = 100  # number of data points
    # fix the seed of the random number generator, so each time generate
    # the same random points;
    np.random.seed(42)
    x = np.concatenate((np.random.randn(dim, m) + np.tile(np.array([4, 1]), (m, 1)).T,
                        np.random.randn(dim, m) + np.tile(np.array([4, 4]), (m, 1)).T), axis=1)
    x = np.concatenate((x, np.random.randn(dim, m) +
                        np.tile(np.array([1, 2]), (m, 1)).T), axis=1)

    # # 
    # x = np.concatenate((x, np.tile(
    #     np.array([-7, -7]), (10, 1)).T + 0.1 * np.random.randn(dim, 10)), axis=1)
    # number of clusters
    cno = 6
m = x.shape[1]

# Initialization of cluster centers
# Set the random seed to ensure reproducibility
np.random.seed(42)
c = 6 * np.random.rand(dim, cno)
c_old = copy.deepcopy(c) + 10

# Store history for animation
c_hist = []
labels_hist = []
P_hist = []
obj_hist = []


i = 1
c_old = np.zeros_like(c)
tic = time.time()
# check whether the cluster centers still change
while np.linalg.norm(c - c_old, ord='fro') > 1e-6:
    # print("--iteration %d \n" % i)

    # record previous c;
    c_old = copy.deepcopy(c)

    # Assign data points to current cluster;
    # Squared norm of cluster centers.
    cnorm2 = np.sum(np.power(c, 2), axis=0)
    tmpdiff = 2 * np.dot(x.T, c) - cnorm2
    labels = np.argmax(tmpdiff, axis=1)

    # Update data assignment matrix;
    # The assignment matrix is a sparse matrix,
    # with size m x cno. Only one 1 per row.
    P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, cno))

    # adjust the cluster centers according to current assignment;
    cstr = ['r.', 'b.', 'g.', 'r+', 'b+', 'g+']
    obj = 0
    for k in range(0, cno):
        idx = find(P[:, k])[0]
        nopoints = idx.shape[0]
        if nopoints == 0:
            # a center has never been assigned a data point;
            # re-initialize the center;
            c[:, k] = np.random.rand(dim, 1)[:, 0]
        else:
            # equivalent to sum(x(:,idx), 2) ./ nopoints;
            c[:, k] = ((P[:, k].T.dot(x.T)).T / float(nopoints))[:, 0]
        obj = obj + \
            np.sum(
                np.sum(np.power(x[:, idx] - np.tile(c[:, k], (nopoints, 1)).T, 2)))
    # Store for animation
    c_hist.append(c.copy())
    labels_hist.append(labels.copy())
    P_hist.append(P.copy())
    obj_hist.append(obj)
    i = i + 1

toc = time.time()

# k-means will be much faster than brute force enumeration
# If you run several times without setting the random seed, you will see that the objective function is different

print('Elapsed time is %f seconds \n' % float(toc - tic))
print('obj =', obj)

# For comparison: two random clusters
ra = np.random.randn(m, 1)
idx1 = np.where(ra > 0)[0]
idx2 = np.setdiff1d(np.arange(0, m, 1), idx1)

center1 = np.mean(x[:, idx1], axis=1)
center2 = np.mean(x[:, idx2], axis=1)

newobj = np.sum(np.sum(np.power(x[:, idx1] - np.tile(center1, (idx1.shape[0], 1)).T, 2)))  \
    + np.sum(np.sum(np.power(x[:, idx2] -
                             np.tile(center2, (idx2.shape[0], 1)).T, 2)))

print('newobj =', newobj)