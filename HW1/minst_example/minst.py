import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, find

def show_image_function(centroids, H, W):    
    '''
    centroids - centroids
    H : height of figure
    W: width of figure
    '''
    N = int((centroids.shape[1]) / (H * W))
    assert (N == 3 or N == 1)
    
    # Organize the images into rows x cols.
    K = centroids.shape[0]
    COLS = round( math.sqrt(K) )
    ROWS = math.ceil(K / COLS)
    
    COUNT = COLS * ROWS
        
    plt.clf()
    
    image = np.ones((ROWS * (H + 1), COLS * (W + 1), N)) * 100
    for i in range(0, centroids.shape[0]):
        r = math.floor(i / COLS)
        c = np.mod(i , COLS)
        
        image[(r * (H + 1) + 1):((r + 1) * (H + 1)), \
            (c * (W + 1) + 1):((c + 1) * (W + 1)), :] = \
            centroids[i, :].reshape((H, W, N))
    
    plt.imshow(image.squeeze(), plt.cm.gray)

matFile = sio.loadmat('data/usps_all.mat')
data = matFile['data']
pixelno = data.shape[0]
digitno = data.shape[1]
classno = data.shape[2]
H = 16
W = 16
plt.figure(0)

# Display all images of digits 1 and 0.
digits_01 = np.concatenate(
    (np.array(data[:, :, 0]), np.array(data[:, :, 9])), axis=1).T
show_image_function(digits_01, H, W)
plt.title('digit 1 and 0')
plt.show()

x0 = np.array(data[:, :, [0, 9]]).reshape((pixelno, digitno * 2))
x = np.array((data[:, :, [0, 9]]).reshape(
    (pixelno, digitno * 2)), dtype="float")
y = np.concatenate((np.ones((1, digitno)), 2 * np.ones((1, digitno))), axis=1)
m = x.shape[1]

# Number of clusters.
cno = 8

# Randomly initialize centroids with data points;
c = x[:, np.random.randint(x.shape[1], size=(1, cno))[0]]

# Number of iterations
iterno = 200

for iter in range(0, iterno):
    # print(f"running --iteration {iter}", end = "\r")

    # norm squared of the centroids;
    c2 = np.sum(np.power(c, 2), axis=0, keepdims=True)

    # For each data point x, computer min_j  -2 * x' * c_j + c_j^2;
    # Note that here is implemented as max, so the difference is negated.
    tmpdiff = (2 * np.dot(x.T, c) - c2)
    labels = np.argmax(tmpdiff, axis=1)

    # Update data assignment matrix;
    # The assignment matrix is a sparse matrix,
    # with size m x cno. Only one 1 per row.
    P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, cno))
    count = P.sum(axis=0)

    # Recompute centroids;
    # x*P implements summation of data points assigned to a given cluster.
    c = np.array((P.T.dot(x.T)).T / count)


# Visualize results.
for i in range(0, cno):
    plt.figure(i + 1)
    # Final cluster assignments in P.
    show_image_function(x0[:, find(P[:, i])[0]].T, H, W)
    plt.title('cluster: %s' % str(i + 1))

plt.show()