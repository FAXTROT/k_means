import numpy as np
from matplotlib import pyplot as plt

data = np.array([        # dataset
    [1, 7],
    [1.1, 8],
    [1.3, 7.6],
    [1.4, 8.1],
    [1.7, 8.3],
    [2.5, 7.3],
    [2.21, 5],
    [2, 6],
    [3, 6],
    [1.2, 3],
    [2, 7.5],
    [2, 7],
    [3, 7.5],
    [2.6, 6.6],
    [1.5, 7.3],
    [1.5, 6.1],
    [3, 7],
    [3.5, 1],
    [4.9, 7.8],
    [9, 1],
    [8, 8.8],
    [6,4],
    [6.2, 3],
    [7, 3.5],
    [7.5, 4],
    [7.5, 3],
    [8, 2],
    [8, 3],
    [7.6, 2.5],
    [6, 5],
    [6.6, 4],
    [7, 5],
    [7.4, 5.6],
    [7.1, 4.9],
    [6.5, 4.5],
    [6.9, 1.9],
    [8, 5],
    [9, 5.5],
    [9, 7],
    ])

x = data[:, 0]
y = data[:, 1]

plt.plot(x, y, 'rx')
plt.title("Initial data")
plt.show()

mug = np.random.random([2, 1]) * 10
muy = np.random.random([2, 1]) * 10   # centroids

plt.plot(x, y, 'rx')
plt.plot(mug[0], mug[1], 'g.')
plt.plot(muy[0], muy[1], 'y.')
plt.title("Putting centroids randomly (green and yellow)")
plt.show()

# ------

epochs = 5         # number of epochs

for i in range(0, epochs):
    dg = np.sum( (data - mug.T)**2, axis=1 )
    dy = np.sum( (data - muy.T)**2, axis=1 )   # d means distence from centroid to the point

    #idx = dg > dy
    #print(idx)

    cluster_g = data
    cluster_y = data

    cluster_g = np.delete(cluster_g, dg > dy, axis=0 )    # delete from dataset points which are more close to another centroid
    cluster_y = np.delete(cluster_y, dg < dy, axis=0 )
    """
    for key, val in enumerate(idx):
        if val:
            cluster_y = np.delete(cluster_y, data[key], axis=0 )
        else:
            cluster_g = np.delete(cluster_g, data[key], axis=0 )
    """

    x_g = cluster_g[:, 0]
    y_g = cluster_g[:, 1]
    x_y = cluster_y[:, 0]
    y_y = cluster_y[:, 1]
    plt.plot(x_g, y_g, 'gx')
    plt.plot(x_y, y_y, 'yx')
    plt.plot(mug[0], mug[1], 'g.')
    plt.plot(muy[0], muy[1], 'y.')
    plt.title(f"Epoch No_{i+1}")
    plt.show()

    mug = np.mean(cluster_g, axis=0, keepdims=1).T   # setting up new coordinates
    muy = np.mean(cluster_y, axis=0, keepdims=1).T

