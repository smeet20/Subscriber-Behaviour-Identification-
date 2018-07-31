import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy import array
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X=np.array([[30,6,0],[28,7,0],[24,8,0],[5,2,0],[3,4,0],[11,3,0],[14,3,0],
           [10,5,0],[12,6,0],[16,9,0],[17,5,0],[6,4,0],[10,3,0],[11,1,0],
           [9,7,0],[14,8,0],[21,6,0],[2,6,0],[5,5,0],[6,10,0],[5,9,1],
           [1,7,1],[21,2,1],
[18,3,1],
[26,8,1],
[21,8,1],
[18,4,1],
[23,3,1],
[12,4,1],
[12,5,1],
[2,6,1],
[5,10,1],
[17,1,1],
[1,6,1],
[15,8,1],
[30,1,1],
[5,3,1],
[27,4,1],
[29,10,1],
[9,1,1],
[7,7,2],
[19,7,2],
[11,10,2],
[8,3,2],
[18,2,2],
[26,2,2],
[9,1,2],
[19,6,2],
[22,1,2],
[13,6,2],
[1,2,2],
[15,2,2],
[3,1,2],
[28,2,2],
[22,6,2],
[7,6,2],
[8,10,2],
[17,6,2],
[10,6,2],
[29,1,2]])
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(X[:,0],X[:,1],X[:,2])

kmeans = KMeans(n_clusters=3,n_init=100,max_iter=3000)
kmeans = kmeans.fit(X)
labels = kmeans.predict(X)
value=kmeans.labels_
for i in range(kmeans.n_clusters):
    C = kmeans.cluster_centers_
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=300)
print(C)
LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k',
                   2 : 'b',
                   3 : 'g'
                   }

label_color = [LABEL_COLOR_MAP[l] for l in labels]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter( X[:, 0], X[:, 1], X[:, 2],c=label_color,s=100)
plt.show()
