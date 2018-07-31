import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy import array
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from itertools import cycle
X=np.array([[4,6],
[5,8],
[8,4],
[1,2],
[6,1],
[2,10],
[6,7],
[3,9],
[8,4],
[2,3],
[7,8],
[5,4],
[1,9],
[9,2],
[5,10],
[4,7],
[7,9],
[8,7],
[1,5],
[3,6],
[5,1],
[2,8],
[9,5],
[4,3],
[4,4],
[5,8],
[8,9],
[4,2],
[3,8],
[2,5]])

af=AffinityPropagation(preference=-50).fit(X)
cluster_centers_indices=af.cluster_centers_indices_
labels=af.labels_
n_clusters_=len(cluster_centers_indices)
print ('Estimated no. of clusters %d'% n_clusters_)
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + 'o')
plt.title('Affinity Propogation')
plt.show()













