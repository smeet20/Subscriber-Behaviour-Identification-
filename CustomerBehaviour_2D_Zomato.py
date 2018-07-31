import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from numpy import array
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X=np.array([[30,6],[28,7],[24,8],[5,2],[3,4],[11,3],[14,3],[10,5],[12,6],[16,9]
            ,[17,5],[6,4],[10,3],[11,1],[9,7],[14,8],[21,6],[2,6],[5,5],[6,10],
            [7,9],[21,4],[14,9],[24,10],[2,3],[2,6],[4,9],[18,10],[28,7],[13,6]
            ,[27,7],[20,2],[11,4],[28,8],[23,2],[21,1],[29,7],[23,7],[10,1],
            [14,5],[15,6],[1,6],[28,7],[8,10],[12,9],[28,3],[7,7],
[26,10],
[5,1],
[26,6],
[18,9],
[9,4],
[4,8],
[1,7],
[1,7],
[23,9],
[23,4],
[6,4],
[27,7],
[19,2],
[24,5],
[11,6],
[22,4],
[2,3],
[19,9],
[20,10],
[18,8],
[17,5],
[5,9],
[7,3],
[19,3],
[4,3],
[18,8],
[14,3],
[8,4],
[16,8],
[7,1],
[13,8],
[6,4],
[27,4],
[25,3],
[7,1],
[15,9],
[18,5],
[21,8],
[13,7],
[16,9],
[21,2],
[26,10],
[29,1],
[25,9],
[3,3],
[29,5],
[5,4],
[17,7],
[29,1],
[1,3],
[6,9],
[28,7]])

kmeans = KMeans(n_clusters=7,n_init=100,max_iter=3000)
# Fitting with inputs
kmeans = kmeans.fit(X)
# Predicting the clusters
labels = kmeans.predict(X)
C = kmeans.cluster_centers_
plt.scatter(X[:,0],X[:,1])
plt.scatter(C[:,0],C[:,1],marker='*', c='#050505', s=150)
#fig=plt.figure()
#fig.suptitle('Food')
plt.xlabel('Time')
plt.ylabel('Frequency')
print(C)
LABEL_COLOR_MAP = {0 : 'b',
                   1 : 'g',
                   2 : 'r',
                   3 : 'c',
                   4 : 'm',
                   5 : 'y',
                   6 : 'k'
                   }

label_color = [LABEL_COLOR_MAP[l] for l in labels]

plt.scatter( X[:, 0], X[:, 1],c=label_color,s=50)
plt.show()
