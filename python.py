import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.construct import random
from sklearn import cluster
from sklearn.cluster import KMeans

df = pd.read_csv('mang.csv')
#print(df.head(3))
x = df.iloc[:, [0,1,2,3]].values

#sq = []
#for i in range(1, 11):
# kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter =300,
#                  n_init =10, random_state =0)
#  kmeans.fit(x)
#  sq.append(kmeans.inertia_)
#plot the elbow graph
#plt.plot(range(1,11), sq)
#plt.show()

kmeans = KMeans(n_clusters=3, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_prdeict(x)

plt.scatter(x[y_kmeans==0, 0], x[y_kmeans==0,1], s=100, c='red', label='Cluster 0')  
plt.scatter(x[y_kmeans==1, 0], x[y_kmeans==1,1], s=100, c='blue', label='Cluster 1')  
plt.scatter(x[y_kmeans==2, 0], x[y_kmeans==2,1], s=100, c='Green', label='Cluster 2')  


plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,1], s=300, c='yellow', label ='Centroids')
plt.title('Cluster of Mangoes')
plt.xlabel('Cluster')
plt.ylabel('Cluster Center')
plt.show

