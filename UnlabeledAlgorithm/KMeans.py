import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Principe de l'algorithme
# I : ensemble de départ(n points)
# k : le nombre de clusters voulues (paramètre de départ)
#
#: Debut:
#       1. On séléctionne K points dans notre ensemble de départ, qui deviendront les centroides
#       2. Pour tous les n points, on les met dans le cluster le plus proche, on calcule la distance euclidienne
#  avec tous les centroides, et on ajoute le point dans le cluster de ce centroide le plus près
#       3. On calcule la distance de tous les points a leur centroide. c'est notre fonction cout. On cherche à la
#  minimiser.
#       4. Les nouveau centroides deviennent les centres de gravités des clusters. Puis on retourne à l'étape 2
# Et cela tant que, soit aucun point n'a changé de cluster, soit on a mis un nombre d'itérations max.
#  Convergence assuré de l'algorithme.


X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)
y_km = km.fit_predict(X)

# Distortion : somme des distances de tous les points au centroid du cluster auquels ils appartiennent
distortion = km.inertia_
print("Distortion du modele : ", distortion)

# On plot le résultat obtenue
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50,
            c='lightgreen',
            marker='s',
            label='cluster 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50,
            c='orange',
            marker='o',
            label='cluster 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50,
            c='lightblue',
            marker='v',
            label='cluster 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='centroids')
plt.legend()
plt.grid()
plt.show()
