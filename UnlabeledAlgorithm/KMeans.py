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


#
# La méthode en bas (elbow model) permet de déterminer la meilleure valeur de k
#


X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

km = KMeans(n_clusters=3,
            init='random',
            n_init=10,  # On repart a l'étape 1 de la séléction initiale aléatoire des centroides 10 fois
            max_iter=300,
            tol=1e-04,  # Tolérance de changement..
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

## Elbow model :
#   Principe :
# Permet de déterminer le nombre de clusters optimal, en fesant tourner l'algorithmes des Kmeans(++)
# plusieurs fois, et en récupérant la distortion à la fin
# L'endroit de la courbe ou la distortion diminue plus faiblement correspond, sur l'axe des abscisses, au nombre de
# clusters qu'il faut prendre

distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
                init='k-means++',  # On utilise k-means++
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()

## Silhouette plots
# Principe
# 1. Pour tous les points i dans un cluster, calculer sa distance à tous les autres points du cluster: a_i
# 2. Pour ce point i calculer sa distance avec tous les points appartenant au cluster le plus près de lui : b_i
# 3. Ainsi le score pour un point est de (b_i - a_i)/ max(a_i,b_i)
# Plus la valeur est près de 1, plus le point est bien placé.
#
# On va tracer une courbe de ces scores pour tous les points. Le but est que le maximum de points est une valeur proche
# de 1. Sur l'axe des ordonnées, tous les points des clusters, triés par cluster. Sur l'axe des absisses le score de
# tous ces points.


# Un mauvais clustering se remarquerait par la différence de taille et de score des différents clusters.

cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(i / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
             edgecolor='none', color=color)

    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color="red", linestyle="--")

plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.show()
# Bon cluster
