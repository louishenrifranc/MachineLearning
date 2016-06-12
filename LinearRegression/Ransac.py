import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

# On prend l'exemple des ventes de maisons
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'housing/housing.data',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
print('Dataset excerpt:\n\n', df.head())

X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

X = df[['RM']].values
y = df['MEDV'].values

# Random Sample Consensus
# Les données sont constitués de inliers et de outliers, c'est à dire des données dont la distribution peut etre
# expliquée
# par un ensemble de paramètres d'un modèle, et les données abérantes
# ALGORITHME
# # # itérateur := 0
# # # meilleur_modèle := aucun
# # # meilleur_ensemble_points := aucun
# # # meilleure_erreur := infini
# # # tant que itérateur < k
# # #     points_aléatoires := n valeurs choisies au hasard à partir des données
# # #     modèle_possible := paramètres du modèle correspondant aux points_aléatoires
# # #     ensemble_points := points_aléatoires
# # #
# # #     Pour chaque point des données pas dans points_aléatoires
# # #         si le point s'ajuste au modèle_possible avec une erreur inférieure à t
# # #             Ajouter un point à ensemble_points
# # #
# # #     si le nombre d'éléments dans ensemble_points est > d
# # #         (ce qui implique que nous avons peut-être trouvé un bon modèle,
# # #         on teste maintenant dans quelle mesure il est correct)
# # #         modèle_possible := paramètres du modèle réajusté à tous les points de ensemble_points
# # #         erreur := une mesure de la manière dont ces points correspondent au modèle_possible
# # #         si erreur < meilleure_erreur
# # #             (nous avons trouvé un modèle qui est mieux que tous les précédents,
# # #             le garder jusqu'à ce qu'un meilleur soit trouvé)
# # #             meilleur_modèle := modèle_possible
# # #             meilleur_ensemble_points := ensemble_points
# # #             meilleure_erreur := erreur
# # #
# # #     incrémention de l’itérateur
# # #
# # # retourne meilleur_modèle, meilleur_ensemble_points, meilleure_erreur

ransac = RANSACRegressor(LinearRegression(),
                         max_trials=1000,
                         min_samples=50,
                         residual_metric=lambda x: np.sum(np.abs(x), axis=1),  # distance vertical du point a la droite
                         residual_threshold=5.0,  # pr que le point soit ajouté au inliers, dist(x,droite) < 5 unité
                         random_state=0)

ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_  # On récupère tous les points pris dans le meilleur modèle trouvé par l'inlier
outlier_mask = np.logical_not(inlier_mask)  # On récupère tous les points qui n'ont pas été pris

line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask],
            c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper left')
plt.show()
plt.close()
