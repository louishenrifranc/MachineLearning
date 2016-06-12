import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/'
                 'housing/housing.data',
                 header=None,
                 sep='\s+')

df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# On cherche a modéliser plusieurs courbes de plusieurs degré différents
# Pour savoir laquelle s'adapte le mieux à notre modèle
X = df[['LSTAT']].values
y = df['MEDV'].values


regr = LinearRegression()

# Pour augmenter le nombre de dimensions, il suffit de modifier les données initiales pour notre regression, en
# éffectuant des combinaisons de features. On le passe ensuite a notre regression pour l'entrainement.
# A noter qu'il faut modifier la dimension des données d'entrainement mais aussi des données de validation !

# Génère une nouvelle matrice contenant toutes les combinaisons de features de degré inférieure ou égale à 2
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Fit les différentes régréssions
# 1 D
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

# 2 D
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

# 3 D
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# On affiche nos points et les courbes obtenues

# Pour afficher des points
plt.scatter(X, y, label='training points', color='lightgray')

# Pour afficher une courbe de Y en fonction de X
plt.plot(X_fit, y_lin_fit,
         label='linear (d=1), $R^2=%.2f$' % linear_r2,  # les informations du graphes
         color='blue',  # couleur
         lw=2,  # largeur du point
         linestyle=':')  # Comment on représente le point

# Pour afficher
plt.plot(X_fit, y_quad_fit,
         label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
         color='red',
         lw=2,
         linestyle='-')
plt.plot(X_fit, y_cubic_fit,
         label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
         color='green',
         lw=2,
         linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc='upper right')
plt.show()
