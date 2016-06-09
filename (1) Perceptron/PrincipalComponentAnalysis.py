import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
X, y = df_wine.iloc[:, 1:], df_wine.iloc[:, 0]

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

# 1 Standardiser
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.fit_transform(X_test)

# 2 Construire matrice de covariance
cov_mat = np.cov(X_train_std.T)

# 3. Trouver vecteurs et valeurs propres
valpropre, vecpropre = np.linalg.eig(cov_mat)

paire_propre = [(np.abs(valpropre[i]), vecpropre[:, i])
                for i in range(len(vecpropre))]
# 4. Trier vecteurs propres suivant valeurs propres
paire_propre.sort(reverse=True)

# 5. Former la matrice de transition en prenant ici deux features
np.hstack(paire_propre[0][1][:, np.newaxis], paire_propre[1, 1][:, np.newaxis])
