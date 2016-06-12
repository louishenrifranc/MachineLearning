import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

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
W = np.hstack((paire_propre[0][1][:, np.newaxis], paire_propre[1][1][:, np.newaxis]))

# 6. Passer dans la nouvelle base
X_train_nbase = X_train_std.dot(W)

############################################################################
# On peut aussi utiliser la librairie Skicitlearn

# Permet d'afficher la variance de chaque feature (et donc son importance)
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
plt.bar(range(1, 14), pca.explained_variance_ratio_, alpha=0.5, align='center')
plt.step(range(1, 14), np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

# On choisit de n'en séléctionner que deux
pca = PCA(n_components=2)
lr = LogisticRegression()


X_test_pca = pca.transform(X_test_std)
lr.fit(X_train_pca, y_train)
y_pred = lr.predict(X_test_pca)
print('Nombre d\'erreur pour la logistic Regression %d' % (y_pred != y_test).sum())

# Rappel que j'avais déja oublié, on ne peut savoir quelles sont les principales composantes puisqu'on a completement changé de dimensions
# PCA doesn't eliminate dimensions and keeps others from the original data. It transforms your data in a number of dimensions whose data are completely different from the original ones
