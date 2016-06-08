import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#############################################################################
# Données sur le vin
df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                      'ml/machine-learning-databases/wine/wine.data',
                      header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values  # iloc comme [][] pour des DataFrames

#############################################################################

# Splitter les données
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)

# random_state : si l'on fixe ce parametre, on est sur d'avoir toujours les mêmes samples, sinon ils sont séparés
# aléatoirement


#############################################################################
# Normalisation des données

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

#############################################################################
# Standardisation des données (MEILLEUR)
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

feat_labels = df_wine.columns[1:]  # Recupère les colonnes sauf la première

#############################################################################
# Selection des features les plus importantes en utilisant un RandomForestClasifier
forest = RandomForestClassifier(n_estimators=10000,
                                random_state=0,
                                n_jobs=-1)

forest.fit(X_train_std, y_train)
importances = forest.feature_importances_  # Importance de chaque paramètre
print('importances :', importances)
indices = np.argsort(importances)[::-1]  # indices des features triés

for f in range(X_train_std.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

# Plotter un diagramme d'importance des features
plt.title('Feature Importances')
plt.bar(range(X_train_std.shape[1]),
        importances[indices],
        color='lightblue',
        align='center')

plt.xticks(range(X_train_std.shape[1]),
           feat_labels[indices], rotation=90)
plt.xlim([-1, X_train_std.shape[1]])
plt.show()

X_selected = forest.transform(X_train_std, threshold=0.15)
X_selected.shape
