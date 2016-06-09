import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ScikitLearn import plot_decision_regions

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
    train_test_split(X, y, test_size=0.3)

# 1 Standardiser
ss = StandardScaler()
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.fit_transform(X_test)

# Scikit LDA, ressortir le vecteur des deux features les plus importantes
lda = LinearDiscriminantAnalysis(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)

# Effectue la logistic Regression
lr = LogisticRegression()
lr = lr.fit(X_train_lda, y_train)

# On transforme nos données de test (il ne faut pas fitter a nouveau, ici on transforme X_test_std en recuperant uniquement les features qui ont été choisies lors
# de l'entrainement
X_test_lda = lda.transform(X_test_std)
# plot_decision_regions(X_train_lda, y_train, lr)
#
plot_decision_regions(X_test_lda, y_test, lr)
plt.show()
