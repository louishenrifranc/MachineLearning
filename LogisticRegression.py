from sklearn import datasets

# On peut aussi recuperer les données sur les fleurs avec la librairie scikitlearn
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# Splitted dataset
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing dataset (centre, reduite)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)  # calcule la moyenne et l'ecart type uniquement
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)
print('Nombre d\'erreur pour la logistic Regression %d' % (y_pred != y_test).sum())
