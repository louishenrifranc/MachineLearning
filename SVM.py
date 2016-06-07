import numpy as np
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

from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
y_pred = svm.predict(X_test_std)
print('Nombre d\'erreurs de la SVM %d' % (y_pred != y_test).sum())

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from ScikitLearn import plot_decision_regions, plt

plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.show()
