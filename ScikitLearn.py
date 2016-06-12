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

# Learning the weights
from sklearn.linear_model import Perceptron

ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# Verifying prediction
y_pred = ppn.predict(X_test_std)
print('Number of error %d' % (y_test != y_pred).sum())  # Nombre très élevé j'abandonne les perceptrons...
# Je m'etais trompé, je passais au perceptron X_train au lieu de X_train_std





X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.suptitle('Perceptron')

# plt.tight_layout()
# plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
plt.show()
