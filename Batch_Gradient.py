import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.
    """

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            #################################################
            # i : sample i
            # j : featur j
            # x_i^j
            # Formula to update weight : n * \sum(i) {y^i - y_output^i)*x_j^i = w_j
            #################################################
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            # Cout que l'on cherche a minimiser --> on le calcule pour pouvoir l'afficher après
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
# On ne prend que les label de deux types de fleurs (sinon on aurait pu tous les prendre
# mais il aurait fallu utiliser la méthode 1 vs all
y = df.iloc[0:100, 4].values
# print(y[45:53])

#  Mise en place de l'affichage des features
y = np.where(y == 'Iris-setosa', -1, 1)  # on transforme
X = df.iloc[0:100, [0, 2]].values  # on extrait seulement la premiere et la troisieme colonne des features

# On utilise maintenant la méthode de descente de gradient, avec deux pas différents pour voir les résultats n = 0.1 et n = 0.01
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

ada1 = AdalineGD(n_iter=50, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(n_iter=50, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')

# On voit bien qu'il ne faut pas prendre un trop large nombre pour le coefficient mu
# Trop grand => diverge
# Trop petit => convergence trop lente
# Il faut normaliser et centrer les variables !!
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada2 = AdalineGD(n_iter=50, eta=0.01).fit(X_std, y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
