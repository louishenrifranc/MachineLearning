import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

np.random.seed(0)
x_xor = np.random.randn(200, 2)

y_xor = np.logical_xor(x_xor[:, 0] > 0, x_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(x_xor[y_xor == 1, 0],
            x_xor[y_xor == 1, 1],
            c='b', marker='x',
            label='1')
plt.scatter(x_xor[y_xor == -1, 0],
            x_xor[y_xor == -1, 1],
            c='r',
            marker='s',
            label='-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=1000.0)  # Plus gamma augmente plus on a risque d'overfitting
svm.fit(x_xor, y_xor)
from ScikitLearn import plot_decision_regions

plt.suptitle('Kernel')

plot_decision_regions(x_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()
