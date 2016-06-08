import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from SVM import X_train_std, plot_decision_regions, y_train, X_combined_std, y_combined

forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10,
                                random_state=1,
                                n_jobs=2)
forest.fit(X_train_std, y_train)

plot_decision_regions(X_combined_std, y_combined,
                      classifier=forest, test_idx=range(105, 150))
plt.suptitle('Random Forest')
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.show()
