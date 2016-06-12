import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from SVM import X_train_std, plot_decision_regions, y_train, X_combined_std, y_combined

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)

tree.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=tree, test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.suptitle('Decision Tree')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

# Pour pouvoir voir l'arbre qui s'est construit, utiliser le programmme GraphViz
from sklearn.tree import export_graphviz

export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])
