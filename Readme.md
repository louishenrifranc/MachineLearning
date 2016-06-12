# Resume de chaque fichier

* __BatchGradient__ : Implementation d'un Adaptative Linear Neutron : c'est à dire plusieurs inputs, qui sont coefficienté pour donner un seul output. Puis passage dans la fonction sigmoid pour passer d'une valeur réelle à une valeur continue. Puis si valeur > 0, alors le label 1, sinon 0. Pour mettre a jour les poids, on utilise la méthode du gradient conjugué. Pour chaque modification d'un poids (un w_j), on est obligé d'itérer sur tous les samples fournies dans les données initiales, ce qui est couteux... Une autre méthode consistant à mettre a jour les poids juste avec un sample, en itérant sample après sample.... (mini batch je crois)
Dans ce programme on affiche aussi le cout a chaque itération, suivant le prametre n dans la formule du cout. On voit aussi la nécéssité de standardiser nos vecteurs

# Prise en main de scikit
* __ScikitLearn__ : Centré, et réduire des données. Séparé les données, les re-rassembler.Utilisation du Perceptron de la bibliothèque scikit. On affiche aussi notre hyperplan séparateur de manière lisible.

# Classifiers

* __DecisionTree__ : Utilisation de l'algorithme de Decision Tree, pas besoin de standardiser les données, utilisation d'une mesure de l'enthropie, dans notre cas on choisit la mesure d'enthropie. max_depth indique la profondeur maximal de notre arbre. Cette arbre de décision peut être exporte au format .dot ce qui est fait a la fin du fichier.

* __KernelSVM__ : Problème des données qui ne peuvent etre séparé linéairement. On a besoin de les projeter dans un plan de dimension supérieure. On utilise donc le Kernel SVM pour projeter nos données dans un hyperplan de dimension supérieur, ou il sera possible les séparer. On choisit d'utiliser le Gaussien Kernel (rbf). On plotte le résultat de notre séparation dans ce plan

* __RandomForest__ : Principe : combiner des algorithmes d'apprentissages faibles. Principe de l'algorithme :
    - On récupère un ensemble de donnée tirés aléatoirement avec remise dans notre ensemble de départ. On construit un arbre de décision pour ces données. On fait ca 1000 à 2000 fois, puis avec chaque weak decision tree, on utilise la technique de Majority Voting pour obtenir une sortie.
Les données n'ont pas besoin d'etre standardiser. 

* __LogisticRegression__ : RAS

* __SVM__ : Methode de la SVM, on cherche a avoir la plus grande bordure possible entre les données. Math behind on a paper. 

# Selection des données

* __TestSet__ : Selection des features les plus importantes en utilisant la méthode de  RandomForest

* __MissingData__ : 
    - Remplacer des données manquantes, ou des NaN. 
    - Encodage One Hot : Remplacer des données non numériques par des données numériques (plusieurs méthodes)

* __LinearDiscriminantAnalysis__ : Faire ressortir les données les plus features les plus "discriminantes", c'est a dire celle qui apporte le plus d'informations concernant la le label.

* __PrincipalComponentAnalysis__ : Projection des données dans un nouvel espace de dimension plus petite. Implementer a la main (matrice de covariance, puis recherche des valeurs propres), puis en utilisant la bibliotheque scikit.
A noter que comme on change de dimension, il n'est pas possible de retrouver quelles étaient les features les plus importantes.

