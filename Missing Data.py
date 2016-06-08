from io import StringIO

import pandas as pd

# -----------------------------------------
# Premier Exemple
# Remplacer des données manquantes
# -----------------------------------------
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
# On trasnforme des donnees csv avec indice manquantes en structure DataFrame
df = pd.read_csv(StringIO(csv_data))
print(df)
# Obtenir le nombre de NaN par ligne
df.isnull().sum()

# On peut supprimer ces NaN valeurs, ce qui est déconseillé ...
# On peut aussi les transformer, en les remplacant par la moyenne de leur colonne respective
from sklearn.preprocessing import Imputer

# strategy = median, mean and most_frequent; axis = 0(colomme), 1 (ligne)
imr = Imputer(missing_values='NaN', strategy='median', axis=0)

# Apprend les valeurs...
imr = imr.fit(df)

# Remplace les valeurs
imputed_value = imr.transform(df.values)
print(imputed_value)

# -----------------------------------------
# Deuxieme Exemple
# Remplacer des données non numériques
# -----------------------------------------


df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
# 1 ere solution : Mapper les valeurs
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping)
print('Mapping:\n', df)
