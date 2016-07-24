import os
import os
import struct
import numpy as np
from numpy import loadtxt
import pandas as pd
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import conda

df = pd.read_csv('ou.out',
                 header=None,
                 sep=' ')
X = df.loc[:, :8].values
y = df.loc[:, 8:].values
df.columns = ["(D-600)/20000.0", "angleScalD", "angleCrossD", "vScalD/1000.0",
              "vCrossD/1000.0", "nextDirScalD", "nextDirCrossD", "(distanceCP1CP2 - 1200.0) / 10000.0",
              "(gameAction.deltaAngle*180/PI+18.0)/36.0", "gameAction.thrust/200.0"]

# Get correlation matrix as a heat max
cols = df.shape[1]
cm = np.corrcoef(df, rowvar=0)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols,
                 xticklabels=cols)
plt.show()

from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=0)
