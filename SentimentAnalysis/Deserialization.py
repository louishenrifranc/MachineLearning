import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from Tokenizing import tokenizer

clf = pickle.load(open(os.path.join('movieclassifier', 'pkl_objects', 'classifier.pkl'), 'rb'))
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2 ** 21,
                         preprocessor=None,
                         tokenizer=tokenizer)

# Predicting
x = "This is a stupid movies,I will not recomend it"
label = {0: 'negative', 1: "positive"}
X = vect.transform(x)
print('Prediction %s\nProbability : %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_log_proba(X) * 100)))
