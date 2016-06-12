import os
import pickle

from sklearn.feature_extraction.text import HashingVectorizer

from Tokenizing import tokenizer

clf = pickle.load(open(os.path.join('movieclassifier', 'pkl_objects', 'classifier.pkl'), 'rb'))
x = "Very silly and stupid movie, Not a recommandation"
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2 ** 21,
                         preprocessor=None,
                         tokenizer=tokenizer)
X = vect.transform(x)
clf = clf.partial_fit(X, 0)

# Predicting
x = "This is a stupid movies,I will not recomend it"
label = {0: 'negative', 1: "positive"}
X = vect.transform(x)
print('Prediction %s\nProbability : %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_log_proba(X) * 100)))
