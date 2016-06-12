import numpy as np
import pyprind
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from Tokenizing import tokenizer


# Ouvre une review une a la fois
def stream_docs(path):
    with open(path, 'r', encoding='ISO-8859-1') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label


# Test  de la fonction de stream de documents
# print(next(stream_docs(path='./movie_data.csv')))


# Fonction qui recupere 'size' reviews, par la fonction stream_doc passé en paramètre
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


vect = HashingVectorizer(decode_error='ignore',
                         n_features=2 ** 21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')

pbar = pyprind.ProgBar(50000)
classes = np.array([0, 1])
for _ in range(8):
    X_train, y_train = get_minibatch(doc_stream, size=5000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
pbar.update()

X_val, y_val = get_minibatch(doc_stream, size=5000)
X_val = vect.transform(X_val)
print('Accuracy: %.3f' % clf.score(X_val, y_val))
# Ameliorer l'apprentissage
clf = clf.partial_fit(X_val, y_val)

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))
clf = clf.partial_fit(X_test, y_test)

# Serializing
import pickle, os

# 1. On cree un nouveau répertoire pour sauvegarder nos données
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

# On sérialise notre classifier ainsi que nos stop words
stop = stopwords.words('english')

pickle.dump(stop,
            open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
            protocol=4)
pickle.dump(clf,
            open(os.path.join(dest, 'classifier.pkl'), 'wb'),
            protocol=4)

# Predicting
x = "This is a stupid movies,I will not recomend it"
label = {0: 'negative', 1: "positive"}
X = vect.transform(x)
print('Prediction %s\nProbability : %.2f%%' % (label[clf.predict(X)[0]], np.max(clf.predict_log_proba(X) * 100)))
