import numpy as np
import matplotlib as plt

# I like deep learning
# I like NLP
# I enjoy flying
la = np.linalg
words = ["I", "like", "enjoy", "deep", "learning",
         "NLP", "flying", "."]

X = np.array([
    [0, 2, 1, 0, 0, 0, 0, 0],
    [2, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 0]])
U, s, Vh = la.svd(X, full_matrices=False)
print(X.shape)
print(U.shape)
print(s.shape)
print(Vh.shape)
for i in range(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])

from nltk.corpus import wordnet as wn
import nltk as nl

panda = wn.synset('panda.n.01')
hyper = lambda s: s.hypernyms()
print(list(panda.closure(hyper)))
