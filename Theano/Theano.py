import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,

                               '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte' % kind)

    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


X_train, y_train = load_mnist('mnist', kind='train')
X_test, y_test = load_mnist('mnist', kind='t10k')

import theano

theano.config.floatX = 'float32'  # Opti pour quand on utilise le GPU

# On caste l'image en 32 bytes
X_train = X_train.astype(theano.config.floatX)
X_test = X_test.astype(theano.config.floatX)

# Convertir
from keras.utils import np_utils

print("First three labels\n", y_train[:3])

y_train_one = np_utils.to_categorical(y_train)
print('FIrst three labels after One Hot Encoder\n', y_train_one[:3])

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

np.random.seed(1)

model = Sequential()
model.add(Dense(input_dim=X_train.shape[1],
                output_dim=50,
                init='uniform',
                activation='tanh'
                ))

model.add(Dense(input_dim=50,
                output_dim=50,
                init='uniform',
                activation='tanh'
                ))

model.add(Dense(input_dim=50,
                output_dim=y_train_one.shape[1],
                init='uniform',
                activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(X_train,
          y_train_one,
          nb_epoch=50,
          batch_size=300,
          verbose=1,
          validation_split=0.1,
          show_accuracy=True)
