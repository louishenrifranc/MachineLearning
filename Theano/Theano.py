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

from sklearn import
