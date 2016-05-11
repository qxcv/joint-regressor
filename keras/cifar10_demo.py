#!/usr/bin/env python2

"""cifar10 demo based on the basic examples."""

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import numpy as np

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


def one_hot(vec):
    rv = np.zeros((vec.size, 10))
    rv[np.arange(vec.size), vec.flatten()] = 1
    return rv


def calc_mean_pixel(input_data):
    return input_data.astype('float32').mean(axis=(0, 2, 3)).reshape((1, 3, 1, 1))


def rescale(input_data, mean_pixel):
    casted = input_data.astype('float32')
    return (casted - mean_pixel) / 255.0


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    Y_train = one_hot(y_train)
    Y_test = one_hot(y_test)
    mp = calc_mean_pixel(X_train)
    X_train_scaled = rescale(X_train, mp)
    X_test_scaled = rescale(X_test, mp)

    try:
        model.fit(X_train_scaled, Y_train, batch_size=128, nb_epoch=10,
                validation_data=(X_test_scaled, Y_test))
    finally:
        model.save_weights('cifar10_1000_epochs.h5')
