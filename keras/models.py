"""Models used by train.py"""

from keras.models import Graph, Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)


def make_conv_triple(model, channels, input_shape=None, **extra_args):
    zp_args = {}
    if input_shape is not None:
        zp_args['input_shape'] = input_shape
    layers = [
        ZeroPadding2D(padding=(1, 1), dim_ordering='th', **zp_args),
        Convolution2D(channels, 3, 3, activation='relu', **extra_args)
    ]
    for layer in layers:
        model.add(layer)


def vggnet16_base(input_shape, init):
    """Like vggnet16_regressor_model, but with no output layers and no
    compilation. Very useful for embedding in other models."""
    model = Sequential()
    make_conv_triple(model, 64, input_shape=input_shape, init=init)
    make_conv_triple(model, 64, init=init)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    make_conv_triple(model, 128, init=init)
    make_conv_triple(model, 128, init=init)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    make_conv_triple(model, 256, init=init)
    make_conv_triple(model, 256, init=init)
    make_conv_triple(model, 256, init=init)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    make_conv_triple(model, 512, init=init)
    make_conv_triple(model, 512, init=init)
    make_conv_triple(model, 512, init=init)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    make_conv_triple(model, 512, init=init)
    make_conv_triple(model, 512, init=init)
    make_conv_triple(model, 512, init=init)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(4096, activation='relu', init=init))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu', init=init))
    model.add(Dropout(0.5))

    return model


def vggnet16_regressor_model(input_shape, num_outputs, solver, init):
    """Just build a standard VGGNet16 model"""
    model = vggnet16_base(input_shape, init)
    model.add(Dense(num_outputs, init=init))
    # MAE is really just L1 loss, except we're averaging it because that might be
    # easier to interpret (?); I hadn't really considered that.
    model.compile(loss='mae', optimizer=solver)
    return model


def vggnet16_joint_reg_class(input_shape, regressor_outputs, solver, init):
    """As above, but this time we have a classifier output as well. At the
    moment, the classifier can only have one output (which will presumably
    correspond to whether not the patch represents the part it is looking for)."""
    model = Graph()
    model.add_input(input_shape=input_shape, name='images')
    base = vggnet16_base(input_shape, init)
    model.add_node(base, name='vgg16', input='images')
    model.add_node(Dense(regressor_outputs, init=init), input='vgg16', name='fc_regr')
    model.add_node(Dense(1, init=init, activation='sigmoid'), input='vgg16', name='fc_clas')
    model.add_output(input='fc_regr', name='joints')
    model.add_output(input='fc_clas', name='class')
    model.compile(
        optimizer=solver, loss={
            'joints': 'mae',
            'class': 'binary_crossentropy'
        }
    )
    return model
