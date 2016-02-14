"""Models used by train.py"""

from keras.models import Graph, Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
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


def vggnet16_regressor_model(input_shape, num_outputs, solver, init):
    """Just build a standard VGGNet16 model"""
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

    model.add(Dense(num_outputs, init=init))

    # MAE is really just L1 loss, except we're averaging it because that might be
    # easier to interpret (?); I hadn't really considered that.
    model.compile(loss='mae', optimizer=solver)

    return model


def vggnet16_joint_model(
        input_shape, regressor_outputs, biposelet_classes, solver, init
    ):
    """As above, but this time we have a classifier output as well"""
    model = Graph()
    model.add_input(input_shape=input_shape, name='input')
    # conv1
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='input', name='pad1')
    model.add_node(Convolution2D(64, 3, 3, init=init, activation='relu'), input='pad1', name='conv1')
    # conv2
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='conv1', name='pad2')
    model.add_node(Convolution2D(64, 3, 3, init=init, activation='relu'), input='pad2', name='conv2')
    # pool1
    model.add_node(MaxPooling2D(pool_size=(2, 2)), input='conv2', name='pool1')

    # conv3
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='pool1', name='pad3')
    model.add_node(Convolution2D(128, 3, 3, init=init, activation='relu'), input='pad3', name='conv3')
    # conv4
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='conv3', name='pad4')
    model.add_node(Convolution2D(128, 3, 3, init=init, activation='relu'), input='pad4', name='conv4')
    # pool2
    model.add_node(MaxPooling2D(pool_size=(2, 2)), input='conv4', name='pool2')

    # conv5
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='pool2', name='pad5')
    model.add_node(Convolution2D(256, 3, 3, init=init, activation='relu'), input='pad5', name='conv5')
    # conv6
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='conv5', name='pad6')
    model.add_node(Convolution2D(256, 3, 3, init=init, activation='relu'), input='pad6', name='conv6')
    # conv7
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='conv6', name='pad7')
    model.add_node(Convolution2D(256, 3, 3, init=init, activation='relu'), input='pad7', name='conv7')
    # pool3
    model.add_node(MaxPooling2D(pool_size=(2, 2)), input='conv7', name='pool3')

    # conv8
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='pool3', name='pad8')
    model.add_node(Convolution2D(512, 3, 3, init=init, activation='relu'), input='pad8', name='conv8')
    # conv9
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='conv8', name='pad9')
    model.add_node(Convolution2D(512, 3, 3, init=init, activation='relu'), input='pad9', name='conv9')
    # conv10
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='conv9', name='pad10')
    model.add_node(Convolution2D(512, 3, 3, init=init, activation='relu'), input='pad10', name='conv10')
    # pool4
    model.add_node(MaxPooling2D(pool_size=(2, 2)), input='conv10', name='pool4')

    # conv11
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='pool4', name='pad11')
    model.add_node(Convolution2D(512, 3, 3, init=init, activation='relu'), input='pad11', name='conv11')
    # conv12
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='conv11', name='pad12')
    model.add_node(Convolution2D(512, 3, 3, init=init, activation='relu'), input='pad12', name='conv12')
    # conv13
    model.add_node(ZeroPadding2D(padding=(1, 1), dim_ordering='th'), input='conv12', name='pad13')
    model.add_node(Convolution2D(512, 3, 3, init=init, activation='relu'), input='pad13', name='conv13')
    # pool5
    model.add_node(MaxPooling2D(pool_size=(2, 2)), input='conv13', name='pool5')
    model.add_node(Flatten(), input='pool5', name='flat')

    # fc1
    model.add_node(Dense(4096, init=init, activation='relu'), input='flat', name='fc1')
    # drop2
    model.add_node(Dropout(0.5), input='fc1', name='drop1')

    # fc2
    model.add_node(Dense(4096, init=init, activation='relu'), input='drop2', name='fc2')
    # drop3
    model.add_node(Dropout(0.5), input='fc2', name='drop2')

    # Both come from drop3, and produce outputs which we will pass to different
    # loss layers
    model.add_node(Dense(regressor_outputs, init=init), input='drop2', name='fc_regr')
    model.add_node(Dense(biposelet_classes, init=init, activation='softmax'), input='drop2', name='fc_clas')

    model.add_output(input='fc_regr', name='out_regr')
    model.add_output(input='fc_clas', name='out_clas')

    model.compile(
        optimizer=solver, loss={
            'out_regr': 'mae',
            'out_clas': 'hinge'
        }
    )

    return model
