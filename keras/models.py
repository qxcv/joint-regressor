"""Models used by train.py"""

import numpy as np

from keras.models import Graph, Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.utils.layer_utils import container_from_config

from utils import register_activation, convolution_softmax


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


def vgg16_twin_base(input_shape, init):
    """There will be two instances of this model created: one for flow and one
    for RGB data."""
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

    return model


def vgg16_twin_final(input_shape, init):
    """This is the second stage, after the two first stages have been joined
    together."""
    model = Sequential()
    make_conv_triple(model, 512, input_shape=input_shape, init=init)
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


def vggnet16_joint_reg_class_flow(shapes, solver, init):
    """Extension of above model to include a parallel network for processing
    flow.

    :param shapes: Dictionary mapping input or output names to their sizes.
                   Should not include batch size or leading dimension in HDF5
                   file (e.g. shape of RGB data might be ``(3, 224, 224)``
                   rather than ``(None, 3, 224, 224)`` or ``(12091, 3, 224,
                   224)``).
    :param solver: Keras solver (e.g. SGD) instance.
    :param init: String describing weight initialiser (e.g. 'glorot_uniform')
    :return: Initialised ``keras.models.Graph``"""
    model = Graph()

    # RGB travels through a bunch of conv layers first
    rgb_shape = shapes['images']
    model.add_input(input_shape=rgb_shape, name='images')
    rgb_base = vgg16_twin_base(rgb_shape, init)
    model.add_node(rgb_base, name='rgb_conv', input='images')

    # Flow travels through a parallel set of layers with the same architecture
    # but different weights
    flow_shape = shapes['flow']
    model.add_input(input_shape=flow_shape, name='flow')
    flow_base = vgg16_twin_base(flow_shape, init)
    model.add_node(flow_base, name='flow_conv', input='flow')

    # The two streams are merged before the first 512-channel conv layer (after
    # max-pooling in the 256 node layer)
    rgb_out_shape = rgb_base.output_shape
    flow_out_shape = flow_base.output_shape
    assert rgb_out_shape == flow_out_shape
    assert len(rgb_out_shape) == 4 and rgb_out_shape[0] == None
    in_channels = rgb_out_shape[1] + flow_out_shape[1]
    shared_shape = (in_channels,) + rgb_out_shape[2:]
    shared_final = vgg16_twin_final(shared_shape, init)
    model.add_node(
        shared_final, inputs=['rgb_conv', 'flow_conv'],
        merge_mode='concat', concat_axis=1, name='shared_layers'
    )

    # Add regressor outputs
    reg_out_names = ['left', 'right', 'head']
    for name in reg_out_names:
        reg_outs, = shapes[name]
        rname = 'fc_regr_' + name
        model.add_node(Dense(reg_outs, init=init), input='shared_layers', name=rname)
        model.add_output(input=rname, name=name)

    # Add classifier outputs
    num_classes = 1+len(reg_out_names)
    assert len(shapes['class']) == 1 and shapes['class'][0] == num_classes
    model.add_node(Dense(num_classes, init=init, activation='softmax'), input='shared_layers', name='fc_clas')
    model.add_output(input='fc_clas', name='class')

    # Done!
    losses = {'class': 'categorical_crossentropy'}
    for name in reg_out_names:
        losses[name] = 'mae'

    model.compile(
        optimizer=solver, loss=losses
    )
    return model


def vggnet16_poselet_class_flow(shapes, solver, init):
    """Similar to above, but just predicitng poselet classes.

    :param shapes: Dictionary mapping input or output names to their sizes.
                   Should not include batch size or leading dimension in HDF5
                   file (e.g. shape of RGB data might be ``(3, 224, 224)``
                   rather than ``(None, 3, 224, 224)`` or ``(12091, 3, 224,
                   224)``).
    :param solver: Keras solver (e.g. SGD) instance.
    :param init: String describing weight initialiser (e.g. 'glorot_uniform')
    :return: Initialised ``keras.models.Graph``"""
    model = Graph()

    # Just like the above model
    rgb_shape = shapes['images']
    model.add_input(input_shape=rgb_shape, name='images')
    rgb_base = vgg16_twin_base(rgb_shape, init)
    model.add_node(rgb_base, name='rgb_conv', input='images')
    flow_shape = shapes['flow']
    model.add_input(input_shape=flow_shape, name='flow')
    flow_base = vgg16_twin_base(flow_shape, init)
    model.add_node(flow_base, name='flow_conv', input='flow')
    rgb_out_shape = rgb_base.output_shape
    flow_out_shape = flow_base.output_shape
    in_channels = rgb_out_shape[1] + flow_out_shape[1]
    shared_shape = (in_channels,) + rgb_out_shape[2:]
    shared_final = vgg16_twin_final(shared_shape, init)
    model.add_node(
        shared_final, inputs=['rgb_conv', 'flow_conv'],
        merge_mode='concat', concat_axis=1, name='shared_layers'
    )

    # Only predict poselet classes this time
    poselet_classes, = shapes['poselet']
    model.add_node(Dense(poselet_classes, init=init, activation='softmax'),
                   input='shared_layers', name='fc_pslt')
    model.add_output(input='fc_pslt', name='poselet')
    losses = {'poselet': 'categorical_crossentropy'}

    model.compile(
        optimizer=solver, loss=losses
    )
    return model

def repr_layer(layer):
    """Pretty name for a Keras layer"""
    conf = layer.get_config()
    name = conf['name']
    input_shape = layer.input_shape
    output_shape = layer.output_shape
    return '{} ({}->{})'.format(name, input_shape, output_shape)

def dense_to_conv(dense_layer, conv_shape, **conv_args):
    """Convert a Dense layer to a 1-channel Convolution2D; will take input from
    a Convolution2D of size conv_shape (not including None)."""
    assert len(dense_layer.output_shape) == 2
    assert len(conv_shape) == 3

    num_outputs = dense_layer.output_shape[1]
    weight_shape = (num_outputs,) + conv_shape
    dense_weights = dense_layer.get_weights()
    assert len(dense_weights) == 2

    print('Old shape: {}, new shape: {}'.format(
        dense_weights[0].shape, weight_shape
    ))

    conv_weights = [dense_weights[0].reshape(weight_shape), dense_weights[1]]

    rv = Convolution2D(
        num_outputs, conv_shape[1], conv_shape[2], weights=conv_weights,
        **conv_args
    )

    return rv

def upgrade_sequential(old_model):
    """Upgrade a ``keras.models.Sequential`` instance to be fully
    convolutional.

    :param old_model: The old (not-fully-convolutional) ``Sequential`` model.
    :returns:  A new ``Sequential`` model with the same weights as the old one,
               but with flattening layers removed and dense layers replaced by
               convolutions."""
    assert isinstance(old_model, Sequential), "only works on sequences"
    rv = Sequential()
    all_layers = list(old_model.layers)

    while all_layers:
        next_layer = all_layers.pop(0)
        if isinstance(next_layer, Flatten):
            assert all_layers, "flatten must be followed by layer"
            next_dense = all_layers.pop(0)
            assert isinstance(next_dense, Dense), \
                "flatten must be followed by dense"
            # Upgrade the dense layer to a convolution with the same filter
            # size as the input
            assert len(next_layer.input_shape) == 4, "input must be conv"
            new_conv = dense_to_conv(next_dense, next_layer.input_shape[1:])
            rv.add(new_conv)

            print('Converted {} via {} to {}'.format(
                repr_layer(next_dense), repr_layer(next_layer),
                repr_layer(new_conv)
            ))
        elif isinstance(next_layer, Dense):
            assert len(next_layer.input_shape) == 2, "input must be conv"
            new_conv = dense_to_conv(
                next_layer, (next_layer.input_shape[1], 1, 1)
            )
            rv.add(new_conv)

            print('Converted {} to {}'.format(
                repr_layer(next_layer), repr_layer(new_conv)
            ))
        else:
            next_layer_copy = container_from_config(
                next_layer.get_config()
            )
            rv.add(next_layer_copy)
            next_layer_weights = next_layer.get_weights()
            next_layer_copy.set_weights(next_layer_weights)

            # Just make sure that weights really are the same
            new_weights = rv.layers[-1].get_weights()
            assert len(new_weights) == len(next_layer_weights)
            assert all(
                np.all(w1 == w2)
                for w1, w2 in zip(new_weights, next_layer_weights)
            )

            print('Added {} to model unchanged (was {})'.format(
                repr_layer(next_layer_copy), repr_layer(next_layer)
            ))

    return rv

def upgrade_multipath_vggnet(old_model):
    # Make sure we have the right model
    node_names = {
        'rgb_conv', 'flow_conv', 'shared_layers', 'fc_regr_left',
        'fc_regr_right', 'fc_regr_head', 'fc_clas'
    }
    assert set(old_model.nodes.keys()) == node_names

    rv = Graph()

    # Upgrade RGB path first (weights handled by upgrade_sequential)
    print('Upgrading RGB path')
    rgb_shape = old_model.inputs['images'].input_shape[1:]
    rv.add_input(
        input_shape=rgb_shape, name='images'
    )
    upgraded_rgb_conv = upgrade_sequential(old_model.nodes['rgb_conv'])
    rv.add_node(upgraded_rgb_conv, input='images', name='rgb_conv')

    # Upgrade flow path
    print('Upgrading flow path')
    flow_shape = old_model.inputs['flow'].input_shape[1:]
    rv.add_input(
        input_shape=flow_shape, name='flow'
    )
    upgraded_flow_conv = upgrade_sequential(old_model.nodes['flow_conv'])
    rv.add_node(upgraded_flow_conv, input='flow', name='flow_conv')

    # Upgrade shared path
    print('Upgrading shared path')
    upgraded_share = upgrade_sequential(old_model.nodes['shared_layers'])
    rv.add_node(
        upgraded_share, inputs=['rgb_conv', 'flow_conv'], merge_mode='concat',
        concat_axis=1, name='shared_layers'
    )

    # Upgrade dense outputs (weights handled by dense_to_conv)
    print('Upgrading regression outputs')
    for out_name in {'left', 'right', 'head'}:
        dense_name = 'fc_regr_' + out_name
        old_dense = old_model.nodes[dense_name]
        new_dense = dense_to_conv(old_dense, (old_dense.input_shape[1], 1, 1))
        rv.add_node(new_dense, input='shared_layers', name=dense_name)
        rv.add_output(input=dense_name, name=out_name)

    print('Upgrading classification output')
    old_clas_dense = old_model.nodes['fc_clas']
    # We're using a custom activation, so we have to register it
    register_activation(convolution_softmax, 'convolution_softmax')
    new_clas_dense = dense_to_conv(
        old_clas_dense, (old_clas_dense.input_shape[1], 1, 1),
        activation='convolution_softmax'
    )
    rv.add_node(new_clas_dense, input='shared_layers', name='fc_clas')
    rv.add_output(input='fc_clas', name='class')

    # Plan is to save weights to HDF5 and architecture to JSON, then a small
    # wrapper can load the data for use from Matlab
    print('All done!')
    return rv
