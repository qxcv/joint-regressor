#!/usr/bin/env python2

"""Make initial set of weights from downloaded ILSVRC weights. Useful when you
want to finetune from ILSVRC instead of starting from scratch."""

from argparse import ArgumentParser

import h5py

from keras.optimizers import SGD

import numpy as np

import models
from train import infer_sizes
from vggnet.upgrade_weights import upgrade_weights
from vggnet.vgg16_keras import VGG_16


parser = ArgumentParser()
parser.add_argument(
    'ilsvrc_weights', metavar='WEIGHTPATH', type=str,
    help='path to original ILSVRC16 weights'
)
parser.add_argument(
    'model_name', metavar='MODELNAME', type=str,
    help='name of model to convert to (should be VGG16-compatible)'
)
parser.add_argument(
    'sample_h5', metavar='DATAPATH', type=str,
    help='path to HDF5 file containing samples; used to figure out data dimensions'
)
parser.add_argument(
    'output_path', metavar='OUTPATH', type=str,
    help='where to write the generated weights'
)

if __name__ == '__main__':
    args = parser.parse_args()

    ds_shape = infer_sizes(args.sample_h5)

    print('Using loader %s' % args.model_name)
    model_to_load = getattr(models, args.model_name)
    solver = SGD()
    model = model_to_load(
        ds_shape, solver, 'glorot_normal'
    )

    print('Loading VGG16 ILSVRC model')
    ilsvrc_model = VGG_16(args.ilsvrc_weights)

    print('Upgrading sequential subset')
    flow_seq = model.nodes['flow_conv']
    rgb_seq = model.nodes['rgb_conv']
    upgrade_weights(flow_seq.layers, ilsvrc_model.layers)
    upgrade_weights(rgb_seq.layers, ilsvrc_model.layers)

    print('Upgrading shared end layers')
    front_layers = len(flow_seq.layers)
    assert front_layers == len(rgb_seq.layers), "Flow and RGB pipelines should be same length"
    back_ilsvrc_layers = ilsvrc_model.layers[front_layers:]
    back_seq = model.nodes['shared_layers']
    upgrade_weights(back_seq.layers, back_ilsvrc_layers)

    print('Saving to ' + args.output_path)
    model.save_weights(args.output_path, overwrite=True)
