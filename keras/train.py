#!/usr/bin/env python2

"""Trains a CNN using Keras. Includes a bunch of fancy stuff so that you don't
lose your work when you Ctrl + C."""

from argparse import ArgumentParser
from itertools import chain, islice
from multiprocessing import Process, Queue, Event

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
        ZeroPadding2D)

INIT='glorot_normal'
INPUT_SHAPE = (6, 224, 224)
NUM_OUTPUTS = 2 * 3  # Just predict left or right side (3 joints)


def make_conv_triple(model, channels, **extra_args):
    layers = [
        ZeroPadding2D(padding=(1, 1), dim_ordering='th'),
        Convolution2D(channels, 3, 3, init=INIT, **extra_args),
        Activation('relu')
    ]
    for layer in layers:
        model.add(layer)


def build_model():
    """Just build a standard VGGNet16 model"""
    model = Sequential()
    make_conv_triple(model, 64, input_shape=INPUT_SHAPE)
    make_conv_triple(model, 64)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    make_conv_triple(model, 128)
    make_conv_triple(model, 128)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    make_conv_triple(model, 256)
    make_conv_triple(model, 256)
    make_conv_triple(model, 256)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    make_conv_triple(model, 512)
    make_conv_triple(model, 512)
    make_conv_triple(model, 512)
    model.add(MaxPooling2D(pool_size=(2, 2)))

    make_conv_triple(model, 512)
    make_conv_triple(model, 512)
    make_conv_triple(model, 512)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(NUM_OUTPUTS))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # MAE is really just L1 loss, except we're averaging it because that might be
    # easier to interpret (?); I hadn't really considered that.
    model.compile(loss='mae', optimizer=sgd)


parser = ArgumentParser(description="Train a CNN to regress joints")
parser.add_argument(
    '--train-h5', dest='train_h5', type=str,
    help='h5 file in which training samples are stored'
)
parser.add_argument(
    '--val-h5', dest='val_h5', type=str,
    help='h5 file in which validation samples are stored'
)
parser.add_argument(
    '--queued-batches', dest='queued_batches', type=int, default=3,
    help='number of unused batches stored in processing queue (in memory)'
)
parser.add_argument(
    '--batch-size', dest='batch_size', type=int, default=128,
    help='batch size for both training (backprop) and validation'
)
parser.add_argument(
    '--checkpoint-dir', dest='checkpoint_dir', type=str,
    help='directory in which to store checkpoint files'
)

# Sketch of data loading system:
# Since we avoid introducing bias into which augmented pairs are sorted into
# which file (so a given augmentation of a given pair has an equal chance of
# being in *any* file, IIRC), we really only need to load up a HDF5 at a time
# and shuffle it before doing a backwards pass with it. Unfortunately,
# fit_with_generator in the standard library does not let you supply a
# generator for the validation set (!!), and I really like having a validation
# set, so I'll have to rewrite the functionality of fit() myself :(

def h5_read_worker(h5_path, batch_size, out_queue, end_evt):
    """This function is designed to be run in a multiprocessing.Process, and
    communicate using a multiprocessing.Queue. It will just keep reading
    batches and pushing them (complete batches!) to the queue in a tight loop;
    obviously this will block as soon as the queue is full, at which point this
    process will wait until it can read again. Note that ``end_evt`` is an
    Event which should be set once the main thread wants to exit."""
    with h5py.File(h5_path, 'r') as fp:
        def index_gen():
            """Yields random indices into the dataset. Fetching data this way
            is slow, but it should be okay given that we're running this in a
            background process."""
            label_set = fp['/label']
            label_size = len(label_set)
            yield np.random.permutation(label_size)

        indices = index_gen()

        while True:
            # First, fetch a batch full of data
            sliced_batch_indices = list(islice(indices, batch_size))
            if not sliced_batch_indices:
                # We're at the end of an epoch
                batch = None
                indices = index_gen()
            else:
                batch_data = fp['/data'][batch_indices]
                batch_labels = fp['/label'][batch_indices]
                batch = (batch_data, batch_labels)

            # This is the push loop
            while True:
                if end_evt.is_set():
                    return
                out_queue.push(batch, timeout=0.05)


if __name__ == '__main__':
    args = parser.parse_args()
    end_event = Event()

    # Training data prefetch
    train_queue = Queue(args.queued_batches)
    train_args = (args.train_h5, args.batch_size, train_queue, end_event)
    train_worker = Process(target=h5_read_worker, args=train_args)

    # Validation data prefetch
    val_queue = Queue(args.queued_batches)
    val_args = (args.val_h5, args.batch_size, val_queue, end_event)
    val_worker = Process(target=h5_read_worker, args=val_args)

    # Go!
    train_worker.start()
    val_worker.start()

    # Now we can start the training loop!
    # TODO

    # Finally, clean everything up
    end_event.set()
    train_worker.join()
    val_worker.join()
