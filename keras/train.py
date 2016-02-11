#!/usr/bin/env python2

"""Trains a CNN using Keras. Includes a bunch of fancy stuff so that you don't
lose your work when you Ctrl + C."""

from argparse import ArgumentParser
from itertools import chain, islice, repeat
from os import path
from multiprocessing import Process, Queue, Event
from sys import stdout

import h5py

from keras.models import Sequential
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
        ZeroPadding2D)
from keras.utils.generic_utils import Progbar

import numpy as np

INIT='glorot_normal'
INPUT_SHAPE = (6, 224, 224)
NUM_OUTPUTS = 2 * 3  # Just predict left or right side (3 joints)


def make_conv_triple(model, channels, **extra_args):
    layers = [
        ZeroPadding2D(padding=(1, 1), dim_ordering='th'),
        Convolution2D(channels, 3, 3, init=INIT, **extra_args),
        Activation('relu')
    ]
    if 'input_shape' in extra_args:
        # Skip the zero-padding in the first layer
        layers = layers[1:]
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
    model.add(Flatten())

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

# Mandatory arguments
parser.add_argument(
    'train_h5', metavar='TRAINDATA', type=str,
    help='h5 file in which training samples are stored'
)
parser.add_argument(
    'val_h5', metavar='VALDATA', type=str,
    help='h5 file in which validation samples are stored'
)
parser.add_argument(
    'checkpoint_dir', metavar='CHECKPOINTDIR', type=str,
    help='directory in which to store checkpoint files'
)

# Optargs
parser.add_argument(
    '--queued-batches', dest='queued_batches', type=int, default=3,
    help='number of unused batches stored in processing queue (in memory)'
)
parser.add_argument(
    '--batch-size', dest='batch_size', type=int, default=128,
    help='batch size for both training (backprop) and validation'
)
parser.add_argument(
    '--checkpoint-epochs', dest='checkpoint_epochs', type=int, default=5,
    help='training intervals to wait before writing a checkpoint file'
)
parser.add_argument(
    '--train-interval-batches', dest='train_interval_batches', type=int,
    default=1000, help='number of batches to train for between validation'
)


def h5_read_worker(h5_path, batch_size, out_queue, end_evt, mark_epochs):
    """This function is designed to be run in a multiprocessing.Process, and
    communicate using a multiprocessing.Queue. It will just keep reading
    batches and pushing them (complete batches!) to the queue in a tight loop;
    obviously this will block as soon as the queue is full, at which point this
    process will wait until it can read again. Note that ``end_evt`` is an
    Event which should be set once the main thread wants to exit.

    Note that at the end of each epoch, ``None`` will be pushed to the output
    queue iff mark_epochs is True. This notifies the training routine that it
    should perform validation or whatever it is training routines do
    nowadays."""
    with h5py.File(h5_path, 'r') as fp:
        def index_gen():
            """Yields random indices into the dataset. Fetching data this way
            is slow, but it should be okay given that we're running this in a
            background process."""
            label_set = fp['/label']
            label_size = len(label_set)
            for idx in np.random.permutation(label_size):
                yield idx

        if mark_epochs:
            # If mark_epochs is True, then we'll have to push None to the queue
            # at the end of each epoch (so infinite chaining is not helpful)
            indices = index_gen()
        else:
            # Otherwise, we can just keep fetching indices forever
            indices = chain.from_iterable(gen() for gen in repeat(index_gen))

        while True:
            # First, fetch a batch full of data
            batch_indices = list(islice(indices, batch_size))
            assert(batch_indices or mark_epochs)
            if mark_epochs and not batch_indices:
                indices = index_gen()
                batch = None
            else:
                # XXX: Since h5py is insane, it wants batch indices to be in
                # ascending order. I'm not sure whether I should do that and
                # then shuffle the data afterwards in memory, or fetch in the
                # order that I actually want the indices (assembling as I go).
                # Presumably h5py has good performance reasons for wanting
                # ascending indices, and those performance reasons likely have
                # something to do with the (very) slow act of reading from
                # disk, so I might reshuffle in memory. On the other hand,
                # paloalto only has SSDs, so perhaps whatever performance
                # consideration h5py was taking into account is not relevant?
                batch_data = fp['/data'][batch_indices]
                batch_labels = fp['/label'][batch_indices]
                batch = (batch_data, batch_labels)

            # This is the push loop
            while True:
                if end_evt.is_set():
                    return
                out_queue.push(batch, timeout=0.05)


def train(model, queue, iterations):
    """Perform a fixed number of training iterations."""
    print('Training for {} iterations'.format(iterations))
    p = Progbar(iterations)
    loss = 0.0
    p.update(0)

    for iteration in xrange(iterations):
        # First, get some data from the queue
        start_time = time()
        (X, y) = queue.get()
        fetch_time = time() - start_time

        # Next, do some backprop
        start_time = time()
        loss = model.train_on_batch(X, y)
        bp_time = time() - start_time

        # Finally, write some debugging output
        extra_info = [
            ('loss', loss),
            ('fetch', '{}s'.format(fetch_time)),
            ('backprop', '{}s'.format(bp_time))
        ]
        p.update(iteration + 1, extra_info)


def validate(model, queue):
    """Perform one epoch of validation."""
    print('Testing on validation set')
    batches = 0
    samples = 0
    weighted_loss = 0.0

    while True:
        batch = queue.get()
        if batch is None:
            break

        X, y = batch
        loss = model.test_on_batch(X, y)

        # Update stats
        batches += 1
        samples += len(X)
        weighted_loss += samples * loss

        if (batches % 100) == 0:
            print('{} validation batches tested'.format(batches))

    print(
        'Finished {} batches ({} samples); mean loss-per-sample {}'
        .format(weighted_loss / max(1, samples))
    )


def save(model, iteration_no, dest_dir):
    """Save the model to a checkpoint file."""
    filename = 'model-iter-{}-r{:06}.h5'.format(iteration_no, randint(1, 1e6-1))
    full_path = path.join(dest_dir, filename)
    # Note that save_weights will prompt if the file already exists. This is
    # why I've added a small random number to the end; hopefully this will
    # allow unattended optimisation runs to complete even when Keras decides
    # that it doesn't want to play nicely.
    print("Saving model to {}".format(full_path))
    model.save_weights(full_path)


if __name__ == '__main__':
    args = parser.parse_args()
    end_event = Event()

    # Training data prefetch
    train_queue = Queue(args.queued_batches)
    train_args = (
        args.train_h5, args.batch_size, train_queue, end_event, False
    )
    train_worker = Process(target=h5_read_worker, args=train_args)

    # Validation data prefetch
    val_queue = Queue(args.queued_batches)
    val_args = (args.val_h5, args.batch_size, val_queue, end_event, True)
    val_worker = Process(target=h5_read_worker, args=val_args)

    # Go!
    try:
        train_worker.start()
        val_worker.start()

        # Now we can start the training loop!
        model = build_model()
        epochs_elapsed = 0
        batches_used = 0

        try:
            while True:
                # Train and validate
                train(model, train_queue, args.batch_size)
                validate(model, val_queue)

                # Update stats
                epochs_elapsed += 1
                batches_used += batch_size
                if epochs_elapsed % args.checkpoint_epochs == 0:
                    save(model, batches_used, args.checkpoint_dir)
        finally:
            # Always save afterwards, even if we get KeyboardInterrupt'd or
            # whatever
            save(model, batches_used, args.checkpoint_dir)
    finally:
        # Make sure workers shut down gracefully
        end_event.set()
        print('Waiting for workers to exit')
        train_worker.join()
        val_worker.join()
