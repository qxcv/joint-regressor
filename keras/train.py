#!/usr/bin/env python2

"""Trains a CNN using Keras. Includes a bunch of fancy stuff so that you don't
lose your work when you Ctrl + C."""

from argparse import ArgumentParser
from itertools import chain, islice, repeat
from os import path
from multiprocessing import Process, Queue, Event
from Queue import Full
from random import randint
from sys import stdout
from time import time

import h5py

from keras.optimizers import SGD
from keras.utils.generic_utils import Progbar

import numpy as np

from scipy.io import loadmat

from models import vggnet16_regressor_model

INIT = 'glorot_normal'

# TODO:
#
# 1) Add flag to get graph model working
#
# 2) Opportunistic refactoring :-)
#
# 3) Figuring out why end_evt doesn't work. It looks like my workers are
# exiting after a few seconds (through a return), yet are still marked alive
# (?!). I guess there's some sort of cleanup going on there which I'm not privy
# to.

def h5_read_worker(
        h5_path, batch_size, out_queue, end_evt, mark_epochs, shuffle,
        mean_pixel
    ):
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

    detail_str = 'Worker started. Details:\n' \
        'h5_path: {h5_path}\n' \
        'batch_size: {batch_size}\n' \
        'mark_epochs: {mark_epochs}\n' \
        'shuffle: {shuffle}\n' \
        'mean_pixel: {mean_pixel}'.format(**locals())
    print(detail_str)

    with h5py.File(h5_path, 'r') as fp:
        def index_gen():
            """Yields random indices into the dataset. Fetching data this way
            is slow, but it should be okay given that we're running this in a
            background process."""
            label_set = fp['/joints']
            label_size = len(label_set)
            if shuffle:
                elems = np.random.permutation(label_size)
            else:
                elems = np.arange(label_size)
            for idx in elems:
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
                # h5py wants its indices sorted (presumably so that each
                # accessed chunk only has to be loaded once), so we give it
                # sorted indices and then use a second set of indices to invert
                # the mapping (so that the actual batch data order is the same
                # as the one described by batch_indices). Note that I'm
                # converting to list() because otherwise h5py gives a cryptic
                # error message saying it only supports "boolean array"
                # indexing (which means that if the input to the indexing
                # function is a numpy array, then it must be boolean).
                sorted_index_indices = np.argsort(batch_indices)
                inverse_indices = list(np.argsort(sorted_index_indices))
                # Oh god the difference between list's __getitem__ and
                # np.array's __getitem__ is driving me loopy
                sorted_indices = list(np.array(batch_indices)[sorted_index_indices])
                batch_im = fp['/images'][sorted_indices].astype('float32')
                batch_flow = fp['/flow'][sorted_indices].astype('float32')
                batch_data = np.concatenate((batch_im, batch_flow), axis=1)
                batch_labels = fp['/joints'][sorted_indices].astype('float32')
                if mean_pixel is not None:
                    # The .reshape() allows Numpy to broadcast it
                    batch_data -= mean_pixel.reshape(
                        (1, len(mean_pixel), 1, 1)
                    )
                batch = (batch_data[inverse_indices], batch_labels[inverse_indices])

            # This is the push loop
            while True:
                if end_evt.is_set():
                    out_queue.close()
                    return

                try:
                    out_queue.put(batch, timeout=0.05)
                    break
                except Full:
                    # Queue.Full (for Queue = the module in stdlib, not the
                    # class) is raised when we time out
                    pass


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
        loss, = model.train_on_batch(X, y)
        bp_time = time() - start_time

        # Finally, write some debugging output
        extra_info = [
            ('loss', loss),
            ('fetchtime', fetch_time),
            ('bproptime', bp_time)
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
        loss, = model.test_on_batch(X, y)

        # Update stats
        batches += 1
        samples += len(X)
        weighted_loss += samples * loss

        if (batches % 100) == 0:
            print('{} validation batches tested'.format(batches))

    print(
        'Finished {} batches ({} samples); mean loss-per-sample {}'
        .format(batches, samples, weighted_loss / max(1, samples))
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


def read_mean_pixel(mat_path):
    if mat_path is None:
        # No mean pixel
        return None
    mat = loadmat(mat_path)
    im_mean = mat['image_mean_pixel'].flatten()
    flow_mean = mat['flow_mean_pixel'].flatten()
    return np.concatenate((im_mean, flow_mean))


def infer_sizes(h5_path):
    """Infer relevant data sizes from a HDF5 file."""
    with h5py.File(h5_path, 'r') as fp:
        im_shape = fp['/images'].shape[1:]
        flow_shape = fp['/flow'].shape[1:]
        assert(im_shape[1:] == flow_shape[1:])
        input_shape = (im_shape[0] + flow_shape[0], im_shape[1], im_shape[2])
        regressor_outputs = fp['/joints'].shape[1]
        if 'poselet' in fp.keys():
            biposelet_classes = max(fp['/poselet'])
        else:
            biposelet_classes = None

    return (input_shape, regressor_outputs, biposelet_classes)


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
    '--queued-batches', dest='queued_batches', type=int, default=16,
    help='number of unused batches stored in processing queue (in memory)'
)
parser.add_argument(
    '--batch-size', dest='batch_size', type=int, default=48,
    help='batch size for both training (backprop) and validation'
)
parser.add_argument(
    '--checkpoint-epochs', dest='checkpoint_epochs', type=int, default=2,
    help='training intervals to wait before writing a checkpoint file'
)
parser.add_argument(
    '--train-interval-batches', dest='train_interval_batches', type=int,
    default=256, help='number of batches to train for between validation'
)
parser.add_argument(
    '--mean-pixel-mat', dest='mean_pixel_path', type=str, default=None,
    help='.mat containing mean pixel'
)
parser.add_argument(
    '--learning-rate', dest='learning_rate', type=float, default=0.0001,
    help='learning rate for SGD'
)
parser.add_argument(
    '--decay', dest='decay', type=float, default=1e-6,
    help='decay for SGD'
)


if __name__ == '__main__':
    args = parser.parse_args()
    end_event = Event()
    mean_pixel = read_mean_pixel(args.mean_pixel_path)

    # Training data prefetch
    train_queue = Queue(args.queued_batches)
    train_args = (
        args.train_h5, args.batch_size, train_queue, end_event, False, True,
        mean_pixel
    )
    train_worker = Process(target=h5_read_worker, args=train_args)

    # Validation data prefetch
    val_queue = Queue(args.queued_batches)
    val_args = (
        args.val_h5, args.batch_size, val_queue, end_event, True, False,
        mean_pixel
    )
    val_worker = Process(target=h5_read_worker, args=val_args)

    try:
        # Protect this in a try: for graceful cleanup of workers
        train_worker.start()
        val_worker.start()

        # Model-building
        input_shape, regressor_outputs, biposelet_classes = infer_sizes(args.train_h5)
        solver = SGD(
            lr=args.learning_rate, decay=args.decay, momentum=0.9, nesterov=True
        )
        model = vggnet16_regressor_model(
            input_shape, regressor_outputs, solver, INIT
        )

        # Stats
        epochs_elapsed = 0
        batches_used = 0

        try:
            while True:
                # Train and validate
                train(model, train_queue, args.train_interval_batches)
                validate(model, val_queue)

                # Update stats
                epochs_elapsed += 1
                batches_used += args.batch_size
                is_checkpoint_epoch = epochs_elapsed % args.checkpoint_epochs == 0
                if epochs_elapsed > 0 and is_checkpoint_epoch:
                    save(model, batches_used, args.checkpoint_dir)
        finally:
            # Always save afterwards, even if we get KeyboardInterrupt'd or
            # whatever
            stdout.write('\n')
            save(model, batches_used, args.checkpoint_dir)
    finally:
        # Make sure workers shut down gracefully
        end_event.set()
        stdout.write('\n')
        print('Waiting for workers to exit')
        train_worker.join(10.0)
        val_worker.join(10.0)
        # XXX: My termination scheme (with end_event) is not working, and I
        # can't tell where the workers are getting stuck.
        print(
            'Train worker alive? {}; Val worker alive? {}; terminating anyway'
            .format(train_worker.is_alive(), val_worker.is_alive())
        )
        train_worker.terminate()
        val_worker.terminate()
