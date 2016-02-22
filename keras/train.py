#!/usr/bin/env python2

"""Trains a CNN using Keras. Includes a bunch of fancy stuff so that you don't
lose your work when you Ctrl + C."""

from argparse import ArgumentParser, ArgumentTypeError
from itertools import groupby
import logging
from logging import info, warn
from os import path
from multiprocessing import Process, Queue, Event
from Queue import Full
from random import randint
from sys import stdout
from time import time

import h5py

from keras.models import Graph
from keras.optimizers import SGD
from keras.utils.generic_utils import Progbar

import numpy as np

from scipy.io import loadmat

from models import vggnet16_joint_reg_class


INIT = 'glorot_normal'


def group_sort_indices(indices):
    """Takes a list of the form `[(major_index, minor_index)]` and returns a
    list of the form `[(major_index, [minor_indices])]`, where major indices
    and minor indices are still grouped as before, but both major and minor
    indices are sorted."""
    sorted_ins = sorted(indices)
    keyfun = lambda indices: indices[0]
    rv = []
    for k, g in groupby(sorted_ins, keyfun):
        rv.append((k, [t[1] for t in g]))
    return rv


class BatchReader(object):
    def __init__(self, h5_paths, inputs, outputs, batch_size, mark_epochs,
                 shuffle, mean_pixels):
        """Initialise the worker.

        :param list h5_paths: List of paths to HDF5 files to read.
        :param list inputs: List of input names, corresponding to HDF5
            datasets.
        :param list outputs: List of output names, again corresponding to HDF5
            datasets.
        :param int batch_size: Number of datums in batches to return.
        :param bool mark_epochs: Whether to push None to the queue at the end
            of an epoch.
        :param bool shuffle: Should data be shuffled?
        :param dict mean_pixels: Dictionary giving mean pixels for each
            channel."""
        # TODO: What are the pros and cons of having a manual way of destroying
        # these open files? Originally I tried to re-open files regularly to
        # refresh the metadata in case more data had been written, but writing
        # while reading only works when SWMR is on (in which case it may not be
        # necessary to re-open the file!)
        self.h5_files = [h5py.File(path, 'r') for path in h5_paths]
        self.inputs = inputs
        self.outputs = outputs
        for filename in inputs + outputs:
            assert not filename.startswith('/')
        assert len(inputs) > 0, "Need at least one input"
        assert len(outputs) > 0, "Need at least one output"
        self.batch_size = batch_size
        self.mark_epochs = mark_epochs
        self.shuffle = shuffle
        self.mean_pixels = mean_pixels
        self._index_pool = []

    def _refresh_index_pool(self):
        self._index_pool = []

        for idx, fp in enumerate(self.h5_files):
            some_output = self.outputs[0]
            label_set = fp[some_output]
            data_size = len(label_set)
            self._index_pool.extend(
                (idx, datum_idx) for datum_idx in xrange(data_size)
            )

        if self.shuffle:
            np.random.shuffle(self._index_pool)

    def _pop_n_indices(self, n):
        rv = self._index_pool[:n]
        del self._index_pool[:n]

        return rv

    def _get_batch_indices(self):
        if self.mark_epochs and not self._index_pool:
            # The index pool has been exhausted, so we need to return [] and
            # refresh the index pool
            self._refresh_index_pool()

        indices = self._pop_n_indices(self.batch_size)

        while len(indices) < self.batch_size:
            start_len = len(indices)
            self._refresh_index_pool()
            indices.extend(self._pop_n_indices(self.batch_size - start_len))
            # Just check that the number of indices we have is actually
            # increasing
            assert len(indices) - start_len > 0, \
                "Looks like we ran out of indices :/"

        return group_sort_indices(indices)

    def _get_ds(self, ds_name, indices):
        sub_batch = None
        for fp_idx, data_indices in indices:
            fp = self.h5_files[fp_idx]
            fp_data = fp[ds_name][data_indices].astype('float32')
            if sub_batch is None:
                sub_batch = fp_data
            else:
                assert fp_data.shape[1:] == sub_batch.shape[1:]
                sub_batch = np.concatenate((sub_batch, fp_data), axis=0)
        assert sub_batch is not None
        mean_pixel = self.mean_pixels.get(ds_name)

        if mean_pixel is not None:
            # The .reshape() allows Numpy to broadcast it
            assert sub_batch.ndim == 4, "Can only mean-subtract images"
            sub_batch -= mean_pixel.reshape(
                (1, sub_batch.shape[1], 1, 1)
            )
        elif sub_batch.ndim > 2:
            warn("There's no mean pixel for dataset %s" % ds_name)

        return sub_batch

    def _get_sub_batches(self, ds_fields, indices):
        sub_batch = {}

        for ds_name in ds_fields:
            sub_batch[ds_name] = self._get_ds(ds_name, indices)

        return sub_batch

    def get_batch(self):
        # First, fetch a batch full of data
        batch_indices = self._get_batch_indices()

        assert(batch_indices or self.mark_epochs)

        if self.mark_epochs and not batch_indices:
            return None

        inputs = self._get_sub_batches(self.inputs, batch_indices)
        outputs = self._get_sub_batches(self.outputs, batch_indices)
        assert inputs.viewkeys() != outputs.viewkeys(), \
            "Can't mix inputs and outputs"
        # 'inputs' is just going to be our batch now
        inputs.update(outputs)
        return inputs


def h5_read_worker(
        out_queue, end_evt, h5_paths, inputs, outputs, mean_pixels,
        batch_size, mark_epochs, shuffle
    ):
    """This function is designed to be run in a multiprocessing.Process, and
    communicate using a ``multiprocessing.Queue``. It will just keep reading
    batches and pushing them (complete batches!) to the queue in a tight loop;
    obviously this will block as soon as the queue is full, at which point this
    process will wait until it can read again. Note that ``end_evt`` is an
    Event which should be set once the main thread wants to exit.

    At the end of each epoch, ``None`` will be pushed to the output queue iff
    ``mark_epochs`` is True. This notifies the training routine that it should
    perform validation or whatever it is training routines do nowadays."""
    reader = BatchReader(
        h5_paths=h5_paths, inputs=inputs, outputs=outputs,
        batch_size=batch_size, mark_epochs=mark_epochs, shuffle=shuffle,
        mean_pixels=mean_pixels
    )
    # Outer loop is to keep pushing forever, inner loop just polls end_event
    # periodically if we're waiting to push to the queue
    while True:
        batch = reader.get_batch()
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


def get_sample_weight(data):
    # XXX: This is a horrible hack to implement masking of the regressor
    # loss when the class (not pose = 0, is pose = 1) is not zero! This
    # mechanism should be controlled by a command-line argument, with a warning
    # when the expected mask (to joints/class) is not applied.
    sample_weight = {}
    assert 'joints' in data, 'Yes, you need to implement better masking :)'
    assert 'class' in data
    if 'class' in data and 'joints' in data:
        classes = data['class']
        assert classes.ndim == 2 and classes.shape[1] == 1
        sample_weight['joints'] = classes.flatten().astype('bool')
    return sample_weight


def train(model, queue, iterations):
    """Perform a fixed number of training iterations."""
    info('Training for %i iterations', iterations)
    p = Progbar(iterations)
    loss = 0.0
    p.update(0)

    for iteration in xrange(iterations):
        # First, get some data from the queue
        start_time = time()
        data = queue.get()
        fetch_time = time() - start_time

        sample_weight = get_sample_weight(data)

        # Next, do some backprop
        start_time = time()
        loss, = model.train_on_batch(data, sample_weight=sample_weight)
        bp_time = time() - start_time

        # Finally, write some debugging output
        extra_info = [
            ('loss', loss),
            ('fetchtime', fetch_time),
            ('bproptime', bp_time)
        ]
        p.update(iteration + 1, extra_info)


def validate(model, queue, batches):
    """Perform one epoch of validation."""
    info('Testing on validation set')
    samples = 0
    weighted_loss = 0.0

    for batch_num in xrange(batches):
        data = queue.get()
        assert data is not None

        sample_weight = get_sample_weight(data)

        loss, = model.test_on_batch(data, sample_weight=sample_weight)

        # Update stats
        sample_size = len(data[data.keys()[0]])
        samples += sample_size
        weighted_loss += sample_size * loss

        if (batch_num % 10) == 0:
            info('%i validation batches tested', batch_num)

    info(
        'Finished %i batches (%i samples); mean loss-per-sample %f',
        batches, samples, weighted_loss / max(1, samples)
    )


def save(model, iteration_no, dest_dir):
    """Save the model to a checkpoint file."""
    filename = 'model-iter-{}-r{:06}.h5'.format(iteration_no, randint(1, 1e6-1))
    full_path = path.join(dest_dir, filename)
    # Note that save_weights will prompt if the file already exists. This is
    # why I've added a small random number to the end; hopefully this will
    # allow unattended optimisation runs to complete even when Keras decides
    # that it doesn't want to play nicely.
    info("Saving model to %s", full_path)
    model.save_weights(full_path)


def read_mean_pixels(mat_path):
    if mat_path is None:
        # No mean pixel
        return {}
    mat = loadmat(mat_path)
    mean_pixels = {
        k: v.flatten() for k, v in mat.iteritems() if not k.startswith('_')
    }
    return mean_pixels


def sub_mean_pixels(mean_pixels, all_data):
    rv = {}

    for name, data in all_data.iteritems():
        mean_pixel = mean_pixels.get(name)

        if mean_pixel is not None:
            assert data.ndim == 4, "Can only mean-subtract images"
            rv[name] = data - mean_pixel.reshape(
                (1, data.shape[1], 1, 1)
            )
        elif data.ndim > 2:
            # Only warn about image-like things
            warn("There's no mean pixel for dataset %s" % name)

    return rv


def infer_sizes(h5_path):
    """Just return shapes of all datasets, assuming that different samples are
    indexed along the first dimension."""
    rv = {}

    with h5py.File(h5_path, 'r') as fp:
        for key in fp.keys():
            rv[key] = fp[key].shape[1:]

    return rv


def _model_io_map(config):
    return {
        cfg['name']: cfg for cfg in config
    }


def get_model_io(model):
    assert isinstance(model, Graph)
    inputs = _model_io_map(model.input_config)
    outputs = _model_io_map(model.output_config)
    return (inputs, outputs)


parser = ArgumentParser(description="Train a CNN to regress joints")


def h5_parser(h5_string):
    # parse strings,like,this without worrying about ,,stuff,like,,this,
    rv = [p for p in h5_string.split(',') if p]
    if not rv:
        raise ArgumentTypeError('Expected at least one path')
    return rv


# Mandatory arguments
parser.add_argument(
    'train_h5s', metavar='TRAINDATA', type=h5_parser,
    help='h5 files in which training samples are stored (comma separated)'
)
parser.add_argument(
    'val_h5s', metavar='VALDATA', type=h5_parser,
    help='h5 file in which validation samples are stored (comma separated)'
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
parser.add_argument(
    '--finetune', dest='finetune_path', type=str, default=None,
    help='finetune from these weights instead of starting again'
)
parser.add_argument(
    '--logfile', dest='log_file', type=str, default=None,
    help='path to log messages to (in addition to std{err,out})'
)
# TODO: Add configuration option to just run through the entire validation set
# like I was doing before. That's a lot faster than using randomly sampled
# stuff.
parser.add_argument(
    '--val-batches', dest='val_batches', type=int, default=5,
    help='number of batches to run during each validation step'
)


if __name__ == '__main__':
    # Start by parsing arguments and setting up logger
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    if args.log_file is not None:
        file_handler = logging.FileHandler(args.log_file, mode='a')
        logging.getLogger().addHandler(file_handler)
    info('Logging started')

    # Model-building
    ds_shape = infer_sizes(args.train_h5s[0])
    # TODO: I need to figure out a more elegant way of doing this than sticking
    # it in train.py. Eventually I want to do it like Caffe does, where input
    # and loss layer names each correspond to HDF5 dataset names.
    input_shape = ds_shape['images']
    regressor_outputs = ds_shape['joints'][0]
    solver = SGD(
        lr=args.learning_rate, decay=args.decay, momentum=0.9, nesterov=True
    )
    model = vggnet16_joint_reg_class(
        input_shape, regressor_outputs, solver, INIT
    )
    if args.finetune_path is not None:
        info("Loading weights from '%s'", args.finetune_path)
        model.load_weights(args.finetune_path)

    inputs, outputs = [d.keys() for d in get_model_io(model)]

    # Prefetching stuff
    end_event = Event()
    mean_pixels = read_mean_pixels(args.mean_pixel_path)
    # Training data prefetch
    train_queue = Queue(args.queued_batches)
    # We supply everything as kwargs so that I know what I'm passing in
    train_kwargs = dict(
        h5_paths=args.train_h5s, batch_size=args.batch_size,
        out_queue=train_queue, end_evt=end_event, mark_epochs=False,
        shuffle=True, mean_pixels=mean_pixels, inputs=inputs, outputs=outputs
    )
    train_worker = Process(target=h5_read_worker, kwargs=train_kwargs)

    # Validation data prefetch
    val_queue = Queue(args.queued_batches)
    val_kwargs = dict(
        h5_paths=args.val_h5s, batch_size=args.batch_size, out_queue=val_queue,
        end_evt=end_event, mark_epochs=False, shuffle=True,
        mean_pixels=mean_pixels, inputs=inputs, outputs=outputs
    )
    val_worker = Process(target=h5_read_worker, kwargs=val_kwargs)

    try:
        # Protect this in a try: for graceful cleanup of workers
        train_worker.start()
        val_worker.start()

        # Stats
        epochs_elapsed = 0
        batches_used = 0

        try:
            while True:
                # Train and validate
                train(model, train_queue, args.train_interval_batches)
                validate(model, val_queue, args.val_batches)

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
        info('Waiting for workers to exit')
        train_worker.join(10.0)
        val_worker.join(10.0)
        # XXX: My termination scheme (with end_event) is not working, and I
        # can't tell where the workers are getting stuck.
        info(
            'Train worker alive? %s; Val worker alive? %s; terminating anyway',
            train_worker.is_alive(), val_worker.is_alive()
        )
        train_worker.terminate()
        val_worker.terminate()
