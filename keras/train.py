#!/usr/bin/env python2

"""Trains a CNN using Keras. Includes a bunch of fancy stuff so that you don't
lose your work when you Ctrl + C."""

from argparse import ArgumentParser, ArgumentTypeError
import datetime
from itertools import groupby
from json import dumps
import logging
from logging import info, warn
from os import path, makedirs, environ
from multiprocessing import Process, Queue, Event, Lock
from Queue import Full
from random import randint
from sys import stdout, argv
from time import time

import h5py

from keras.models import Graph
from keras.optimizers import SGD
from keras.utils.generic_utils import Progbar

import numpy as np

from scipy.io import loadmat

import models


INIT = 'glorot_normal'


def mkdir_p(dir_path):
    try:
        makedirs(dir_path)
    except OSError as e:
        # 17 means "already exists"
        if e.errno != 17:
            raise e


class NumericLogger(object):
    def __init__(self, dest):
        self.dest = dest
        self.lock = Lock()

    def append(self, data):
        if 'time' not in data:
            data['time'] = datetime_str()
        json_data = dumps(data)
        # No file-level locking because YOLO
        with self.lock:
            with open(self.dest, 'a') as fp:
                # Should probably use a binary format or something. Oh well.
                fp.write(json_data + '\n')

# Needs to be set up in main function
numeric_log = None


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


def get_sample_weight(data, classname, masks):
    # [:] is for benefit of h5py
    class_data = data[classname][:].astype('int')
    assert class_data.ndim == 2
    # Quick checks to ensure it's valid one-hot data
    assert class_data.shape[1] > 1
    assert (np.sum(class_data, axis=1) == 1).all()
    assert np.logical_or(class_data == 1, class_data == 0).all()
    # The data is one-hot, so we need to change it to be just integer class
    # labels
    classes = np.argmax(class_data, axis=1)
    assert classes.ndim == 1
    # num_classes = class_data.shape[1]
    # Make sure that the number of masks is the number of classes or the number
    # of classes + 1
    mask_names = set(masks.keys())

    # XXX: I've commented out these assertions because they only made sense
    # when we needed a 1:1 mapping between classes and outputs.
    # assert len(mask_names) == len(masks)
    # assert num_classes - 1 <= len(mask_names) <= num_classes
    # mask_vals = {val for name, val in masks}
    # assert(len(mask_vals) == len(mask_names))
    # Masks are in [0, num_classes); assert that this is the case
    # Note that if the number of classes is one fewer than the number of input
    # masks, then we assume that the zeroth class does not control any external
    # loss.
    # assert max(mask_vals) == num_classes - 1
    # assert len(mask_names) < num_classes or min(mask_vals) == 0

    sample_weight = {}

    for mask_name, mask_vals in masks.iteritems():
        sample_weight[mask_name] = np.in1d(classes, mask_vals).astype('float32')

    assert len(sample_weight) == len(mask_names)

    return sample_weight


def train(model, queue, iterations, mask_class_name, masks):
    """Perform a fixed number of training iterations."""
    assert (mask_class_name is None) == (masks is None)

    info('Training for %i iterations', iterations)
    p = Progbar(iterations)
    loss = 0.0
    p.update(0)
    mean_loss = 0

    for iteration in xrange(iterations):
        # First, get some data from the queue
        start_time = time()
        data = queue.get()
        fetch_time = time() - start_time

        if mask_class_name is not None:
            sample_weight = get_sample_weight(data, mask_class_name, masks)
        else:
            sample_weight = {}

        # Next, do some backprop
        start_time = time()
        loss, = model.train_on_batch(data, sample_weight=sample_weight)
        loss = float(loss)
        bp_time = time() - start_time
        learning_rate = model.optimizer.get_config()['lr']

        # Finally, write some debugging output
        extra_info = [
            ('loss', loss),
            ('lr', learning_rate),
            ('fetcht', fetch_time),
            ('bpropt', bp_time)
        ]
        p.update(iteration + 1, extra_info)

        # Update mean loss
        mean_loss += float(loss) / iterations

        # Log results to numeric log
        numlog_data = dict(extra_info)
        numlog_data['type'] = 'train'
        numlog_data['bsize'] = len(data[data.keys()[0]])
        numeric_log.append(numlog_data)

    learning_rate = model.optimizer.get_config()['lr']
    info('Finished {} training batches, mean loss-per-batch {}, LR {}'.format(
        iterations, mean_loss, learning_rate
    ))


def validate(model, queue, batches, mask_class_name, masks):
    """Perform one epoch of validation."""
    info('Testing on validation set')
    samples = 0
    weighted_loss = 0.0

    for batch_num in xrange(batches):
        data = queue.get()
        assert data is not None

        if mask_class_name is not None:
            sample_weight = get_sample_weight(data, mask_class_name, masks)
        else:
            sample_weight = {}

        loss, = model.test_on_batch(data, sample_weight=sample_weight)
        loss = float(loss)

        # Update stats
        sample_size = len(data[data.keys()[0]])
        samples += sample_size
        weighted_loss += sample_size * loss

        numeric_log.append({
            'type': 'val_batch',
            'loss': loss,
            'w_loss': weighted_loss,
            'bsize': sample_size
        })

        if (batch_num % 10) == 0:
            info('%i validation batches tested', batch_num)

    mean_loss = weighted_loss / max(1, samples)
    info(
        'Finished %i batches (%i samples); mean loss-per-sample %f',
        batches, samples, mean_loss
    )
    numeric_log.append({
        'type': 'val_done',
        'mean_loss': mean_loss
    })


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


def h5_parser(h5_string):
    # parse strings,like,this without worrying about ,,stuff,like,,this,
    rv = [p for p in h5_string.split(',') if p]
    if not rv:
        raise ArgumentTypeError('Expected at least one path')
    return rv

def loss_mask_parser(arg):
    """Takes as input a string dictating which losses should be enabled by
    which class values, and returns a dictionary with the same information in a
    more malleable form.

    :param arg: String of form
                ``<class>:<output>=<value>,<output>=<value>,...``
    :returns: Dictionary with keys corresponding to seen outputs, each with an
              array value indicating which class values should unmask it; if
              the class holds any of the values in the set associated with an
              output, then that output should count towards the total loss of
              the model."""
    classname, rest = arg.split(':', 1)
    nosep = rest.split(',')
    pairs = [s.split('=', 1) for s in nosep if '=' in s]
    set_dict = {}
    for label, clas in pairs:
        set_dict.setdefault(label, set()).add(int(clas))
    # We return a dictionary mapping output names to *Numpy arrays* of
    # corresponding classes. This lets us use np.in1d to test for membership in
    # a vectorised way.
    rv_dict = {
        k: np.fromiter(v, dtype='int') for k, v in set_dict.iteritems()
    }
    return classname, rv_dict


def datetime_str():
    return datetime.datetime.now().isoformat()


def setup_logging(work_dir):
    global numeric_log
    logging.basicConfig(level=logging.DEBUG)
    log_dir = path.join(work_dir, 'logs')
    mkdir_p(log_dir)
    t = datetime_str()
    num_log_fn = path.join(log_dir, 'numlog-' + t + '.log')
    log_file = path.join(log_dir, 'log-' + t + '.log')
    file_handler = logging.FileHandler(log_file, mode='a')
    logging.getLogger().addHandler(file_handler)
    numeric_log = NumericLogger(num_log_fn)
    info('=' * 80)
    info('Logging started at ' + datetime_str())
    info('Human-readable log path: {}'.format(log_file))
    info('Numeric log path: {}'.format(num_log_fn))


def load_model(args):
    ds_shape = infer_sizes(args.train_h5s[0])
    solver = SGD(
        lr=args.learning_rate, decay=args.decay, momentum=0.9, nesterov=True
    )
    model_to_load = getattr(models, args.model_name)
    info('Using loader %s' % args.model_name)
    model = model_to_load(
        ds_shape, solver, INIT
    )
    info('Loaded model of type {}'.format(type(model)))
    opt_cfg = model.optimizer.get_config()
    info('Solver data: decay={decay}, lr={lr}, momentum={momentum}, '
         'nesterov={nesterov}'.format(**opt_cfg))
    if args.finetune_path is not None:
        info("Loading weights from '%s'", args.finetune_path)
        model.load_weights(args.finetune_path)
    return model


def get_parser():
    """Grab the ``argparse.ArgumentParser`` for this application. For some
    reason ``argparse`` needs access to ``sys.argv`` to build an
    ``ArgumentParser`` (not just to actually evaluate it), so I've put this
    into its own function so that it doesn't get executed when running from
    environments with no ``sys.argv`` (e.g. Matlab)"""
    parser = ArgumentParser(description="Train a CNN to regress joints")

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
        'working_dir', metavar='WORKINGDIR', type=str,
        help='directory in which to store checkpoint files and logs'
    )

    # Optargs
    parser.add_argument(
        '--queued-batches', dest='queued_batches', type=int, default=32,
        help='number of unused batches stored in processing queue (in memory)'
    )
    parser.add_argument(
        '--batch-size', dest='batch_size', type=int, default=16,
        help='batch size for both training (backprop) and validation'
    )
    parser.add_argument(
        '--checkpoint-epochs', dest='checkpoint_epochs', type=int, default=5,
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
    # TODO: Add configuration option to just run through the entire validation set
    # like I was doing before. That's a lot faster than using randomly sampled
    # stuff. Edit: I think I pushed down the validaiton block size, so now random
    # selection should be relatively fast.
    parser.add_argument(
        '--val-batches', dest='val_batches', type=int, default=50,
        help='number of batches to run during each validation step'
    )
    parser.add_argument(
        # Syntax proposal: 'class:out1=1,out2=2,out3=3', where 'class' is the name
        # of the class output (we only look at the ground truth) and out1,out2,out3
        # are names of outputs to be masked. In this case, out1's loss is only
        # enabled when the GT class is 1 (or [0 1 0 0 ...] in one-hot notation),
        # out2's loss is only enabled whne the GT class is 2 ([0 0 1 0 ...]), etc.
        '--cond-losses', dest='loss_mask', type=loss_mask_parser, default=None,
        help="use given GT class to selectively enable losses"
    )
    parser.add_argument(
        '--model-name', dest='model_name', type=str,
        default='vggnet16_joint_reg_class_flow',
        help='name of model to use'
    )

    return parser


if __name__ == '__main__':
    # Start by parsing arguments and setting up logger
    parser = get_parser()
    args = parser.parse_args()

    # Set up checkpointing and logging
    work_dir = args.working_dir
    checkpoint_dir = path.join(work_dir, 'checkpoints')
    mkdir_p(checkpoint_dir)
    setup_logging(work_dir)
    info('argv: {}'.format(argv))
    info('THEANO_FLAGS: {}'.format(environ.get('THEANO_FLAGS')))

    # Model-building
    model = load_model(args)
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

    if args.loss_mask is not None:
        mask_class_name, masks = args.loss_mask
    else:
        warn('No masks supplied for conditional regression!')
        mask_class_name, masks = None, None

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
                validate(
                    model, val_queue, args.val_batches, mask_class_name, masks
                )
                train(
                    model, train_queue, args.train_interval_batches,
                    mask_class_name, masks
                )

                # Update stats
                epochs_elapsed += 1
                batches_used += args.train_interval_batches
                is_checkpoint_epoch = epochs_elapsed % args.checkpoint_epochs == 0
                if epochs_elapsed > 0 and is_checkpoint_epoch:
                    save(model, batches_used, checkpoint_dir)
        finally:
            # Always save afterwards, even if we get KeyboardInterrupt'd or
            # whatever
            stdout.write('\n')
            save(model, batches_used, checkpoint_dir)
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
