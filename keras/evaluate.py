"""General evaluation code for dataset. This will probably be called from an
IPython notebook so that I can visualise the result easily."""

import numpy as np

from train import read_mean_pixels, get_model_io, sub_mean_pixels


def label_to_coords(label):
    """Convert convnet output to ``k*2`` coordinate array."""
    # use label.shape instead of label.ndim for h5py (which I don't think
    # supports .ndim?)
    ndim = len(label.shape)
    if ndim == 1:
        return label.reshape((-1, 2))
    elif ndim == 2:
        return label.reshape((label.shape[0], -1, 2))
    else:
        raise ValueError("label should be output for single sample or a batch")


def get_predictions(model, mean_pixel_path, data, batch_size=32,
                    coord_sets=('joints',)):
    """Evaluate model on given images and flows in order to produce
    predictions.

    :param model: a ``keras.models.Model``
    :param mean_pixel_path: string pointing to a meat pixel to subtract
    :param data: dictionary mapping input names to ``n*c*h*w`` arrrays of image
                 data
    :param batch_size: size of batches which will be pushed through the
                       network. This is very helpful when you have a
                       ``h5py.Dataset`` to evaluate on.
    :param coord_sets: an iterable of output names. The corresponding names
                       will be treated as flattened coordinate arrays, and
                       reshaped so that they are of size `n*k*2` (where `k` is
                       the half the number of outputs associated with that
                       label).
    :return: a dictionary mapping output names to actual predictions, where
             each prediction is an `np.ndarray` with zeroth axis of size
             `n`."""
    mean_pixels = read_mean_pixels(mean_pixel_path)
    inputs, outputs = get_model_io(model)

    # Make sure that our output shapes are right (we will add to the output as
    # we go along)
    rv = {}
    for output_name in outputs:
        shape = model.output_shape[output_name]
        rv_shape = (shape[0],)
        if output_name in coord_sets:
            assert len(shape) == 2
            assert shape[1] % 2 == 0
            rv_shape += (shape[1] // 2, 2)
        else:
            rv_shape += rv_shape[1:]
        rv[output_name] = np.zeros(rv_shape)

    num_samples = len(data[inputs[0]])
    num_batches = -(-num_samples // batch_size)

    for batch_num in xrange(num_batches):
        print('Doing batch {}/{}'.format(batch_num+1, num_batches))
        slice_start = batch_num * batch_size
        slice_end = slice_start + batch_size

        batch_data = {}

        for input_name in inputs:
            bd = data[input_name][slice_start:slice_end]
            batch_data[input_name] = bd.astype('float32')

        subbed_batch_data = sub_mean_pixels(mean_pixels, batch_data)
        results_dict = model.predict(subbed_batch_data)
        for out_name, out_val in results_dict.iteritems():
            if out_name in coord_sets:
                out_val = label_to_coords(out_val)
            rv[out_name][slice_start:slice_end] = out_val

    return rv


def score_predictions_acc(gt_joints, predictions, thresholds):
    """Calculate proportion of given joints which are localised correctly at
    each given pixel distance threshold (Euclidean).

    :param gt_joints: ``n*k*2`` array of ground-truth joints.
    :param predictions: ``n*k*2`` array of predictions.
    :param thresholds: list or array of scalars representing pixel distances at
                       which to measure accuracy.
    :return: list or array of accuracies in ``[0, 1]``, with an accuracy
             corresponding to each measured threshold."""
    assert gt_joints.shape == predictions.shape
    assert gt_joints.ndim == 3
    distances = np.linalg.norm(gt_joints - predictions, axis=2)
    total_joints = float(len(distances))
    rv = []
    for threshold in thresholds:
        num_matches = np.sum(distances < threshold, axis=0)
        rv.append(num_matches / total_joints)
    return rv


def score_predictions_pcp(gt_joints, predictions, limbs):
    """Calculate strict PCP for each of the supplied limbs.

    :param gt_joints: ``n*k*2`` array of ground-truth joints.
    :param predictions: ``n*k*2`` array of predictions.
    :param limbs: list of ``(idx1, idx2)`` tuples giving indices the joints and
                  predictions arrays denoting limbs.
    :return: list or array of PCPs in ``[0, 1]``, with a PCP corresponding to
             each limb."""
    assert gt_joints.shape == predictions.shape
    assert gt_joints.ndim == 3
    rv = []
    for limb in limbs:
        gts = gt_joints[:, limb, :]
        preds = predictions[:, limb, :]
        assert gts.shape == preds.shape and gts.ndim == 3
        first_limb, second_limb = gts.transpose((1, 0, 2))
        lengths = np.linalg.norm(first_limb - second_limb, axis=1)
        assert lengths.shape == (len(gts),)
        gt_dists = np.linalg.norm(preds - gts, axis=2)
        assert gt_dists.shape == (len(gts), 2)
        # reshape just to broadcast
        valid = gt_dists < 0.5 * lengths.reshape((len(lengths), 1))
        first_valid, second_valid = valid.T
        all_valid = np.logical_and(first_valid, second_valid)
        rv.append(np.mean(all_valid))
    return rv
