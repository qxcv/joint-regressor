"""General evaluation code for dataset. This will probably be called from an
IPython notebook so that I can visualise the result easily."""

import numpy as np

from train import read_mean_pixel


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


def get_predictions(model, mean_pixel_path, images=None, flows=None,
                    batch_size=32):
    """Evaluate model on given images and flows in order to produce
    predictions.

    :param model: a ``keras.models.Model``
    :param mean_pixel_path: string pointing to a meat pixel to subtract
    :param images: an ``n*c*h*w`` arrray of image data
    :param flows: an ``n*2*h*w`` set of flows
    :param batch_size: size of batches which will be pushed through the
                       network. This is very helpful when you have a
                       ``h5py.Dataset`` to evaluate on.
    :return: a ``n*k*2`` array of ``(x, y)`` joint coordinates."""
    use_flow = flows is not None
    use_rgb = images is not None
    assert use_flow or use_rgb

    if use_flow:
        num_samples = flows.shape[0]
    else:
        num_samples = images.shape[0]

    mp = read_mean_pixel(mean_pixel_path, use_flow=use_flow, use_rgb=use_rgb)
    num_outputs = model.layers[-1].output_dim
    rv = np.zeros((num_samples, num_outputs / 2, 2))
    num_batches = -(-num_samples // batch_size)

    for batch_num in xrange(num_batches):
        print('Doing batch {}/{}'.format(batch_num+1, num_batches))
        slice_start = batch_num * batch_size
        slice_end = slice_start + batch_size

        if use_flow:
            flow_data = flows[slice_start:slice_end].astype('float32')

        if use_rgb:
            image_data = images[slice_start:slice_end].astype('float32')

        if use_flow and use_rgb:
            stacked = np.concatenate((image_data, flow_data), axis=1)
        elif use_flow:
            stacked = flow_data
        elif use_rgb:
            stacked = image_data

        stacked -= mp.reshape((len(mp), 1, 1))
        results = model.predict(stacked)
        assert results.ndim == 2
        rv[slice_start:slice_end] = label_to_coords(results)

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
        num_matches = np.sum(distances < threshold)
        rv.append(num_matches / total_joints)
    return rv


def score_predictions_pcp(gt_joints, predictions, limbs):
    """Calculate PCP for each of the supplied limbs.

    :param gt_joints: ``k*2`` array of ground-truth joints.
    :param predictions: ``k*2`` array of predictions.
    :param limbs: list of ``(idx1, idx2)`` tuples giving indices the joints and
                  predictions arrays denoting limbs.
    :return: list or array of PCPs in ``[0, 1]``, with a PCP corresponding to
             each limb."""
    assert gt_joints.shape == predictions.shape
    assert gt_joints.ndim == 3
