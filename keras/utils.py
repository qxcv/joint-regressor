"""Various utilities which don't belong anywhere else."""

import keras.backend as K


def convolution_softmax(volume):
    """Like K.softmax, but we handle arbitrary volumes by only computing down
    axis 1."""
    # The subtraction of K.max is for numeric stability. See T.nnet docs for
    # more.
    exps = K.exp(volume - K.max(volume, axis=1, keepdims=True))
    return exps / K.sum(exps, axis=1, keepdims=True)


def register_activation(function, name):
    """Register a custom activation function in ``keras.activations``"""
    from keras import activations
    if hasattr(activations, name):
        assert getattr(activations, name) == function
    else:
        setattr(activations, name, function)

def get_model_lr(model):
    """Get the current learning rate of a Keras model (must have been compiled)"""
    opt = model.optimizer
    lr = opt.lr.get_value()
    decay = opt.decay.get_value()
    iterations = opt.iterations.get_value()
    return lr * (1.0 / (1.0 + decay * iterations))

def label_to_coords(label):
    """Convert centroid (or CNN regressor label) to coordinates using
    appropriate unflattening strategy.."""
    if label.ndim == 1:
        return label.reshape((-1, 2))
    assert label.ndim == 2
    return label.reshape((len(label), -1, 2))

def get_centroids(classes, centroids):
    """Grab the centroid of the cluster associated with each classifier output.

    :param classes: One-of-K vector (N-first, K-second) indicating appropriate
                    centroids
    :param centroids: list of P elements, each of which is a CxJp array, where
                      P is the number of poselets, C is the number of clusters
                      per poselet, and Jp is the number of regressor outputs per
                      cluster
    :returns: N-element list of the form [(class, poselet, centroid)], where
              class is in [0, P] (0 is background), centroid is None (for
              background) or a Jp/2x2 array giving (x, y) locations for each
              joint, and poselet is a poselet index in [0, C)"""
    assert classes.ndim == 2, "Expect one-of-K classes!"
    class_nums = np.argmax(classes, axis=1)
    classes_per_poselet = (classes.shape[1] - 1) / len(centroids)
    rv = []

    for num in class_nums:
        if num == 0:
            rv.append((0, None, None))
            continue
        poselet_class_idx = (num - 1) % classes_per_poselet
        poselet = (num - 1) // classes_per_poselet
        centroid = centroids[poselet][poselet_class_idx]
        rv.append((
            poselet + 1, poselet_class_idx, label_to_coords(centroid)
        ))

    return rv
