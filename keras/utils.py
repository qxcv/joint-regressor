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
