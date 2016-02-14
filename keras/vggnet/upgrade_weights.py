# Copy weights from the downloaded VGG_16 model to the new model, with the
# following caveats:
#
# 1) If the number of channels in the first dimension is different, then the
# weights from the first layer will be replicated across channels as required.
#
# 2) If the number of channels in the last layer (the output layer) is
# different, then the output layer will not have weights copied.

import numpy as np

from vgg16_keras import VGG_16


def upgrade_weights(new_model, old_weights_path):
    old_model = VGG_16(old_weights_path)

    # First, make sure that layers are all of the same type
    assert len(new_model.layers) == len(old_model.layers)

    # Now some safety checks on the first two layers
    for l0, l1 in [new_model.layers[:2], old_model.layers[:2]]:
        assert l0.get_config()['name'] == 'ZeroPadding2D'
        assert l1.get_config()['name'] == 'Convolution2D'

    # Next, we can copy across the first layer's convolution2D weights
    new_l1 = new_model.layers[1]
    new_l1_weights = new_l1.get_weights()
    old_l1_weights = old_model.layers[1].get_weights()
    print('Concatenating old L1 dimensions to change shape, {}->{}'.format(
            old_l1_weights[0].shape, new_l1_weights[0].shape
    ))
    # Ceiling division
    num_repeats = -(-new_l1_weights[0].shape[1] // old_l1_weights[0].shape[1])
    repped = np.repeat(old_l1_weights[0], num_repeats, axis=1)
    new_l1.set_weights([
        repped[:, :new_l1_weights[0].shape[1], :, :],
        old_l1_weights[1]
    ])
    print('First layer done')

    # Check compatibility of layers 2-(-2) (so skip last layer and first two
    # because we know they are different), then copy across
    zipped = zip(new_model.layers[2:-1], old_model.layers[2:-1])
    for new_layer, old_layer in zipped:
        new_cfg = new_layer.get_config()
        old_cfg = old_layer.get_config()
        # XXX: Might need more aggressive checks than this
        assert new_layer.input_shape == old_layer.input_shape
        assert new_cfg['name'] == old_cfg['name']
        print('Changing {}'.format(new_cfg['name']))

        # If we got this far, we should be good!
        new_layer.set_weights(old_layer.get_weights())

    # Just for the sake of chaining
    return new_model
