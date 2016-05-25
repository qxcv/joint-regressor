function model = cnn_get_model(def_path, weights_path, cnn_conf)
%CNN_GET_MODEL Loads a Keras model from a (JSON) definition and a set of
%trained weights.

if nargin < 3
    cnn_conf.gpu = 0;
    cnn_conf.cnmem = 0.2;
end

% Monkey patch in my custom activation
configure_env(cnn_conf);
utils = cnn_get_module('utils');
utils.register_activation.feval(utils.convolution_softmax, 'convolution_softmax');

% Load the model
fprintf('Loading definition from %s\n', def_path);
json_string = fileread(def_path);
keras_models = cnn_get_module('keras.models');
model = keras_models.model_from_json.feval(json_string);
% This assertion needs to stay
assert(py.hasattr(model, '_predict'), 'Saved model needs loss and optimiser');
fprintf('Loading weights from %s\n', weights_path);
model.load_weights(weights_path);
end
