function model = cnn_get_model(def_path, weights_path)
%CNN_GET_MODEL Loads a Keras model from a (JSON) definition and a set of
%trained weights.

% Monkey patch in my custom activation
utils = cnn_get_module('utils');
utils.register_activation.feval(utils.convolution_softmax, 'convolution_softmax');
optimizers = cnn_get_module('keras.optimizers');

% Load the model
fprintf('Loading definition from %s\n', def_path);
json_string = fileread(def_path);
keras_models = cnn_get_module('keras.models');
model = keras_models.model_from_json.feval(json_string);
% TODO: Remove the next little bit of compiling code once I get a model
% with a loss and optimiser
sgd = optimizers.SGD.feval();
losses = py.dict({{'left', 'mae'}, {'right', 'mae'}, {'head', 'mae'}, {'class', 'categorical_crossentropy'}});
model.compile(sgd, losses);
% This assertion needs to stay
assert(py.hasattr(model, '_predict'), 'Saved model needs loss and optimiser');
fprintf('Loading weights from %s\n', weights_path);
model.load_weights(weights_path);
end