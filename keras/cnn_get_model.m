function model = cnn_get_model(def_path, weights_path)
%CNN_GET_MODEL Loads a Keras model from a (JSON) definition and a set of
%trained weights.

% Monkey patch in my custom activation
utils = cnn_get_module('utils');
utils.register_activation.feval(utils.convolution_softmax, 'convolution_softmax');

% Load the model
json_string = fileread(def_path);
keras_models = cnn_get_module('keras.models');
model = keras_models.model_from_json.feval(json_string);
model.load_weights(weights_path);
end