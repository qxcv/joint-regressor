function cnn_train(cnnpar, ~)
%CNN_TRAIN Train a CNN using Keras

% TODO: Make training automatic. I can do this manually, but people who
% want to reproduce my results can't. Roughly, you need to:
% 1) Activate the virtualenv
% 2) Run train.py with the appropriate arguments for an apprporiate number
%    of iterations.
% 3) Convert the net to a fully convolutional one and save the weights and
%    model definition (see debugging-convnet.ipynb).

% TODO: Should fix this before I release
if ~(exist(cnnpar.deploy_json, 'file') && exist(cnnpar.deploy_weights, 'file'))
    error('jointregressor:nocnn', ...
        ['You need to run train.py to train a network, then use the ' ...
         'provided notebook to convert it to an FC net. This should ' ...
         'give you a model definition (%s) and a weights file (%s).'], ...
         cnnpar.deploy_json, cnnpar.deploy_weights);
end
end

