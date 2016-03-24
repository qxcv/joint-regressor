function [pyra, unary_map] = imCNNdet(im_stack, flow, model, upS)
if ~exist('upS', 'var')
    upS = 1; % by default, we do not upscale the image
end

assert(size(im_stack, 3) == 6, 'Need RGBRGB channels');
assert(all(size(im_stack) == size(flow) | [0 0 1]), ...
    'Stack and flow should have same width/height');

% This seems to be necessary to make sure that Keras memory is cleaned up
force_pygc();
cnnpar = model.cnn;
persistent cnn_model;
if isempty(cnn_model)
    if ~exist(cnnpar.deploy_json, 'file') || ~exist(cnnpar.deploy_weights, 'file')
        error('jointregressor:invalidCNNConfig', ...
            'Need "%s" and "%s" to initialise CNN', cnnpar.deploy_json, ...
            cnnpar.deploy_weights);
    end
    cnn_model = cnn_get_model(cnnpar.deploy_json, cnnpar.deploy_weights);
end
assert(~isempty(cnn_model));

if upS > 1
    % ensure largest length < 1200
    [imx, imy, ~] = size(im);
    upS = min(upS, 600 / max(imx,imy));
end
cnn_size = cnnpar.window(1);
assert(all(cnn_size == cnnpar.window));
pyra = impyra(im_stack, flow, cnn_model, cnnpar.mean_pixels, ...
    cnnpar.step, cnn_size, model.interval, upS);
max_scale = numel(pyra);
FLT_MIN = realmin('single');
% 0.01;

unary_map = cell(max_scale, 1);
num_subposes = length(model.subpose_pa);
model_parts = model.components;

for scale_idx = 1:max_scale
    % the first dimension is the reponse of background, but app_global_ids
    % accounts for that
    feat = pyra(scale_idx).feat;
    
    % marginalize
    unary_map{scale_idx} = cell(num_subposes, 1);
    for subpose_idx = 1:num_subposes
        app_global_ids = model_parts(subpose_idx).app_global_ids;
        assert(all(app_global_ids > 1));
        % at each location l_i (for subpose index i, type index t_i),
        % gives p(s=i,t=t_i | I(l_i); theta).
        subpose_map = feat(:, :, app_global_ids);
        % convert to log space
        unary_map{scale_idx}{subpose_idx} = log(max(subpose_map, FLT_MIN));
    end
end
end