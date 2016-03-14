function [pyra, unary_map] = imCNNdet(im, model, upS)
if ~exist('upS', 'var')
  upS = 1;        % by default, we do not upscale the image
end

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
% XXX: This does not pass the right parameters into impyra
pyra = impyra(im, model, cnn_model, upS);
max_scale = numel(pyra);
FLT_MIN = realmin('single');
% 0.01;

nbh_IDs = model.nbh_IDs;
unary_map = cell(max_scale, 1);
num_subparts = numel(nbh_IDs);
model_parts = model.components{1};

for scale_idx = 1:max_scale
  joint_prob = pyra(scale_idx).feat;
  % the first dimension is the reponse of background
  joint_prob = joint_prob(:,:,2:end);
  
  % marginalize
  unary_map{scale_idx} = cell(num_subparts, 1);
  for subpose_idx = 1:num_subparts
    app_global_ids = model_parts(subpose_idx).app_global_ids;
    % TODO: What? What is it doing here?!
    unary_map{scale_idx}{subpose_idx} = sum(joint_prob(:, :, app_global_ids), 3);
    % convert to log space
    unary_map{scale_idx}{subpose_idx} = log(max(unary_map{scale_idx}{subpose_idx}, FLT_MIN));
  end
end


