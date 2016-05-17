function results = stitching_weight_search(pair_dets, valid_parts, pose_gts, limbs)
%STITCHING_WEIGHT_SEARCH Naively find good stitching weights
% Uses randomised search internally (apparently that is better than grid
% search?)

flat_gts = cat(2, pose_gts{:});
calc_thresh = @(gt_pose) max(nanmax(gt_pose, [], 1) - nanmin(gt_pose, [], 1));
threshs = cellfun(calc_thresh, flat_gts);
% Max PCK threshold will be around the bbox of a pose
max_thresh = quantile(threshs, 0.75);
assert(max_thresh > 0 && isfinite(max_thresh));
pck_thresholds = linspace(0, max_thresh, 50);
count = 100;

% Generate a bunch of weight vector configurations randomly. Hopefully 100
% weights will be fine. If this produces radically different results each
% time, then I might want to try increasing the weight count or maybe even
% using something from Matlab's global optimisation toolbox.
param_configs = random_stitching_weights(count);

% Now we can choose the best
results = struct('config', cell([1, count]), ...
    'pcp_fit', nan([1, count]), ...
    'pck_fit', nan([1, count]));
parfor param_idx=1:length(param_configs)
    fprintf('Testing configuration %i/%i\n', param_idx, length(param_configs));
    
    % Grab config
    current_params = param_configs(param_idx);
    results(param_idx).config = current_params;
    
    % Get  results
    pose_dets = stitch_all_seqs(pair_dets, current_params, valid_parts);
    flat_dets = cat(2, pose_dets{:});
    
    % Measure fitness
    results(param_idx).pck_fit = fitness_pck(flat_dets, flat_gts, pck_thresholds);
    results(param_idx).pcp_fit = fitness_pcp(flat_dets, flat_gts, limbs);
    
    fprintf('Got scores %f (PCK), %f (PCP) for configuration %i/%i\n', ...
        results(param_idx).pck_fit, results(param_idx).pcp_fit, ...
        param_idx, length(param_configs));
end
end

function mean_pck = fitness_pck(flat_dets, flat_gts, pck_thresholds)
    % pcks is |pck_thresholds|-length cell array
    pcks = pck(flat_dets, flat_gts, pck_thresholds);
    % pck_mat will be a J*|pck_thresholds| matrix of PCKs
    pck_mat = cat(2, pcks{:});
    mean_pck = mean(pck_mat(:));
end

function mean_pcp = fitness_pcp(flat_dets, flat_gts, limbs)
    % pcks is |pck_thresholds|-length cell array
    all_pcps = pcp(flat_dets, flat_gts, {limbs.indices});
    assert(isvector(all_pcps) && ~isempty(all_pcps));
    mean_pcp = mean(all_pcps(:));
end
