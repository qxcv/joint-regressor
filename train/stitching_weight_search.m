function best_params = stitching_weight_search(pair_dets, valid_parts, pose_gts)
%STITCHING_WEIGHT_SEARCH Naively find good stitching weights
% Uses grid search or randomised search internally (haven't decided which
% yet).

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
config_pcks = nan([1, count]);
parfor param_idx=1:length(param_configs)
    fprintf('Testing configuration %i/%i\n', param_idx, length(param_configs));
    current_params = param_configs(param_idx);
    config_pcks(param_idx) = fitness(pair_dets, current_params, ...
        valid_parts, flat_gts, pck_thresholds);
    fprintf('Got score %f for configuration %i/%i\n', config_pcks(param_idx));
end

assert(~any(isnan(config_pcks)));
assert(all(0 <= config_pcks && config_pcks <= 1));
[best_pck, best_param_idx] = max(config_pcks);
assert(best_pck > 0, 'Uh, that''s not good');
fprintf('Best PCK: %f\n', best_pck);
best_params = param_configs(best_param_idx);
end

function mean_pck = fitness(pair_dets, current_params, valid_parts, flat_gts, pck_thresholds)
    pose_dets = stitch_all_seqs(pair_dets, current_params, valid_parts);
    flat_dets = cat(2, pose_dets{:});
    % pcks is |pck_thresholds|-length cell array
    pcks = pck(flat_dets, flat_gts, pck_thresholds);
    % pck_mat will be a J*|pck_thresholds| matrix of PCKs
    pck_mat = cat(2, pcks{:});
    mean_pck = mean(pck_mat(:));
end
