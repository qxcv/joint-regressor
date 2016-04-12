function accs = pck(preds, gts, threshs, norm_joints)
%PCK Compute Percentage Correct Keypoints (PCK), as per MODEC paper.
% See Section 6.1 of that paper. `preds` is cell array of predicted poses,
% `gts` is cell array of GT poses, `threshs` is array of thresholds,
% `norm_joints` is a pair indicating which two joints should be used for
% normalisation.
will_norm = exists(norm_joints, 'var');
if will_norm
    assert(length(norm_joints) == 2);
else
    warning('JointRegressor:pck:nonorms', ...
        'No normalisation joints provided, computing PCK at image scale');
end
assert(iscell(preds) && iscell(gts) && isnumeric(threshs));

% Find gt_norms, which gives a scale for each joint
pred_mat = cat(3, preds{:});
gt_mat = cat(3, gts{:});
if will_norm
    scale_diffs = squeeze(gt_mat(norm_joints(1), :, :) - gt_mat(norm_joints(2), :, :));
    scales = sqrt(sum(scale_diffs.^2, 1));
    assert(all(scales) > 0, 'Can''t have intersecting normalisation joints!');
else
    scales = ones(size(pred_mat, 1), length(preds));
end

% Now find distances for each joint
all_diffs = pred_mat - gt_mat;
all_norms = squeeze(sqrt(sum(all_diffs.^2, 2))) ./ scales;
assert(all(size(all_norms) == [size(pred_mat, 1), length(preds)]));

accs = zeros([1 length(threshs)]);
for tidx=1:length(threshs)
    thresh = threshs(tidx);
    accs(tidx) = sum(all_norms < thresh, 2) / size(all_norms, 2);
end
end

