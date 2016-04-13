function pcps = pcp(preds, gts, limbs)
%PCP Compute Percentage Correct Parts (strict) on input sequences
assert(iscell(preds) && iscell(gts));
assert(length(preds) >= 1);
assert(length(preds) == length(gts));
assert(ismatrix(preds{1}) && size(preds{1}, 2) == 2);

pred_mat = cat(3, preds{:});
gt_mat = cat(3, gts{:});
pcps = nan([1 limbs]);

parfor limb_id=1:length(limbs)
    limb = limbs{limb_id};
    start_gts = squeeze(gt_mat(limb(1), :, :));
    end_gts = squeeze(gt_mat(limb(2), :, :));
    lengths = dists_2d(start_gts, end_gts);
    threshs = lengths / 2;
    
    start_preds = squeeze(pred_mat(limb(1), :, :));
    end_preds = squeeze(pred_mat(limb(2), :, :));
    start_dists = dists_2d(start_preds, start_gts);
    end_dists = dists_2d(end_preds, end_gts);
    
    valid = (start_dists < threshs) & (end_dists < threshs);
    assert(length(valid) == length(preds));
    pcps(limb_id) = sum(valid) / length(valid);
end
end

function dists = dists_2d(mat1, mat2)
% Used to compute coordinatewise dists between matrices of 2D coordinates
assert(ismatrix(mat1) && ismatrix(mat2) && size(mat1, 1) == 2 ...
    && all(size(mat1) == size(mat2)));
dists = sqrt(sum((mat2 - mat1).^2, 1));
end
