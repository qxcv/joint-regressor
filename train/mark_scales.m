function [dataset, tsize] = mark_scales(dataset, subposes, step, template_scale, other_scales)
%MARK_SCALES Determine overall dataset scale and scales for each datum.
% - dataset should be a unified dataset with .pairs and .data attributes.
% - subpose_pa should be a parents array giving the locations of each subpose
% - step gives the downsampling factor of the classification CNN (e.g. /32
%   for VGGNet)
% - dataset will be annotated with scale_x and scale_y for each pair,
%   possibly along with some other stuff
% - tsize gives a template size for the dataset

if ~exist('other_scales', 'var')
    other_scales = [];
end

num_subposes = length(subposes);
num_pairs = length(dataset.pairs);

fprintf('Calculating scales for %i pairs with %i extra scales\n', ...
    num_pairs, length(other_scales));

for pair_idx=1:num_pairs
    pair = dataset.pairs(pair_idx);
    fst = dataset.data(pair.fst);
    snd = dataset.data(pair.snd);
    num_joints = size(fst.joint_locs, 1);
    all_joints = cat(1, fst.joint_locs, snd.joint_locs);
    subpose_sizes = zeros([1 num_subposes]);
    for subpose_idx=1:num_subposes
        inds = subposes(subpose_idx).subpose;
        all_inds = [inds, inds + num_joints];
        subpose_locs = all_joints(all_inds, :);
        bbox = get_bbox(subpose_locs);
        % bbox(3:4) is width and height
        patch_size = template_scale * max(bbox(3:4));
        assert(patch_size > 1);
        subpose_sizes(subpose_idx) = patch_size;
    end
    dataset.pairs(pair_idx).scale = max(subpose_sizes);
    assert(dataset.pairs(pair_idx).scale < 460);
    assert(isscalar(dataset.pairs(pair_idx).scale));
end

all_scales = [[dataset.pairs.scale] other_scales];
smallish_scale = quantile(all_scales, 0.01);
assert(isscalar(smallish_scale));
tsize = round(smallish_scale / step);
end