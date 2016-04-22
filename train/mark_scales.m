function [dataset, tsize] = mark_scales(dataset, subposes, step, template_scale, other_scales)
%MARK_SCALES Determine overall dataset scale and scales for each datum.
% - dataset should be a unified dataset with .pairs and .data attributes.
% - subpose_pa should be a parents array giving the locations of each subpose
% - step gives the downsampling factor of the classification CNN (e.g. /32
%   for VGGNet)
% - template_scale is a small factor (e.g. 1.15) used to increase the
%   calculated scale so that bboxes calculated based on the scale are
%   sufficiently large.
% - other_scales can be used to pass in .scale attributes calculated from
%   other datasets so that there is more data available to calculate .tsize
%   (which is like the X-th percentile of scales or something divided by
%   the CNN step)

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
    new_scale = calc_pair_scale(fst.joint_locs, snd.joint_locs, ...
        subposes, template_scale);
    assert(isscalar(new_scale));
    dataset.pairs(pair_idx).scale = new_scale;
end

all_scales = [[dataset.pairs.scale] other_scales];
smallish_scale = quantile(all_scales, 0.01);
assert(isscalar(smallish_scale));
tsize = round(smallish_scale / step);
end
