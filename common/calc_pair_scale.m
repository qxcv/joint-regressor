function datum_scale = calc_pair_scale(fst_locs, snd_locs, subposes, template_scale)
%CALC_PAIR_SCALE Calculate scale of datum based on joint positions
% Used in mark_scales.m and also in the testing pipeline
assert(ismatrix(fst_locs) && ismatrix(snd_locs) ...
    && isstruct(subposes) && isscalar(template_scale));

num_joints = size(fst_locs, 1);
all_joints = cat(1, fst_locs, snd_locs);
assert(size(all_joints, 2) == 2 && numel(all_joints) == 2 * numel(fst_locs));
subpose_sizes = zeros([1 length(subposes)]);

for subpose_idx=1:length(subposes)
    inds = subposes(subpose_idx).subpose;
    all_inds = [inds, inds + num_joints];
    subpose_locs = all_joints(all_inds, :);
    bbox = get_bbox(subpose_locs);
    assert(numel(bbox) == 4);
    if any(isnan(bbox))
        assert(all(isnan(subpose_locs(:))), ...
            'All joints should be nans if bounding box is');
        subpose_sizes(subpose_idx) = nan;
        continue
    end
    % bbox(3:4) is width and height
    patch_size = max(bbox(3:4));
    assert(patch_size > 1);
    subpose_sizes(subpose_idx) = patch_size;
end

datum_scale = round(template_scale * max(subpose_sizes));

assert(isscalar(datum_scale) && ~isnan(datum_scale));
end

