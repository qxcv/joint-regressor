function scores = score_pairs(pairs, d1, d2, true_scale, subposes, ...
    biposelets, ssvm_model, cache_dir)
%SCORE_PAIRS Scores a cell array of pose pairs for a given datum pair
pair_valid = @(p) iscell(p) && length(p) == 2;
assert(iscell(pairs), 'Pairs array must be array of cell pairs');
assert(all(cellfun(pair_valid, pairs)), 'Pairs must be length-2 cell arrays');

im1 = readim(d1); im2 = readim(d2);
% I cache flow in detect.m, but I think it's sufficiently fast that it
% doesn't really matter here.
flow = imflow(im1, im2);
im_stack = cat(3, im1, im2);
% Need to rescale the pair so that the GT scale is in the middle. Don't
% both supplying a bounding box (since it's complicated to compute and
% liable to cover the whole image, negating any computational benefit).
[im_stack, flow, ~, cs_xtrim, cs_ytrim, cs_scale] = ...
    cropscale_pos(im_stack, flow, [], ssvm_model.cnn.window(1), true_scale);

% Calculate image pyramid, saving to speed things up
cnn_save_fn = sprintf('pyra-%s-to-%s.mat', dname(d1), dname(d2));
cnn_save_path = fullfile(cache_dir, 'scored-pose-pyra', cnn_save_fn);
cnn_args = {im_stack, flow, ssvm_model, cnn_save_path};
cnnStart = tic;
[pyra, unary_map] = imCNNdet(cnn_args{:});
cnnStop = toc(cnnStart);
fprintf('imCNNdet took %fs\n', cnnStop);

scales = ssvm_model.pyra_scales;
scores = nan([1 length(pairs)]);
unflat_biposelets = unflatten_all_biposelets(biposelets);
cnnpar = ssvm_model.cnn;
pairLoopStart = tic;
parfor pair_idx=1:length(pairs)
    pair = pairs{pair_idx};
    trans_pair = ...
        cellfun(@(j) bsxfun(@minus, j, [cs_xtrim, cs_ytrim]), pair, ...
                'UniformOutput', false);
    dists = nan([1 length(scales)]);
    pair_scores = nan([1 length(scales)]);
    
    for scale_idx=1:length(scales)
        pyra_scale = scales(scale_idx);
        total_scale = pyra_scale * cs_scale;
        assert(isscalar(total_scale));
        to_scale = @(j) (j - 1) * total_scale + 1;
        tp1 = to_scale(trans_pair{1});
        tp2 = to_scale(trans_pair{2});
        scaled_bp = cellfun(to_scale, unflat_biposelets, 'UniformOutput', false);
        
        % sp_locs gives top left corner of biposelet
        [sp_types, sp_locs, dists(scale_idx)] = ...
            get_subposes(tp1, tp2, subposes, scaled_bp);
        
        % Now convert sp_locs to unary_map coordinates
        map_locs = ...
            image_to_map_loc(sp_locs, pyra(scale_idx), cnnpar); %#ok<PFBNS>

        % Extract a score
        pair_scores(scale_idx) = score_types_locs(sp_types, map_locs, ...
            unary_map{scale_idx}, ssvm_model); %#ok<PFBNS>
    end

    % The score 
    assert(~any(isnan(dists)) && ~any(isnan(pair_scores)));
    [~, best_idx] = min(dists);
    scores(pair_idx) = pair_scores(best_idx);
end
pairLoopStop = toc(pairLoopStart);
fprintf('Pair loop took %fs\n', pairLoopStop);

assert(~any(isnan(scores)));
end

function uf_bp_cells = unflatten_all_biposelets(bp_cells)
assert(isvector(bp_cells) && iscell(bp_cells));
uf_bp_cells = cell([1 length(bp_cells)]);
for sp_idx=1:length(bp_cells)
    flat = bp_cells{sp_idx};
    num_types = size(flat, 1);
    num_joints = size(flat, 2) / 2;
    unflat = nan([num_joints, 2, num_types]);
    for bp_idx=1:num_types
        unflat(:, :, bp_idx) = unflatten_coords(flat(bp_idx, :));
    end
    assert(~any(isnan(unflat(:))));
    uf_bp_cells{sp_idx} = unflat;
end
end
