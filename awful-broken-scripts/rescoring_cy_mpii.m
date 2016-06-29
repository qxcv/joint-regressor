function rescoring_cy_mpii(cy_pred_path)
%RESCORING_CY_MPII Prediction by re-scoring Chen & Yuille results
% Requires a model and set of CY predictions for each frame to exist
% already.

% Start by loading everything we need, so that errors are caught early.
fprintf('Loading configuration, model and existing predictions\n');
startup;
conf = get_conf_mpii;
% I was going to change the cache directory for safety, but I've decided I
% don't care.
% conf.cache_dir = ...
%     [regexprep(conf.cache_dir, [filesep '+$'], '') '-cy-rescore' filesep];
try
    [cy_preds, test_seqs] = ...
        parload(cy_pred_path, 'results', 'mpii_test_seqs');
    ssvm_model = ...
        parload(fullfile(conf.cache_dir, 'graphical_model.mat'), 'model');
    biposelets = ...
        parload(fullfile(conf.cache_dir, 'centroids.mat'), 'centroids');
catch e
    if ~isempty(regexp(e.identifier, '^MATLAB:load:', 'ONCE'))
        error('JointRegressor:rescoring_cy_mpii:loadError', ...
            ['Error loading data from cache: ' e.message]);
    else
        throw(e);
    end
end

% test_seqs were cached by CY code,so some paths (and possibly other
% things?) are different
test_seqs = convert_test_seqs(test_seqs);

fprintf('Converting cells\n');
cy_cells = to_pred_cells(cy_preds, conf.trans_spec);
fprintf('Getting pair detections\n');
pair_dets = to_pair_dets(cy_cells, test_seqs, ssvm_model, conf.subposes, ...
    biposelets, conf.cache_dir);

fprintf('Stitching sequences\n');
% Don't supply a cache directory to ensure that results are never cached
pose_dets = stitch_all_seqs(pair_dets, conf.num_stitch_dets, ...
    conf.stitch_weights, conf.valid_parts);

fprintf('Calculating statistics\n');
pose_gts = get_gts(test_seqs);
% Get joint transform spec and number of joints for original test seq data
t_ts = conf.test_trans_spec;
t_njs = size(test_seqs.data(1).orig_joint_locs, 1);
cells_to_flat = @(cells) cellfun(@(j) spec_trans_back(j, t_ts, t_njs), ...
    cells, 'UniformOutput', false);
results = cellfun(cells_to_flat, pose_dets, 'UniformOutput', false); %#ok<NASGU>
mkdir_p(conf.results_dir);
results_path = fullfile(conf.results_dir, 'cy-stitch-results.mat');
found_pose_dets = pose_dets; %#ok<NASGU>
found_pose_gts = pose_gts; %#ok<NASGU>
save(results_path, 'results', 'test_seqs', 'found_pose_dets', 'found_pose_gts');
fprintf('Saved to %s\n', results_path);

fprintf('Calculating original statistics\n');
cy_results_path = fullfile(conf.results_dir, 'cy-results-no-rescore.mat');
cy_best_dets = cell([1 length(cy_cells)]);
for seq_idx=1:length(cy_best_dets)
    orig_seq = cy_cells{seq_idx};
    cy_best_dets{seq_idx} = cell([1 length(orig_seq)]);
    for frame_idx=1:length(orig_seq)
        val = orig_seq{frame_idx}{1};
        assert(ismatrix(val) && isnumeric(val) && size(val, 2) == 2);
        cy_best_dets{seq_idx}{frame_idx} = val;
    end
end
cy_results = cellfun(cells_to_flat, cy_best_dets, 'UniformOutput', false); %#ok<NASGU>
save(cy_results_path, 'cy_results', 'test_seqs', 'cy_best_dets', 'found_pose_gts');
fprintf('Saved to %s\n', cy_results_path);
end

function pair_dets = to_pair_dets(cy_cells, test_seqs, ssvm_model, ...
    subposes, biposelets, cache_dir)
pair_dets = cell([1 length(test_seqs.seqs)]);
num_seqs = length(test_seqs.seqs);
pair_cache_dir = fullfile(cache_dir, 'cy-rescore-saved-pairs');
mkdir_p(pair_cache_dir);
for seq_idx=1:num_seqs
    seq = test_seqs.seqs{seq_idx};
    num_pairs = length(seq) - 1;
    empt = @() cell([1 num_pairs]);
    detections = struct('rscores', empt, 'recovered', empt);
    for fst_idx=1:num_pairs
        fprintf('Pair %i/%i (seq %i/%i)\n', fst_idx, num_pairs, ...
            seq_idx, num_seqs);
        cache_fn = sprintf('seq-%i-pair-%i.mat', fst_idx, seq_idx);
        cache_path = fullfile(pair_cache_dir, cache_fn);
        try
            [rscores, recovered ] = parload(cache_path, 'rscores', 'recovered');
            detections(fst_idx).rscores = rscores;
            detections(fst_idx).recovered = recovered;
            continue
        catch
            fprintf('Regenerating...\n');
        end
        d1 = test_seqs.data(seq(fst_idx));
        d2 = test_seqs.data(seq(fst_idx+1));
        fst_cy_preds = cy_cells{seq_idx}{fst_idx};
        snd_cy_preds = cy_cells{seq_idx}{fst_idx+1};
        pairs = cellprod(fst_cy_preds, snd_cy_preds);
        true_scale = calc_pair_scale(d1.joint_locs, d2.joint_locs, ...
            subposes, ssvm_model.template_scale);
        new_scores = score_pairs(pairs, d1, d2, true_scale, subposes, ...
            biposelets, ssvm_model, cache_dir);
        [~, best_idxs] = sort(new_scores, 'descend');
        top_idxs = best_idxs(1:100);
        rscores = new_scores(top_idxs);
        recovered = pairs(top_idxs);
        detections(fst_idx).rscores = rscores;
        detections(fst_idx).recovered = recovered;
        save(cache_path, 'rscores', 'recovered');
    end
    pair_dets{seq_idx} = detections;
end
end

function rv = to_pred_cells(orig_predictions, trans_spec)
% Convert cell of cells of structs (!!) to cell of cell of cells with right
% joint layout (transformed layout, not original layout)
rv = cell([1 length(orig_predictions)]);
parfor seq_num=1:length(orig_predictions)
    seq_structs = orig_predictions{seq_num};
    seq_cells = cell([1 length(seq_structs)]);
    for struct_idx=1:length(seq_structs)
        new_str = translate_points(seq_structs{struct_idx}, trans_spec);
        seq_cells{struct_idx} = {new_str.point};
    end
    rv{seq_num} = seq_cells;
end
end

function str = translate_points(str, trans_spec)
% Convert .poinrt/.score struct array to new layout
assert(isstruct(str));
for i=1:length(str)
    str(i).point = skeltrans(str(i).point, trans_spec);
    assert(~any(isnan(str(i).point(:))));
end
end

function test_seqs = convert_test_seqs(test_seqs)
for i=1:length(test_seqs.data)
    old_path = test_seqs.data(i).image_path;
    new_path = regexprep(old_path, '^./dataset/mpii/', './datasets/');
    assert(~~exist(new_path, 'file'), 'Image %s is missing', new_path);
    test_seqs.data(i).image_path = new_path;
end
end
