function total_pck_table = clustering_fun(experiment)
%CLUSTERING_FUN Comparison of clustering methods and parameters
startup;
conf = get_conf_mpii;

% Experiment parameters
default_exp.num_classes = conf.biposelet_classes;
default_exp.cnn_step = conf.cnn.step;
if nargin < 1
    experiment = default_exp;
else
    in_fields = fieldnames(experiment);
    def_fields = fieldnames(default_exp);
    
    % First, make sure that all in_fields are also default_exp fields
    assert(isempty(setdiff(in_fields, def_fields)), ...
        'Unknown fields specified in input configuration');
    
    % Now extend experiment with missing fields
    missing = setdiff(def_fields, in_fields);
    for m_idx=1:length(missing)
        m_name = missing{m_idx};
        experiment.(m_name) = default_exp.(m_name);
    end
end
display(experiment);

% Now set other parameters
cnnpar = conf.cnn;
cnnpar.step = experiment.cnn_step;

% Actual experiment code
fprintf('Loading data\n');
train_patch_dir = fullfile(conf.cache_dir, 'train-patches');
train_h5s = files_with_extension(train_patch_dir, '.h5');
% We will only use the validation dataset for testing
[~, val_dataset, ~] = get_mpii_cooking(...
    conf.dataset_dir, conf.cache_dir, conf.pair_mean_dist_thresh, ...
    conf.subposes, cnnpar.step, conf.template_scale, conf.trans_spec);

fprintf('Generating centroids\n');
centStart = tic;
biposelets = generate_centroids_from_h5s(...
    train_h5s, conf.subposes, experiment.num_classes);
fprintf('Centroid generation took %fs\n', toc(centStart));
unflat_biposelets = unflatten_all_biposelets(biposelets);
num_pairs = length(val_dataset.pairs);
guesses = cell([1, 2 * num_pairs]);
truths = cell([1, 2 * num_pairs]);

fake_seqs = cell([1 num_pairs]);
fake_dets = cell([1 num_pairs]);
for pair_idx=1:num_pairs
    out_idx = 2 * pair_idx - 1;
    pair = val_dataset.pairs(pair_idx);
    d1 = val_dataset.data(pair.fst);
    d2 = val_dataset.data(pair.snd);
    
    % TODO: May need to figure out a more realistic way of associating
    % validation samples with biposelets, since in reality the CNN is going
    % to be trained to predict the "nearest" (in terms of L2^2) biposelet
    % to each subpose.
    [guesses{out_idx}, guesses{out_idx+1}] = ...
        best_effort_subposes(pair, d1, d2, conf.subposes, ...
                             unflat_biposelets, biposelets, ...
                             conf.pyra.scales, cnnpar);
    truths{out_idx} = d1.joint_locs;
    truths{out_idx+1} = d2.joint_locs;
    
    % These may be used for visualisation using headless_detection_vis
    fake_seqs{pair_idx} = [pair.fst, pair.snd];
    fake_dets{pair_idx} = guesses(out_idx:out_idx+1);
end

% Really hacky stats code
all_pcps = pcp(guesses, truths, {conf.limbs.indices});
fprintf('Limb\tPCP\n');
for limb_idx=1:length(conf.limbs)
    fprintf('%s\t%.2f\n', conf.limbs(limb_idx).names, all_pcps(limb_idx));
end
fprintf('\n');

pck_thresholds = [5 10:10:100];
all_pck_joints = containers.Map(...
    {'Total'}, ...
    {1:conf.num_joints} ...
);
fprintf('PCK at various thresholds:\n');
all_pcks = pck(guesses, truths, pck_thresholds);
pck_table = format_pcks(all_pcks, pck_thresholds, conf.pck_joints);
disp(pck_table);
total_pck_table = format_pcks(all_pcks, pck_thresholds, all_pck_joints);
disp(total_pck_table);

% XXX: this is wayyyy hacky. Need to make save path more sensible
% vis_dest_dir = '/home/sam/delete-me/clustering-fun-vis/';
% fprintf('Visalising (saving to %s)\n', vis_dest_dir);
% fake_ds.data = val_dataset.data;
% fake_ds.seqs = fake_seqs;
% headless_detection_vis(fake_ds, fake_dets, vis_dest_dir);
end
