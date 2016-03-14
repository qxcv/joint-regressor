function model = train_model(conf, pos_val, neg_val, edge_means, tsize)
% IMPORTANT: pos_val and neg_val must be structs with .data and .pairs
% members (probably .name as well), as produced by unify_dataset.m
cachedir = conf.cache_dir;
subpose_pa = conf.subpose_pa;
subposes = conf.subposes;
cls = 'graphical_model';
try
    model = parload([cachedir cls], 'model');
catch
    % learn clusters, and derive labels
    % must have already been learnt!
    clusters = parload(fullfile(cachedir, 'centroids.mat'), 'centroids');
    % label_val = ...
    % XXX: As far as I can tell, this isn't used except to extract its
    % .near attribute below (which I'm not calculating at the moment).
    derive_labels(cachedir, subpose_pa, pos_val, clusters, ...
        subposes, edge_means, tsize);
    
    % XXX: Should pass this in more elegantly :/
    mean_pixels = load(fullfile(cachedir, 'mean_pixel.mat'));
    model = build_model(subpose_pa, conf.biposelet_classes, conf.cnn, ...
        mean_pixels, conf.interval, tsize);
%     % add near filed to provide mixture supervision
%     for ii = 1:numel(pos_val)
%         % XXX: Really need to add a .near field when I do my clustering :/
%         % Mine will probably contain *poselets* which are nearby to the
%         % best poselet (one vector of those for each subpose).
%         pos_val(ii).near = label_val(ii).near;
%     end
    assert(false, 'you need to fix train()');
    model = train(cls, model, pos_val, neg_val, 1);
    parsave([cachedir cls], model);
end
