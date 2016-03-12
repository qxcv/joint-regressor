function model = train_model(cachedir, subpose_pa, pos_val, neg_val, tsize)
% IMPORTANT: pos_val and neg_val must be structs with .data and .pairs
% members (probably .name as well), as produced by unify_dataset.m
cls = 'graphical_model';
try
    model = parload([cachedir cls], 'model');
catch
    % learn clusters, and derive labels
    % must have already been learnt!
    clusters = parload(fullfile(cachedir, 'centroids.mat'), 'centroids');
    label_val = derive_labels(cachedir, subpose_pa, clusters, pos_val, tsize);
    assert(false, 'you need to fix build_model');
    
    model = build_model(subpose_pa, clusters, tsize);
    % add near filed to provide mixture supervision
    for ii = 1:numel(pos_val)
        % XXX: Really need to add a .near field when I do my clustering :/
        pos_val(ii).near = label_val(ii).near;
    end
    model = train(cls, model, pos_val, neg_val, 1);
    parsave([cachedir cls], model);
end
