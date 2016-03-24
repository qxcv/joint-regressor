function model = train_model(conf, pos_val, neg_val, subpose_disps, tsize)
cachedir = conf.cache_dir;
subpose_pa = conf.subpose_pa;
subposes = conf.subposes;
cls = 'graphical_model';
try
    model = parload([cachedir cls], 'model');
catch
    clusters = parload(fullfile(cachedir, 'centroids.mat'), 'centroids');
    labels = derive_labels(cachedir, subpose_pa, pos_val, clusters, subposes, ...
        conf.biposelet_classes, conf.cnn.window);
    for pair_idx=1:length(pos_val.pairs)
        % We only care about .near because we use that for supervision
        pos_val.pairs(pair_idx).near = labels(pair_idx).near;
    end
    
    % XXX: Should pass this in more elegantly. Same goes for clusters.
    mean_pixels = load(fullfile(cachedir, 'mean_pixel.mat'));
    model = build_model(subpose_pa, conf.biposelet_classes, subpose_disps, ...
        conf.cnn, mean_pixels, conf.interval, tsize, conf.memsize);
    model = train(cls, model, pos_val, neg_val, 1);
    parsave([cachedir cls], model);
end
