% Train and evaluate a model on the MPII dataset

startup;
conf = get_conf_mpii;
[train_dataset, val_dataset, test_seqs] = get_mpii_cooking(...
    conf.dataset_dir, conf.cache_dir, conf.pair_mean_dist_thresh, ...
    conf.subposes, conf.cnn.step, conf.template_scale, conf.trans_spec);
main_generic(conf, train_dataset, val_dataset, test_seqs);
