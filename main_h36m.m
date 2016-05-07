% Train and evaluate a model on the Human3.6M dataset

startup;
conf = get_conf_h36m;
[train_dataset, val_dataset, test_seqs] = get_h36m(...
    conf.dataset_dir, conf.cache_dir, conf.subposes, conf.cnn.step, ...
    conf.template_scale, conf.trans_spec);
main_generic(conf, train_dataset, val_dataset, test_seqs);
