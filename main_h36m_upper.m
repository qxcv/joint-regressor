% Train and evaluate a model on upper body in H36M dataset

startup;
conf = get_conf_h36m_upper;
[train_dataset, val_dataset, test_seqs] = get_h36m(...
    conf.dataset_dir, conf.cache_dir, conf.subposes, conf.cnn.step, ...
    conf.template_scale, conf.trans_spec);
fprintf('Training set has %i pairs, validation set has %i pairs\n', ...
    train_dataset.num_pairs, val_dataset.num_pairs);
main_generic(conf, train_dataset, val_dataset, test_seqs);
