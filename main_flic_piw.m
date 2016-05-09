% Train a model on Frames Labelled In Cinema (FLIC) and evaluate it on
% Poses in the Wild (PIW)

startup;
conf = get_conf_flic_piw;
[flic_train_dataset, flic_val_dataset] = get_flic(...
    conf.dataset_dir, conf.cache_dir, conf.subposes, conf.cnn.step, ...
    conf.template_scale);
piw_test_seqs = get_piw(conf.dataset_dir, conf.cache_dir);
main_generic(conf, flic_train_dataset, flic_val_dataset, piw_test_seqs);
