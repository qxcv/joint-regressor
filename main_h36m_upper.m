% Train and evaluate a model on upper body in H36M dataset

startup;
conf = get_conf_h36m_upper;

% Set up the diary
diary_dir = 'diaries';
mkdir_p(diary_dir);
diary_out = fullfile(diary_dir, datestr(datetime, 'yyyy-mm-ddTHH:MM:SSZ'));
diary(diary_out);
echo on; % Ensures that commands in this file are written to diary
fprintf('Logging to %s\n', diary_out);

% Now do what we actually wanted to
getDataStart = tic();
[train_dataset, val_dataset, test_seqs] = get_h36m(...
    conf.dataset_dir, conf.cache_dir, conf.subposes, conf.cnn.step, ...
    conf.template_scale, conf.trans_spec, conf.h36m_keep_frac, ...
    conf.h36m_test);
fprintf('Getting data set took %fs\n', toc(getDataStart));
fprintf('Training set has %i pairs, validation set has %i pairs\n', ...
    train_dataset.num_pairs, val_dataset.num_pairs);

main_generic(conf, train_dataset, val_dataset, test_seqs);
