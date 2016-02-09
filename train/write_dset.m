function write_dset(all_data, pairs, cache_dir, patch_dir, num_hdf5s, cnn_window, poselet, aug)
%WRITE_DSET Write out a data set (e.g. pairs from the train set or pairs
%from the test set).

confirm_path = fullfile(patch_dir, '.written');

if ~exist(patch_dir, 'dir')
    mkdir(patch_dir);
else
    if exist(confirm_path, 'file')
        % Don't re-write
        fprintf('Patches in %s already exist; skipping\n', patch_dir);
        return
    end
end

% Just cache the flow. We'll use it later.
parfor i=1:size(pairs, 1)
    fst_idx = pairs(i, 1);
    snd_idx = pairs(i, 2);
    fst = all_data(fst_idx);
    snd = all_data(snd_idx);
    cached_imflow(fst, snd, cache_dir);
end
    
for i=1:size(pairs, 1)
    fprintf('Working on pair %d/%d\n', i, size(pairs, 1));
    fst = all_data(pairs(i, 1));
    snd = all_data(pairs(i, 2));
    
    stack_start = tic;
    stacks = get_stacks(...
        fst, snd, poselet, cache_dir, cnn_window, aug.flips, aug.rots, ...
        aug.scales, aug.randtrans);
    stack_time = toc(stack_start);
    fprintf('get_stack() took %fs\n', stack_time);
    
    write_start = tic;
    for j=1:length(stacks)
        % Get stack and labels; we don't add in dummy dimensions because
        % apparently Matlab can't tell the difference between a
        % j*k*l*1*1*1*1... matrix and a j*k*l matrix.
        stack = stacks(j).stack;
        joint_labels = stacks(j).joint_labels;
        
        % Choose a file, regardless of whether it exists
        h5_idx = randi(num_hdf5s);
        filename = fullfile(patch_dir, sprintf('samples-%06i.h5', h5_idx));
        create = false;
        if ~exist(filename, 'file')
            create = true;
        end
        
        % Write!
        store2hdf5(filename, stack, joint_labels, create);
    end
    write_time = toc(write_start);
    fprintf('Writing %d examples took %fs\n', length(stacks), write_time);
end

fid = fopen(confirm_path, 'w');
fprintf(fid, '');
fclose(fid);
end

