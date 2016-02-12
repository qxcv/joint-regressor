function write_dset(all_data, pairs, cache_dir, patch_dir, num_hdf5s, ...
    cnn_window, poselet, left_parts, right_parts, aug, chunksz)
%WRITE_DSET Write out a data set (e.g. pairs from the train set or pairs
%from the test set).

% opts will be used later for writing to hdf5s
opts.chunksz = chunksz;
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

% We use a POSIX semaphore wrapped by an external library to synchronise
% access to the file, since there's really no more elegant way of doing
% what I'm about to do. Note that I'm using a random key so that if this
% code crashes during the parfor and is run again then it will most likely
% not die or run into undefined behaviour on the 'create' call.
semaphore_key = randi(2^15-1);
fprintf('Creating semaphore with key %i\n', semaphore_key);
semaphore('create', semaphore_key, 1);
fprintf('Semaphore created\n');

parfor i=1:size(pairs, 1)
    fprintf('Working on pair %d/%d\n', i, size(pairs, 1), labindex);
    fst = all_data(pairs(i, 1));
    snd = all_data(pairs(i, 2));
    
    stack_start = tic;
    stacks = get_stacks(...
        fst, snd, poselet, left_parts, right_parts, cache_dir, cnn_window, ...
        aug.flips, aug.rots, aug.scales, aug.randtrans);
    stack_time = toc(stack_start);
    fprintf('get_stack() took %fs\n', labindex, stack_time);
    
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
        
        % Write!
        assert(size(stack, 3) == 8, 'you need to rewrite this to handle flow');
        % We split the flow out from the images so that we can write the
        % images as uint8s
        stack_flow = single(stack(:, :, 7:8, :));
        stack_im = stack(:, :, 1:6, :);
        stack_im_bytes = uint8(stack_im * 255);
        semaphore('wait', semaphore_key);
        store3hdf6(filename, opts, '/flow', stack_flow, ...
                                   '/images', stack_im_bytes, ...
                                   '/joints', joint_labels);
        semaphore('post', semaphore_key);
    end
    write_time = toc(write_start);
    fprintf('Writing %d examples took %fs\n', length(stacks), labindex, write_time);
end

fprintf('Destroying semaphore\n');
semaphore('destroy', semaphore_key);
fprintf('Semaphore destroyed\n');

fid = fopen(confirm_path, 'w');
fprintf(fid, '');
fclose(fid);
end