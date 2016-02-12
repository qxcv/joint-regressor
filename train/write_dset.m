function write_dset(all_data, pairs, cache_dir, patch_dir, num_hdf5s, cnn_window, poselet, left_parts, right_parts, aug)
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

% We use a POSIX semaphore wrapped by an external library to synchronise
% access to the file, since there's really no more elegant way of doing
% what I'm about to do. Note that I'm using a random key so that if this
% code crashes during the parfor and is run again then it will most likely
% not die or run into undefined behaviour on the 'create' call.
semaphore_key = randi(2^31-1);
fprintf('Creating semaphore with key %i\n', semaphore_key);
semaphore('create', semaphore_key, 1);
fprintf('Semaphore created\n');
    
% TODO: Change this to parfor once I know the code is working
for i=1:size(pairs, 1)
    fprintf('Working on pair %d/%d on lab %i\n', i, size(pairs, 1), labindex);
    fst = all_data(pairs(i, 1));
    snd = all_data(pairs(i, 2));
    
    stack_start = tic;
    stacks = get_stacks(...
        fst, snd, poselet, left_parts, right_parts, cache_dir, cnn_window, ...
        aug.flips, aug.rots, aug.scales, aug.randtrans);
    stack_time = toc(stack_start);
    fprintf('get_stack() on lab %i took %fs\n', labindex, stack_time);
    
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
        fprintf('Lab %i locking semaphore\n', labindex);
        semaphore('wait', semaphore_key);
        fprintf('Lab %i locked semaphore successfully\n', labindex);
        store3hdf6(filename, {}, '/data', stack, '/label', joint_labels);
        fprintf('Lab %i releasing semaphore\n', labindex);
        semaphore('post', semaphore_key);
        fprintf('Lab %i released semaphore successfully\n', labindex);
    end
    write_time = toc(write_start);
    fprintf('Writing %d on lab %i examples took %fs\n', length(stacks), labindex, write_time);
end

fprintf('Destroying semaphore\n');
semaphore('destroy', semaphore_key);
fprintf('Semaphore destroyed\n');

fid = fopen(confirm_path, 'w');
fprintf(fid, '');
fclose(fid);
end