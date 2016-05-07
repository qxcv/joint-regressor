function write_dset(dataset, patch_dir, num_hdf5s, cnn_window, cnn_step, ...
    subposes, left_parts, right_parts, aug, chunksz)
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

% I'm using a nested for/parfor like Anoop suggested to parallelise
% augmentation calculation. This lets me write to a single file in without
% making everything sequential or resorting to locking hacks. The implicit
% barrier is annoying, but shouldn't matter too much given that
% augmentations are the same each time.
pool = gcp;
batch_size = pool.NumWorkers;

rem_pairs_path = fullfile(patch_dir, 'rem_pairs.mat');

try
    % XXX: This is suboptimal. Should have a unique name for pair cache.
    loaded = load(rem_pairs_path);
    rem_pairs = loaded.rem_pairs_trimmed;
    fprintf('Loaded remaining pairs from cache\n');
    fprintf('%i pairs left\n', length(rem_pairs));
catch ex
    if ~any(strcmp(ex.identifier, {'MATLAB:load:couldNotReadFile', 'MATLAB:nonExistentField'}))
        ex.rethrow();
    end
    rem_pairs = dataset.pairs;
    fprintf('No cached pairs; starting anew\n');
end

if isempty(rem_pairs)
    fprintf('No pairs to write; exiting\n');
    return;
end

assert(isstruct(rem_pairs) && isvector(rem_pairs));

for start_index = 1:batch_size:length(rem_pairs)
    true_batch_size = min(batch_size, length(rem_pairs) - start_index + 1);
    ds_data = dataset.data;
    results = struct('joint_labels', {}, 'stack', {}, 'subpose_num', {});
    
    % Calculate in parallel
    fprintf('Augmenting samples %i to %i\n', ...
        start_index, start_index + true_batch_size - 1);
    
    parfor result_index=1:true_batch_size
        mpii_index = start_index + result_index - 1;
        pair = rem_pairs(mpii_index); %#ok<PFBNS>
        fst = ds_data(pair.fst); %#ok<PFBNS>
        snd = ds_data(pair.snd);
        
        stack_start = tic;
        % Werd, yo
        deze_stackz = get_stacks(...
            fst, snd, pair.scale, subposes, left_parts, right_parts, ...
            cnn_window, cnn_step, aug);
        stack_time = toc(stack_start);
        fprintf('get_stack() took %fs\n', stack_time);
        results = [results deze_stackz];
    end
    
    fprintf('Got %i stacks\n', length(results));
    
    % Concatenate all stacks into a set of args that can be passed to
    % store3hdf6 together
    merge_start = tic;
    store3hdf6_args = get_store3hdf6_args(results, subposes);
    merge_time = toc(merge_start);
    fprintf('Merging together took %fs\n', merge_time);
    
    % Now write those args
    write_start = tic;
    h5_idx = randi(num_hdf5s);
    filename = fullfile(patch_dir, sprintf('samples-%06i.h5', h5_idx));
    store3hdf6(filename, opts, store3hdf6_args{:});
    
    % Write out the remaining pairs for resumable training
    rem_pairs_trimmed = rem_pairs(start_index+true_batch_size:end); %#ok<NASGU>
    save(rem_pairs_path, 'rem_pairs_trimmed');
    
    write_time = toc(write_start);
    fprintf('Writing %d examples took %fs\n', length(results), write_time);
end

fid = fopen(confirm_path, 'w');
fprintf(fid, '');
fclose(fid);
end

function out_args = get_store3hdf6_args(results, subposes)
% Concatenate a bunch of results into a set of arguments for store3hdf6. It
% would be easier to write one "stack" (a particular transformation of a
% training pair) at once, but this is much faster because it saves on a
% bunch of HDF5 overhead (something which Matlab makes much worse by
% exposing a stateless high-level HDF5 API instead of a stateful,
% class-based one).

assert(isstruct(results) && isstruct(subposes));

num_stacks = length(results);
num_subposes = length(subposes);
assert(num_stacks > 0, 'Need at least one example to write');

% Initialise the map and all of its keys
stack_dims = size(results(1).stack);
assert(numel(stack_dims) == 3 && stack_dims(3) == 8, 'Need RGBRGBFF (8-channel) stacks');

all_flow = single(zeros([stack_dims(1:2), 2, num_stacks]));
all_images = uint8(zeros([stack_dims(1:2), 6, num_stacks]));
all_classes = uint8(zeros([num_subposes + 1, num_stacks]));

all_subpose_data = cell([1 num_subposes]);
for subpose_idx=1:num_subposes
    % 2* for (x, y) and 2* for fact that there are two frames
    sp_vals = 2 * 2 * length(subposes(subpose_idx).subpose);
    all_subpose_data{subpose_idx} = single(zeros([sp_vals, num_stacks]));
end

for stack_num=1:num_stacks;
    stack = results(stack_num).stack;
    assert(size(stack, 3) == 8 && ndims(stack) == 3, ...
        'Need RGBRGBFF (8-channel) stacks');
    
    % We split the flow out from the images so that we can write the
    % images as uint8s. Most elegant solution would probably be to write
    % out images separately too and concatenate them in-network if
    % necessary, but oh well.
    stack_flow = single(stack(:, :, 7:8, :));
    all_flow(:, :, :, stack_num) = stack_flow;
    
    stack_im = stack(:, :, 1:6, :);
    stack_im_bytes = uint8(stack_im * 255);
    all_images(:, :, :, stack_num) = stack_im_bytes;
    
    % 1-of-K array of class labels. Ends up having dimension K*N,
    % where N is the unmber of samples and K is the number of
    % classes (i.e. number of subposes plus one for background
    % class).
    class_labels = uint8(one_of_k(results(stack_num).subpose_num + 1, ...
        length(subposes) + 1)');
    all_classes(:, stack_num) = class_labels;
    
    for subpose_idx=1:num_subposes
        if subpose_idx ~= results(stack_num).subpose_num;
            num_values = 4 * length(subposes(subpose_idx).subpose);
            subpose_data = zeros([num_values, 1]);
        else
            subpose_data = results(stack_num).joint_labels;
        end
        assert(size(subpose_data, 2) == 1);
        
        all_subpose_data{subpose_idx}(:, stack_num) = single(subpose_data);
    end
end

sp_args = cell([1, 2 * num_subposes]);
for subpose_idx=1:num_subposes
    ds_name = sprintf('/%s', subposes(subpose_idx).name);
    base = subpose_idx * 2 - 1;
    sp_args{base} = ds_name;
    this_data = all_subpose_data{subpose_idx};
    assert(isa(this_data, 'single'));
    sp_args{base+1} = this_data;
end

assert(isa(all_images, 'uint8') ...
    && isa(all_flow, 'single') ...
    && isa(all_classes, 'uint8'));
fixed_args = {'/images', all_images, '/flow', all_flow, '/class', all_classes};

out_args = [fixed_args sp_args];
end
