function write_poselet_dset(flic_data, frames, cache_dir, patch_dir, num_hdf5s, cnn_window, aug)
%WRITE_POSELET_DSET Write out dataset for poselet-based regressor
if ~exist(patch_dir, 'dir')
    mkdir(patch_dir);
end

parfor i=frames
    fprintf('Working on frame %d', i);
    datum = flic_data(i);
    
    stack_start = tic;
    stacks = get_stacks(...
        datum, [], cache_dir, cnn_window, aug.flips, aug.rots, ...
        aug.scales, aug.randtrans);
    stack_time = toc(stack_start);
    fprintf('get_stack() took %fs\n', stack_time);
    
    write_start = tic;
    for j=1:length(stacks)
        % Get stack and labels; we don't add in dummy dimensions because
        % apparently Matlab can't tell the difference between a
        % j*k*l*1*1*1*1... matrix and a j*k*l matrix.
        stack = stacks(j).stack;
        labels = stacks(j).labels;
        
        % Choose a file, regardless of whether it exists
        h5_idx = randi(num_hdf5s);
        filename = fullfile(patch_dir, sprintf('samples-%06i.h5', h5_idx));
        create = 0;
        if ~exist(filename, 'file')
            create = 1;
        end
        
        % Write!
        store2hdf5(filename, stack, labels, create);
    end
    write_time = toc(write_start);
    fprintf('Writing %d examples took %fs\n', length(stacks), write_time);
end
end

