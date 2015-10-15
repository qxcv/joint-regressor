% Self-contained demo script. After you download FLIC, you can run this,
% sit back and watch the blinkenlights :-)

function demo

startup;
conf = get_conf;
[flic_data, pairs] = get_flic(conf.dataset_dir, conf.cache_dir);

% Just cache the flow. We'll use it later.
parfor i=1:size(pairs, 1)
    fst_idx = pairs(i, 1);
    snd_idx = pairs(i, 2);
    fst = flic_data(fst_idx);
    snd = flic_data(snd_idx);
    cached_imflow(fst, snd, conf.cache_dir);
end

patch_dir = fullfile(conf.cache_dir, 'patches');
if ~exist(patch_dir, 'dir')
    mkdir(patch_dir);
end
    
for i=1:size(pairs, 1)
    fprintf('Working on pair %d/%d\n', i, size(pairs, 1));
    fst = flic_data(pairs(i, 1));
    snd = flic_data(pairs(i, 2));
    
    stack_start = tic;
    stacks = get_stacks(...
        conf, fst, snd, conf.aug.flips, conf.aug.rots, conf.aug.scales, ...
        conf.aug.randtrans);
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
        h5_idx = randi(conf.num_hdf5s);
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