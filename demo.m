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

% We store a matrix of samples which we write out to a HDF5 file and reset
% once full.
first_datum = flic_data(1);
first_im = readim(first_datum);
width = size(first_im, 2);
height = size(first_im, 1);
channels = 8;
num_outputs = 2 * numel(first_datum.joint_locs);

data_mat = zeros(width, height, channels, conf.hdf5_samples);
label_mat = zeros(conf.hdf5_samples, num_outputs);
result_samples = 0;
h5_idx = 1;

for i=randperm(size(pairs, 1))
    fst_idx = pairs(i, 1);
    snd_idx = pairs(i, 2);
    fst = flic_data(fst_idx);
    snd = flic_data(snd_idx);
    
    result_samples = result_samples + 1;
    [stacked, labels] = get_stack(conf, fst, snd, 0, 0, 1.0, 0);
    data_mat(:, :, :, result_samples) = stacked;
    label_mat(:, result_samples) = labels;
        
    % Now write out to HDF5, if necessary
    if result_samples >= conf.hdf5_samples
        result_samples = 0;
        filename = h5_name(h5_idx);
        store2hdf5(filename, data_mat, label_mat, 1);
        h5_idx = h5_idx + 1;
    end
end

function name = h5_name(idx)
name = fullfile(conf.cache_dir, 'patches', sprintf('samples-%06i.h5', idx));