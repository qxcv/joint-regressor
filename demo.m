% Self-contained demo script. After you download FLIC, you can run this,
% sit back and watch the blinkenlights :-)

function demo

startup;
conf = get_conf;
[flic_data, pairs] = get_flic(conf.dataset_dir, conf.cache_dir);

parfor i=1:size(pairs, 1)
    fst_idx = pairs(i, 1);
    snd_idx = pairs(i, 2);
    fst = flic_data(fst_idx);
    snd = flic_data(snd_idx);
    % Yeah, just compute the flow. We'll use it later.
    cached_imflow(fst, snd, conf.cache_dir);
end

% % We store a matrix of samples which we write out to a HDF5 file and reset
% % once full.
% first_datum = flic_data(1);
% first_im = readim(first_datum);
% width = size(first_im, 2);
% height = size(first_im, 1);
% channels = 8;
% num_outputs = 2 * numel(first_datum.joint_locs);
% 
% data_mat = zeros(width, height, channels, conf.hdf5_samples);
% label_mat = zeros(num_outputs, conf.hdf5_samples);
% result_samples = 0;
% h5_idx = 1;

% for i=randperm(size(pairs, 1))
%     fst_idx = pairs(i, 1);
%     snd_idx = pairs(i, 2);
%     fst = flic_data(fst_idx);
%     snd = flic_data(snd_idx);
%     im1 = readim(fst);
%     im2 = readim(snd);
%     % Make sure dims are right
%     assert(all(size(im1) == size(im2)));
%     assert(size(im1, 1) == size(flow, 1) && size(im1, 2) == size(flow, 2));
%     assert(size(flow, 3) == 2);
%     assert(size(im1, 3) == 3);
%     % Concatenate along channels, then permute to be width-first
%     result_samples = result_samples + 1;
%     data_mat(:, :, :, result_samples) = ...
%         permute(cat(norm_im(im1), ...
%                     norm_im(im2), ...
%                     norm_flow(flow), 3), ...
%                 [2 1 3]);
%     
%     % Now write out to HDF5, if necessary
%     if result_samples >= conf.hdf5_samples
%         result_samples = 0;
%         filename = h5_name(h5_idx);
%         store2hdf5(filename, data_mat, label_mat, 1);
%         h5_idx = h5_idx + 1;
%     end
% end

function normed = norm_im(im)
normed = single(im) / 255.0;

% Try to keep most flow in [-1, 1]
function normed = norm_flow(flow)
normed = single(flow) / max(conf.cnn_window);

function name = h5_name(idx)
name = fullfile(conf.cache_dir, 'patches', sprintf('samples-%06i.h5', idx));