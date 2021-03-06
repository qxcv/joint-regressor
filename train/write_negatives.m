function write_negatives(dataset, biposelets, patch_dir, ...
    cnn_window, aug, chunksz, subposes, filename)
%WRITE_NEGATIVES Analogue of write_dset for negative patches.
%Note that unlike write_dset, this function will intentionally avoid
%whatever poses are present in the images it is given (this functionality
%will only work on MPII cooking activities, where there is precisely one
%person present in each frame). This allows it to write some patches in
%which there are no people ("easy negatives") and some patches in which
%there are people, but no subposes with a reasonable matching centroid
%("hard negatives").

dest_path = fullfile(patch_dir, filename);
if exist(dest_path, 'file')
    fprintf('Negatives already exist at "%s", skipping\n', dest_path);
    return
end

% opts will be used later for writing to hdf5s
opts.chunksz = chunksz;
% beyond level 5, text data doesn't compress much; I assume it's the same
% for scientific data
% Edit: commented this out because for some reason enabling compression
% resulted in HUGE amounts of unaccounted space (like 30GiB unaccounted
% space for 500MiB data).
% opts.deflate = 5;
num_pairs = length(dataset.pairs);

if hasfield(dataset.data, 'joint_locs')
    have_joints = true;
else
    have_joints = false;
    % For datasets like INRIA Person we don't have any joint locations
    % because there are no people.
    % TODO: Sort out this behaviour. It's totally counter-intuitive at the
    % moment (passing a dataset without joint locations magically triggers
    % the totally different behaviour of ignoring aug.{easy,hard}_negs and
    % using aug.inria_negs instead).
    fprintf(['No joint locations present; will just write out %i easy ' ...
        'negatives per pair\n'], aug.inria_negs);
    if ~aug.inria_negs
        fprintf('aug.inria_negs == 0; existing data writer\n');
        return
    end
end

for pair_idx=1:num_pairs
    fprintf('Cropping pair %i/%i\n', pair_idx, num_pairs);
    pair = dataset.pairs(pair_idx);
    fst = dataset.data(pair.fst);
    snd = dataset.data(pair.snd);
    [im1, im2, flow] = get_pair_data(fst, snd);
    imstack = cat(3, im1, im2);
    
    % Start by getting a list of rectangles to crop
    imsize = size(im1);
    % [x y width height]
    pair_frame = [1 1 imsize([2 1])];
    
    if have_joints
        all_joints = cat(1, fst.joint_locs, snd.joint_locs);
        pose_box = get_bbox(all_joints);
        base_crop_size = pair.scale;
        % Get crops around the same scale as the acutal pose
        min_crop_size = 0.8 * base_crop_size;
        max_crop_size = 1.2 * base_crop_size;
        easy_crop_rects = random_nonint_rects(pair_frame, pose_box, min_crop_size, ...
            max_crop_size, aug.easy_negs);
        mean_l2_thresh = cnn_window(1) / 5; % XXX: This is pretty hacky
        hard_crop_rects = random_hard_rects(fst.joint_locs, snd.joint_locs, ...
            subposes, base_crop_size, cnn_window, aug.hard_negs, biposelets, ...
            mean_l2_thresh);
        crop_rects = double([easy_crop_rects; hard_crop_rects]);
    else
        % Make a crop size up! Yay!
        pose_box = [-1 -1 0 0];
        im_size = min(size(im1, 1), size(im1, 2));
        min_crop_size = min(im_size * 0.75, max(im_size / 3, 100));
        max_crop_size = im_size;
        easy_crop_rects = random_nonint_rects(pair_frame, pose_box, min_crop_size, ...
            max_crop_size, aug.inria_negs);
        crop_rects = double(easy_crop_rects);
    end
    assert(ismatrix(crop_rects) && size(crop_rects, 2) == 4);
    
    % Crop each rectangle in turn and write them as a batch
    lcr = size(crop_rects, 1);
    final_images = uint8(zeros([cnn_window(1:2) ...
                                size(im1, 3) + size(im2, 3) ...
                                lcr]));
    final_flow = zeros([cnn_window(1:2) 2 lcr]);
    parfor crop_num=1:lcr
        crop = crop_rects(crop_num, :);
        cropped_imstack = impcrop(imstack, crop);
        resized_imstack = imresize(cropped_imstack, cnn_window);
        % pmask tells permute() to swap h/w
        pmask = [2 1 3];
        final_images(:, :, :, crop_num) = permute(resized_imstack, pmask);
        
        cropped_flow = impcrop(flow, crop);
        resized_flow = imresize(cropped_flow, cnn_window);
        correct_flow = rescale_flow_mags(resized_flow, size(flow), size(resized_flow));
        final_flow(:, :, :, crop_num) = permute(correct_flow, pmask);
    end

    % All negatives are "class 1" in Matlab, which translates into class 0
    % in Python (and other languages with zero-based indexing); the
    % conversion is implicit in the one-of-k representation.
    class_labels = one_of_k(ones([1, lcr]), length(subposes)+1)';
    assert(isa(final_images, 'uint8'));
    
    % Produce fake joints for each part of the subpose
    joint_args = {};
    for i=1:length(subposes)
        subpose_name = subposes(i).name;
        subpose_idxs = subposes(i).subpose;
        num_vals = 4 * length(subpose_idxs);
        ds_name = sprintf('/%s', subpose_name);
        fake_data = zeros([num_vals size(final_images, 4)]);
        joint_args{length(joint_args)+1} = ds_name; %#ok<AGROW>
        joint_args{length(joint_args)+1} = fake_data; %#ok<AGROW>
    end

    store3hdf6(dest_path, opts, '/flow', single(final_flow), ...
        '/images', uint8(final_images), ...
        '/class', uint8(class_labels), ...
        joint_args{:});
end
end
