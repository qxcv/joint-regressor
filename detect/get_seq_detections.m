function detections = get_seq_detections(dataset, seq_num, ssvm_model, ...
    biposelets, subposes, num_joints, num_results, cache_dir)
%GET_SEQ_DETECTIONS Evaluate on a single test sequence
% Differs from get_test_detections in that this only works on a single
% sequence, whereas get_test_detections works on a whole set of detections
seq = dataset.seqs{seq_num};

% Is it possible for joints to be marked invisible?
has_vis = hasfield(dataset.data, 'visible');
if has_vis
    % First, find bounding boxes for the entire sequence. This will be used
    % as a backup if there are no appropriate bounding boxes for a subpose
    % in a particular frame.
    seq_bboxes = get_subpose_bboxes({dataset.data(seq).joint_locs}, subposes);
    use_bbox_seq = ~any(isnan(seq_bboxes.xy(:)));
    if ~use_bbox_seq
        warning('JointRegressor:get_seq_detections:noBBox', ...
            'No bounding box available for sequence %i', seq_num);
    end
end

num_pairs = length(seq) - 1;
empt = @() cell([1 num_pairs]);
detections = struct('rscores', empt, 'recovered', empt);
for pair_idx = 1:num_pairs
    fprintf('Working on pair %i/%i...', pair_idx, num_pairs);
    % Where do we cache the results?
    boxes_save_path = fullfile(cache_dir, ...
        sprintf('test-boxes/seq%i/boxes-pair-%i.mat', seq_num, pair_idx));
    try
        boxes = parload(boxes_save_path, 'boxes');
        fprintf('Loaded boxes from file\n');
    catch
        fprintf('Recalculating boxes\n');
        idx1 = seq(pair_idx);
        idx2 = seq(pair_idx+1);
        im1_info = dataset.data(idx1);
        im2_info = dataset.data(idx2);
        fst_joints = im1_info.joint_locs;
        snd_joints = im2_info.joint_locs;
        
        if has_vis
            % Only bother using a bounding box around each subpose when all
            % joints are visible
            use_bbox = use_bbox_seq && (all(im1_info.visible | im2_info.visible));
        end
        
        if ~has_vis || use_bbox
            if has_vis
                % Replace bounding box for subpose with sequence bounding box if no
                % bounding box can be found for a subpose.
                % TODO: This code is rather pointless in light of the code
                % above
                bbox = get_subpose_bboxes({fst_joints, snd_joints}, subposes, ...
                    {im1_info.visible, im2_info.visible}, seq_bboxes);
            else
                bbox = get_subpose_bboxes({fst_joints, snd_joints}, subposes);
            end
            
            % The image will be cropped and scaled around the bbox we give it
            % if this is defined.
            bbox_args = {'BBox', bbox};
        else
            % If we don't have a bounding box for some subpose *anywhere in the
            % sequence* then we can't really crop reliably.
            bbox_args = {};
        end
        % This should work fine regardless of whether joints are invisible
        true_scale = calc_pair_scale(fst_joints, snd_joints, subposes, ...
            ssvm_model.template_scale);
        
        % Run the detector
        start = tic;
        [boxes, ~, ~] = detect(im1_info, im2_info, ssvm_model, ...
            'NumResults', num_results, 'CacheDir', cache_dir, ...
            'TrueScale', true_scale, bbox_args{:});
        time_taken = toc(start);
        
        save_dir = fileparts(boxes_save_path);
        mkdir_p(save_dir);
        save(boxes_save_path, 'boxes');
        
        % Debugging output
        assert(length(boxes) <= num_results, ...
            'Expected %i detections, got %i', num_results, length(boxes));
        p95_score = prctile([boxes.rscore], 0.95);
        max_score = max([boxes.rscore]);
        fprintf(' took %fs (95%% score %.4f, best %.4f)\n', ...
            time_taken, p95_score, max_score);
    end
    
    % Recover poses from biposelet detections
    % Originally stored `boxes` in `raw` field, but that took a LOT of space
    detections(pair_idx).rscores = [boxes.rscore];
    recovered = cell([1 length(boxes)]);
    window = ssvm_model.cnn.window;
    parfor det=1:length(boxes)
        % This will actually produce singles
        recovered{det} = boxes2pose(boxes(det), biposelets, ...
            window, subposes, num_joints);
    end
    detections(pair_idx).recovered = recovered; 
end
end
