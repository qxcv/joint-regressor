% From the README:
%
% The ground-truth poses for the dataset are provided as .mat files in 'gt_poses' directory.
% 
% The pose files are given as, pose_<TrackIndex>_<Index>.mat
% and the images are given as,  img_<TrackIndex>_<Index>.mat
% 
% <TrackIndex> is the index of the continuous image sequence (activity track)
% <Index> is just the image index in this evaluation set.
%
% Each pose file contains the location of 10 parts (torso and head each consists of two points),
%
%      pose(1,:) -> torso upper point
%      pose(2,:) -> torso lower point
%      pose(3,:) -> right shoulder
%      pose(4,:) -> left shoulder
%      pose(5,:) -> right elbow
%      pose(6,:) -> left elbow
%      pose(7,:) -> right wrist
%      pose(8,:) -> left wrist
%      pose(9,:) -> right hand
%      pose(10,:)-> left hand
%      pose(11,:)-> head upper point
%      pose(12,:)-> head lower point

function [train_dataset, val_dataset, test_seqs, tsize] = get_mpii_cooking(...
    dest_dir, cache_dir, dump_thresh, subposes, step, template_scale)
%GET_MPII_COOKING Fetches continuous pose estimation data from MPII
MPII_POSE_URL = 'http://datasets.d2.mpi-inf.mpg.de/MPIICookingActivities/poseChallenge-1.1.zip';
MPII_CONTINUOUS_URL = 'http://datasets.d2.mpi-inf.mpg.de/MPIICookingActivities/poseChallengeContinuous-1.0.zip';
CONTINUOUS_DEST_PATH = fullfile(dest_dir, 'mpii-cooking-pose-challenge-continuous');
POSE_DEST_PATH = fullfile(dest_dir, 'mpii-cooking-pose-challenge');
CONTINUOUS_CACHE_PATH = fullfile(cache_dir, 'poseChallengeContinuous-1.0.zip');
POSE_CACHE_PATH = fullfile(cache_dir, 'poseChallenge-1.1.zip');
% We only care about images with two frames in between them (so every third
% frame)
TRAIN_FRAME_SKIP = 2;
% Yeah, don't skip anything on the validation set because it's much lower
% frequency
VAL_FRAME_SKIP = 0;
% The continuous dataset frames between these two (inclusive) are
% mislabelled horribly, so I have to take them out.
CONT_EVIL_FRAMES = struct(...
    'start', {'pose_010890_000487.mat', 'pose_002871_001148.mat'}, ...
    'end', {'pose_003474_001093.mat', 'pose_009976_001186.mat'}, ...
    'diff', {606, 38});

data_path = fullfile(cache_dir, 'mpii_data.mat');
if exist(data_path, 'file')
    fprintf('Found existing data, so I''ll just use that\n');
    [train_dataset, val_dataset, test_seqs, tsize] = parload(data_path, ...
        'train_dataset', 'val_dataset', 'test_seqs', 'tsize');
    return
else
    fprintf('Need to regenerate all data :(\n');
end

% First we get the (much larger) continuous pose estimation dataset
if ~exist(CONTINUOUS_DEST_PATH, 'dir')
    assert(~~exist(CONTINUOUS_CACHE_PATH, 'file'));  % We can't download this since I don't know where it is :P
    if ~exist(CONTINUOUS_CACHE_PATH, 'file')
        fprintf('Downloading MPII continuous pose challenge from %s\n', MPII_CONTINUOUS_URL);
        websave(CONTINUOUS_CACHE_PATH, MPII_CONTINUOUS_URL);
    end
    fprintf('Extracting continuous MPII pose challenge data to %s\n', CONTINUOUS_DEST_PATH);
    unzip(CONTINUOUS_CACHE_PATH, CONTINUOUS_DEST_PATH);
end

% Next we grab the smaller original pose estimation dataset, which has a
% continuous (but low-FPS) training set but discontinuous testing set.
% We'll use the training set from that as our validation set.
if ~exist(POSE_DEST_PATH, 'dir')
    assert(~~exist(POSE_CACHE_PATH, 'file'));  % We can't download this since I don't know where it is :P
    if ~exist(POSE_CACHE_PATH, 'file')
        fprintf('Downloading MPII pose challenge from %s\n', MPII_POSE_URL);
        websave(POSE_CACHE_PATH, MPII_POSE_URL);
    end
    fprintf('Extracting basic MPII pose challenge data to %s\n', POSE_DEST_PATH);
    unzip(POSE_CACHE_PATH, POSE_DEST_PATH);
end

fprintf('Generating data\n');
train_data = load_files_continuous(CONTINUOUS_DEST_PATH, CONT_EVIL_FRAMES);
val_data = load_files_basic(POSE_DEST_PATH);

train_data = split_mpii_scenes(train_data, 0.2);
val_data = split_mpii_scenes(val_data, 0.1);

fprintf('Generating pairs\n');
train_pairs = find_pairs(train_data, TRAIN_FRAME_SKIP, dump_thresh);
val_pairs = find_pairs(val_data, VAL_FRAME_SKIP, dump_thresh);

% Combine into structs with .data and .pairs attributes
train_dataset = unify_dataset(train_data, train_pairs, 'train_dataset_mpii_cont');
val_dataset = unify_dataset(val_data, val_pairs, 'val_dataset_mpii_base');

% Write out scale data
[train_dataset, ~] = mark_scales(train_dataset, subposes, step, ...
    template_scale);
[val_dataset, tsize] = mark_scales(val_dataset, subposes, step, ...
    template_scale, [train_dataset.pairs.scale]);

% Grab sequences and split out a test set. Sequences must have 10 frames or
% more.
all_seqs = pairs2seqs(train_dataset, 10);

% Arbitrarily pick last 15 sequences as test set (should be enough? Maybe
% 200-300 frames in that lot)
num_test_seqs = 15; % TODO make this configurable
test_part = all_seqs(end-num_test_seqs+1:end);
lowest_frame = min([test_part{:}]);
highest_frame = max([test_part{:}]);

fprintf('Test set has %i seqs and %i frames\n', ...
    length(test_part), sum(cellfun(@length, test_part)));
test_seqs = make_test_set(train_dataset, test_part);

fprintf('Removing frames %i to %i (%i frames) from train set\n', ...
    lowest_frame, highest_frame, highest_frame - lowest_frame + 1);
old_np = train_dataset.num_pairs;
train_dataset = remove_pairs(train_dataset, lowest_frame, highest_frame);
fprintf('Went from %i training pairs to %i training pairs\n', ...
    old_np, train_dataset.num_pairs);

% Cache
save(data_path, 'train_dataset', 'val_dataset', 'test_seqs', 'tsize');
end

function rv = make_test_set(some_dataset, test_seqs)
rv = some_dataset;
rv.seqs = test_seqs;
rv = rmfield(rv, {'pairs', 'num_pairs'});
rv.name = [rv.name '_seqs_for_test'];
end

function ds = remove_pairs(ds, rstart, rend)
% Remove pairs from dataset ds between frame rstart and frame rend
in_range = @(vec) vec >= rstart & vec <= rend;
to_remove = in_range([ds.pairs.fst]) | in_range([ds.pairs.snd]);
ds.pairs = ds.pairs(~to_remove);
ds.num_pairs = length(ds.pairs);
end

function cont_data = load_files_continuous(dest_path, evil_frames)
pose_dir = fullfile(dest_path, 'data', 'gt_poses');
pose_fns = dir(pose_dir);
pose_fns = pose_fns(3:end); % Remove . and ..

cont_data = struct(); % Silences Matlab warnings about growing arrays
for fn_idx=1:length(pose_fns)
    data_fn = pose_fns(fn_idx).name;
    [track_index, index] = parse_continuous_fn(data_fn);
    % "track_index" is the index within the dataset for the action
    % track. "index" is the index within the dataset. They really could
    % have picked a more optimal name :/
    cont_data(fn_idx).frame_no = index;
    cont_data(fn_idx).action_track_index = track_index;
    file_name = sprintf('img_%06i_%06i.jpg', track_index, index);
    cont_data(fn_idx).image_path = fullfile(dest_path, 'data', 'images', file_name);
    cont_data(fn_idx).pose_fn = data_fn;
    loaded = load(fullfile(pose_dir, data_fn), 'pose');
    cont_data(fn_idx).joint_locs = loaded.pose;
    cont_data(fn_idx).is_val = false;
end
cont_data = sort_by_frame(cont_data);

% Exorcise evil frames
assert(length(cont_data) == length(pose_fns));
old_frame_count = length(cont_data);
good_mask = true([1 old_frame_count]);
total_evil = 0;

for evil_seq_num=1:length(evil_frames)
    evil_seq = evil_frames(evil_seq_num);
    pose_fns_strs = {cont_data.pose_fn};
    evil_start_idx = find(strcmp(evil_seq.start, pose_fns_strs));
    evil_end_idx = find(strcmp(evil_seq.end, pose_fns_strs));
    assert(isscalar(evil_start_idx) && isscalar(evil_end_idx), ...
        'Evil frames should appear once each');
    assert(evil_end_idx - evil_start_idx == evil_seq.diff, ...
        'Incorrect number of evil frames identified');
    good_mask(evil_start_idx:evil_end_idx) = false;
    total_evil = total_evil + evil_seq.diff + 1;
end

assert(total_evil >= length(evil_frames)); % just a sanity check
cont_data = cont_data(good_mask);
assert(old_frame_count - length(cont_data) == total_evil, ...
    'Incorrect number of evil frames removed');
end

function basic_data = load_files_basic(dest_path)
pose_dir = fullfile(dest_path, 'data', 'train_data', 'gt_poses');
pose_fns = dir(pose_dir);
pose_fns = pose_fns(3:end);
basic_data = struct(); % Silences Matlab warnings about growing arrays
for fn_idx=1:length(pose_fns)
    data_fn = pose_fns(fn_idx).name;
    frame_no = parse_basic_fn(data_fn);
    basic_data(fn_idx).frame_no = frame_no;
    file_name = sprintf('img_%06i.jpg', frame_no);
    basic_data(fn_idx).image_path = fullfile(dest_path, 'data', 'train_data', 'images', file_name);
    loaded = load(fullfile(pose_dir, data_fn), 'pose');
    basic_data(fn_idx).joint_locs = loaded.pose;
    basic_data(fn_idx).is_val = true;
end
basic_data = sort_by_frame(basic_data);
end

function sorted = sort_by_frame(data)
[~, sorted_indices] = sort([data.frame_no]);
sorted = data(sorted_indices);
end

function [track_index, index] = parse_continuous_fn(fn)
% Parse a filename like pose_<TrackIndex>_<Index>.mat
tokens = regexp(fn, '[^\d]*(\d+)_(\d+)', 'tokens');
assert(length(tokens) >= 1);
assert(length(tokens{1}) == 2);
track_index = str2double(tokens{1}{1});
index = str2double(tokens{1}{2});
end

function index = parse_basic_fn(fn)
tokens = regexp(fn, '[^\d]*(\d+)', 'tokens');
assert(length(tokens) >= 1);
assert(length(tokens{1}) == 1);
index = str2double(tokens{1}{1});
end

function seqs = pairs2seqs(dataset, min_length)
% Use known pair numbers to find contiguous sequences for same scene.
% Assumes that pairs were generated with a frame skip of 1, and accounting
% for scenes (so there are no pairs which cross scene boundaries). This is
% a safe assumption for this data, in this file, but may not hold
% elsewhere.
pair_idxs = [[dataset.pairs.fst]; [dataset.pairs.snd]]';
assert(ismatrix(pair_idxs) && size(pair_idxs, 2) == 2);
sorted_idxs = sortrows(pair_idxs);
assert(all(unique(sorted_idxs(:, 1)) == sorted_idxs(:, 1)), ...
    'Cannot have duplicate pair starts');
% Following should succeed because the frame skip is uniform
assert(all(unique(sorted_idxs(:, 2)) == sorted_idxs(:, 2)), ...
    'Cannot have duplicate pair ends');
assert(all(sorted_idxs(:, 1) < sorted_idxs(:, 2)), ...
    'First frame must have lower ID than second');

% Indexes from frame number to sequence number
seq_map = containers.Map('KeyType', 'double', 'ValueType', 'double');
seq_num = 1;
seqs = {};
for i=1:size(sorted_idxs, 1)
    start = sorted_idxs(i, 1);
    finish = sorted_idxs(i, 2);
    if seq_map.isKey(start)
        this_seq = seq_map(start);
        seq_map(finish) = seq_map(start);
    else
        this_seq = seq_num;
        seq_map(finish) = this_seq;
        seqs{this_seq} = start; %#ok<AGROW>
        seq_num = seq_num + 1;
    end
    seqs{this_seq} = [seqs{this_seq} finish]; %#ok<AGROW>
end

% Trim out sequences that are too small, assert that everything is sorted
% and sane.
is_sane = @(arr) ~isempty(arr) && all(arr == unique(arr));
assert(all(cellfun(is_sane, seqs)));
% Turns out that to do logical indexing into a cell array, you need to use
% parentheses. Huh.
seqs = seqs(cellfun(@length, seqs) >= min_length);

% Where sequences overlap, always choose the largest one
starts = cellfun(@min, seqs);
ends = cellfun(@max, seqs);
lengths = cellfun(@length, seqs);
should_keep = true([1 length(seqs)]);

for seq_num=1:length(starts)
    start = starts(seq_num);
    finish = ends(seq_num);
    this_length = lengths(seq_num);
    collides = (start <= ends) & (finish >= starts);
    collides(seq_num) = false;
    
    % Give priority to sequences which (a) are longer or (b) are equal
    % length and come before this one
    if any(lengths(collides) > this_length) ...
            || any(find(lengths == this_length & collides) < seq_num)
        should_keep(seq_num) = false;
    end
end

seqs = seqs(should_keep);
end

function pairs = find_pairs(data, frame_skip, dump_thresh)
% Find pairs with frame_skip frames between them
frame_nums = [data.frame_no];
scene_nums = [data.scene_num];
fst_inds = 1:(length(data)-frame_skip-1);
snd_inds = fst_inds + frame_skip + 1;
good_inds = (frame_nums(snd_inds) - frame_nums(fst_inds) == frame_skip + 1) ...
    & (scene_nums(fst_inds) == scene_nums(snd_inds));
dropped = 0;
for i=find(good_inds)
    fst = data(fst_inds(i));
    snd = data(snd_inds(i));
    if mean_dists(fst.joint_locs, snd.joint_locs) > dump_thresh
        fprintf('Dropping %s->%s (%i->%i)\n', ...
            fst.image_path, snd.image_path, fst_inds(i), snd_inds(i));
        good_inds(i) = false;
        dropped = dropped + 1;
    end
end
if dropped
    fprintf('Dropped %i pairs due to threshold %f\n', dropped, dump_thresh);
end
pairs = cat(2, fst_inds(good_inds)', snd_inds(good_inds)');
end