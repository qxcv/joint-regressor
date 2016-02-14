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

function [mpii_data, train_pairs, val_pairs] = get_mpii_cooking(dest_dir, cache_dir)
%GET_MPII_COOKING Fetches continuous pose estimation data from MPII
% To be honest, I have no idea where I got this dataset from, so it just
% does some mystery symlinking.
MPII_URL = 'XXX';
DEST_PATH = fullfile(dest_dir, 'mpii-cooking-pose-challenge-continuous');
% CACHE_PATH = fullfile(cache_dir, 'poseChallengeContinuous-1.0.zip');
CACHE_PATH = '/home/sam/etc/mpii-cooking-activities/poseChallengeContinuous-1.0.zip';
% We only care about images with two frames in between them (so every third
% frame)
FRAME_SKIP = 2;

if ~exist(DEST_PATH, 'dir')
    assert(~~exist(CACHE_PATH, 'file'));  % We can't download this since I don't know where it is :P
    if ~exist(CACHE_PATH, 'file')
        fprintf('Downloading MPII cooking from %s\n', MPII_URL);
        websave(CACHE_PATH, MPII_URL);
    end
    fprintf('Extracting MPII cooking data to %s\n', DEST_PATH);
    unzip(CACHE_PATH, DEST_PATH);
end

data_path = fullfile(cache_dir, 'mpii_data.mat');
% regen_pairs tells us whether we should regenerate pairs regardless of
% whether the file exists.
regen_pairs = false;
if ~exist(data_path, 'file')
    fprintf('Regenerating data\n');
    pose_dir = fullfile(DEST_PATH, 'data', 'gt_poses');
    pose_fns = dir(pose_dir);
    pose_fns = pose_fns(3:end); % Remove . and ..
    mpii_data = struct(); % Silences Matlab warnings about growing arrays
    for i=1:length(pose_fns)
        data_fn = pose_fns(i).name;
        [track_index, index] = parse_fn(data_fn);
        % "track_index" is the index within the dataset for the action
        % track. "index" is the index within the dataset. They really could
        % have picked a more optimal name :/
        mpii_data(i).frame_no = index;
        mpii_data(i).action_track_index = track_index;
        file_name = sprintf('img_%06i_%06i.jpg', track_index, index);
        mpii_data(i).image_path = fullfile(DEST_PATH, 'data', 'images', file_name);
        loaded = load(fullfile(pose_dir, data_fn), 'pose');
        mpii_data(i).joint_locs = loaded.pose;
    end
    
    % Sort by frame number *within this dataset*
    [~, sorted_indices] = sort([mpii_data.frame_no]);
    mpii_data = mpii_data(sorted_indices);
    
    % Now we need to make sure we have scene numbers
    mpii_data = split_mpii_scenes(mpii_data, 0.2);
    
    save(data_path, 'mpii_data');
    regen_pairs = true;
else
    fprintf('Loading data from file\n');
    loaded = load(data_path);
    mpii_data = loaded.mpii_data;
end

% Now split it into train and tests. This is a poor method of doing that
% because some actors will be in both the train and test sets, but whatever
% (doesn't matter that much for this PoC; when this is no longer a PoC I
% should be more careful).
pair_path = fullfile(cache_dir, 'mpii_pairs.mat');
if ~exist(pair_path, 'file') || regen_pairs
    fprintf('Generating pairs\n');
    [all_pairs, pair_scenes] = find_pairs(mpii_data, FRAME_SKIP);
    
    % Choose which scenes to use for training and which to use for
    % validation
    all_scenes = unique(pair_scenes);
    all_scenes = all_scenes(randperm(length(all_scenes)));
    num_train_scenes = floor(length(all_scenes) * 0.7);
    train_scenes = all_scenes(1:num_train_scenes);
    val_scenes = all_scenes(num_train_scenes+1:end);
    disp(train_scenes);
    disp(val_scenes);
    
    % Find the pair indices corresponding to the chosen scenes
    train_indices = ismember(pair_scenes, train_scenes);
    val_indices = ismember(pair_scenes, val_scenes);
    fprintf('Have %i training pairs and %i validation pairs\n', ...
        sum(train_indices), sum(val_indices));
    
    % Now save!
    train_pairs = all_pairs(train_indices, :);
    val_pairs = all_pairs(val_indices, :);
    save(pair_path, 'train_pairs', 'val_pairs');
else
    fprintf('Loading pairs from file\n');
    loaded = load(pair_path);
    % Yes, I know I could just load() without assigning anything, but I
    % like this way better.
    train_pairs = loaded.train_pairs;
    val_pairs = loaded.val_pairs;
end
end

function [track_index, index] = parse_fn(fn)
% Parse a filename like pose_<TrackIndex>_<Index>.mat
tokens = regexp(fn, '[^\d]*(\d+)_(\d+)', 'tokens');
assert(length(tokens) >= 1);
assert(length(tokens{1}) == 2);
track_index = str2double(tokens{1}{1});
index = str2double(tokens{1}{2});
end

function [pairs, scenes] = find_pairs(mpii_data, frame_skip)
% Find pairs with frame_skip frames between them
frame_nums = [mpii_data.frame_no];
scene_nums = [mpii_data.scene_num];
fst_inds = 1:(length(mpii_data)-frame_skip-1);
snd_inds = fst_inds + frame_skip + 1;
good_inds = (frame_nums(snd_inds) - frame_nums(fst_inds) == frame_skip + 1) ...
    & (scene_nums(fst_inds) == scene_nums(snd_inds));
pairs = cat(2, fst_inds(good_inds)', snd_inds(good_inds)');
scenes = scene_nums(fst_inds(good_inds));
end