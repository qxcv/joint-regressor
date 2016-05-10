% Load FLIC data. Also downloads FLIC data, if necessary.
% flic_data is a struct array giving information about frames in the FLIC
% data set.
% pairs is a m*2 array giving the indices of adjacent examples which are
% separated by a small number of frames.

% Guide to the FLIC skeleton:
% 1 L shoulder
% 2 L elbow
% 3 L wrist
% 4 R shoulder
% 5 R elbow
% 6 R wrist
% 7 L hip
% 8-9 N/A
% 10 R hip
% 11-12 N/A
% 13 L eye
% 14 R eye
% 15-16 N/A
% 17 Nose
% 19-29 N/A
% We probably only want to return a subset of those
%
% Note that there are two potentially handy attributes in FLIC for avoiding
% crappy poses:
%
% 1) .isbad tells you when a pose is hard to estimate (occlusion,
%    non-frontal people, several people in the same frame, etc.). It
%    doesn't take much to get flagged by .isbad, which hurts its utility in
%    our application.
% 2) .isunchecked seems to be set iff the pose wasn't checked by a
%    researcher after being labelled by a turker. Some of those are
%    probably really terrible (e.g. I've seen poses with totally
%    nonsensical labels which probably had that flag set).
%
% Since neither of those flags do exactly what I want (find fully occluded
% joints), I'll probably have to go through the pairs my code finds
% manually and correct them.

function [train_dataset, val_dataset] = get_flic(dest_dir, cache_dir, ...
    subposes, step, template_scale, trans_spec)
FLIC_URL = 'http://vision.grasp.upenn.edu/video/FLIC-full.zip';
DEST_PATH = fullfile(dest_dir, 'FLIC-full/');
CACHE_PATH = fullfile(cache_dir, 'FLIC-full.zip');
% If two samples are within 20 frames of each other, then they can be used
% for training. Some frames are too far apart to reliably compute flow, so
% we ignore them.
FRAME_THRESHOLD = 15;
% If the mean distance between joints for potentially neighbouring frames
% is greater than this amount, then we don't consider them neighbours. This
% avoids situations where one frame is included twice with different people
% labelled each time.
L2_THRESHOLD = 50;

% Just in case :)
mkdir_p(cache_dir);

if ~exist(DEST_PATH, 'dir')
    if ~exist(CACHE_PATH, 'file')
        fprintf('Downloading FLIC from %s\n', FLIC_URL);
        websave(CACHE_PATH, FLIC_URL);
    end
    fprintf('Extracting FLIC data to %s\n', DEST_PATH);
    unzip(CACHE_PATH, dest_dir);
end

flic_examples_s = load(fullfile(DEST_PATH, 'examples.mat')');
flic_examples = flic_examples_s.examples;
empty = cell([1 length(flic_examples)]);
flic_data = struct('frame_no', empty, 'movie_name', empty, ...
    'image_path', empty, 'orig_joint_locs', empty, 'joint_locs', empty, ...
    'torso_box', empty, 'is_train', empty, 'is_test', empty, ...
    'is_unchecked', empty, 'is_hard', empty);
parfor i=1:length(flic_examples)
    ex = flic_examples(i);
    flic_data(i).frame_no = ex.currframe;
    flic_data(i).movie_name = ex.moviename;
    file_name = sprintf('%s-%08i.jpg', flic_data(i).movie_name, flic_data(i).frame_no);
    flic_data(i).image_path = fullfile(DEST_PATH, 'images', file_name);
    orig_locs = convert_joints(ex.coords);
    flic_data(i).orig_joint_locs = orig_locs;
    flic_data(i).joint_locs = skeltrans(orig_locs, trans_spec);
    flic_data(i).torso_box = ex.torsobox;
    flic_data(i).is_train = ex.istrain;
    flic_data(i).is_test = ex.istest;
    flic_data(i).is_unchecked = ex.isunchecked;
    flic_data(i).is_hard = ex.isbad;
end

% Find close pairs of frames from within the FLIC+ indices. The .istrain
% and .istest attributes happen to be useless for our application, since
% none of the test pairs are adjacent, and the training pairs weren't
% explicitly chosen to be adjacent.
test_movies = {...
'bourne-supremacy', 'goldeneye', 'collateral-disc1', 'daredevil-disc1', ...
'battle-cry', 'million-dollar-baby'};
val_mask = zeros(1, length(flic_data));
for i=1:length(test_movies)
    movie = test_movies{i};
    val_mask = val_mask | strcmp({flic_data.movie_name}, movie);
end

val_inds = find(val_mask);
val_pairs = find_pairs(val_inds, flic_data, FRAME_THRESHOLD, L2_THRESHOLD);
val_dataset = unify_dataset(flic_data, val_pairs, 'val_dataset_flic');

train_inds = find(~val_mask);
train_pairs = find_pairs(train_inds, flic_data, FRAME_THRESHOLD, L2_THRESHOLD);
train_dataset = unify_dataset(flic_data, train_pairs, 'train_dataset_flic');

[train_dataset, ~] = mark_scales(train_dataset, subposes, step, ...
    template_scale);
[val_dataset, ~] = mark_scales(val_dataset, subposes, step, ...
    template_scale, [train_dataset.pairs.scale]);
end

function pairs = find_pairs(inds, flic_data, frame_skip_thresh, ...
    pose_mean_l2_thresh)
pairs = zeros([0 2]);
lookahead = 10;

for ind_idx=1:length(inds)-1
    % For each index into a frame we care about, we'll look ahead up to 10
    % indices (arbitrary) to find a frame which (a) has the same movie name
    % (b) has a frame number within thresh of the current frame number and
    % (c) has joint labels which actually make sense.
    ind = inds(ind_idx);
    eligible_inds = inds(ind_idx+1:min(ind_idx+lookahead, end));
    eligible_data = flic_data(eligible_inds);
    names_valid = strcmp(flic_data(ind).movie_name, ...
        {eligible_data.movie_name});
    frame_diffs = [eligible_data.frame_no] ...
        - flic_data(ind).frame_no;
    frames_valid = frame_diffs > 0 & frame_diffs <= frame_skip_thresh;
    these_joints = flic_data(ind).joint_locs;
    all_mean_l2s = cellfun(@(p) mean_dists(these_joints, p), ...
        {eligible_data.joint_locs});
    dists_valid = all_mean_l2s < pose_mean_l2_thresh;
    
    okay_idxs = eligible_inds(names_valid & frames_valid & dists_valid);
    
    if ~isempty(okay_idxs)
        pairs = [pairs; ind okay_idxs(1)]; %#ok<AGROW>
    end
end
end

function locs = convert_joints(orig_locs)
locs = orig_locs';
locs(isnan(locs)) = 0;  
end
