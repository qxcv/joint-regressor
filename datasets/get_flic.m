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

save_path = fullfile(cache_dir, 'flic_data.mat');
if exist(save_path, 'file')
    fprintf('Loading data from %s\n', save_path);
    [train_dataset, val_dataset] = parload(save_path, 'train_dataset', ...
        'val_dataset');
    return
else
    fprintf('%s does not exist; have to generate data again\n', save_path);
end

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

% pathological pairs are ones with significant occlusion, people facing in
% the wrong direction or with incomprehensible poses, several people who
% overlap, no people at all (really; one of the FLIC-full samples is an
% animated title card that doesn't even depict a person!)
patho_pairs = parload(fullfile(dest_dir, 'flic-bad-pairs.mat'), 'flic_bad_pairs');

% Could also use this to get rid of some extra "hard" pairs, but it wastes
% to much data for me to have the heart to turn it on :P
% good_mask = ~[flic_data.is_hard];

val_inds = find(val_mask);
val_pairs = find_pairs(val_inds, flic_data, patho_pairs, FRAME_THRESHOLD, L2_THRESHOLD);
val_dataset = unify_dataset(flic_data, val_pairs, 'val_dataset_flic');

train_inds = find(~val_mask);
train_pairs = find_pairs(train_inds, flic_data, patho_pairs, FRAME_THRESHOLD, L2_THRESHOLD);
train_dataset = unify_dataset(flic_data, train_pairs, 'train_dataset_flic');

[train_dataset, ~] = mark_scales(train_dataset, subposes, step, ...
    template_scale);
[val_dataset, ~] = mark_scales(val_dataset, subposes, step, ...
    template_scale, [train_dataset.pairs.scale]);

save(save_path, 'train_dataset', 'val_dataset');
end

function pairs = find_pairs(inds, flic_data, patho_pairs, frame_skip_thresh, ...
    pose_mean_l2_thresh)
pairs = zeros([0 2]);
lookahead = 10;

% Because of the way I store pathological pairs (as pairs of indices into
% the original FLIC-full examples.mat), it's important that the FLIC data
% loading proces doesn't change. This is a weak check of that. A stronger
% check would be to check the hashes of example.mat:
%
% md5sum: 81caf04c08d1a84f8d3a23a23fdf86d1  examples.mat
% sha1sum: 5f44c0503e54201b091f097657af74f56fa264df  examples.mat
%
% Also, wc -c examples.mat gives "4270702 examples.mat" (so 4.2MB of data),
% and stat reports that the modify time is 2012-09-05 10:25:08 (presumably
% extracted from the downloaded archive, since it's 2016 now!).
assert(length(flic_data) == 20928);

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

pairs = remove_patho_pairs(pairs, patho_pairs);
assert(size(pairs, 2) == 2 && ismatrix(pairs));
end

function locs = convert_joints(orig_locs)
locs = orig_locs';
locs(isnan(locs)) = 0;  
end

function new_pairs = remove_patho_pairs(old_pairs, patho_pairs)
% Take a 2D matrix of pathological pairs and a matrix of tentative pairs
% (say validation or training pairs) in `old_pairs` and return the tenative
% pairs without the pathological ones.
assert(ismatrix(old_pairs) && ismatrix(patho_pairs) ...
    && size(old_pairs, 2) == 2 && size(patho_pairs, 2) == 2);
new_pairs = setdiff(old_pairs, patho_pairs, 'rows');
fprintf('Trimmed pairs, %i->%i using %i pathological pairs\n', ...
    size(old_pairs, 1), size(new_pairs, 1), size(patho_pairs, 1));
end
