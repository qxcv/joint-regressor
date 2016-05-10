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

function [train_dataset, val_dataset] = get_flic(dest_dir, cache_dir, ...
    subposes, step, template_scale, trans_spec)
FLIC_URL = 'http://vision.grasp.upenn.edu/video/FLIC-full.zip';
DEST_PATH = fullfile(dest_dir, 'FLIC-full/');
CACHE_PATH = fullfile(cache_dir, 'FLIC-full.zip');
% If two samples are within 20 frames of each other, then they can be used
% for training. Some frames are too far apart to reliably compute flow, so
% we ignore them.
FRAME_THRESHOLD = 20;

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
    'torso_box', empty, 'is_train', empty, 'is_test', empty);
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
val_pairs = find_pairs(val_inds, flic_data, FRAME_THRESHOLD);
val_dataset = unify_dataset(flic_data, val_pairs, 'val_dataset_flic');

train_inds = find(~val_mask);
train_pairs = find_pairs(train_inds, flic_data, FRAME_THRESHOLD);
train_dataset = unify_dataset(flic_data, train_pairs, 'train_dataset_flic');

[train_dataset, ~] = mark_scales(train_dataset, subposes, step, ...
    template_scale);
[val_dataset, ~] = mark_scales(val_dataset, subposes, step, ...
    template_scale, [train_dataset.pairs.scale]);
end

function pairs = find_pairs(inds, flic_data, thresh)
fst_inds = inds(1:end-1);
snd_inds = inds(2:end);
names_eq = strcmp({flic_data(fst_inds).movie_name}, {flic_data(snd_inds).movie_name});
fdiffs = [flic_data(snd_inds).frame_no] - [flic_data(fst_inds).frame_no];
firsts = find(names_eq & fdiffs > 0 & fdiffs <= thresh);
pairs = cat(2, firsts', firsts'+1);
end

function locs = convert_joints(orig_locs)
locs = orig_locs';
locs(isnan(locs)) = 0;  
end
