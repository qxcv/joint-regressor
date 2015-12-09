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
% 10 R hip
% 13 L eye
% 14 R eye
% 17 Nose
% We probably only want to return a subset of those

function [flic_data, train_pairs, test_pairs] = get_flic(dest_dir, cache_dir, poselet)
FLIC_URL = 'http://vision.grasp.upenn.edu/video/FLIC-full.zip';
DEST_PATH = fullfile(dest_dir, 'FLIC-full/');
CACHE_PATH = fullfile(cache_dir, 'FLIC-full.zip');
FLIC_PLUS_URL = 'http://cims.nyu.edu/~tompson/data/tr_plus_indices.mat';
FLIC_PLUS_DEST = fullfile(dest_dir, 'tr_plus_indices.mat');
% If two samples are within 20 frames of each other, then they can be used
% for training. Some frames are too far apart to reliably compute flow, so
% we ignore them.
FRAME_THRESHOLD = 20;

if ~exist(DEST_PATH, 'dir')
    if ~exist(CACHE_PATH, 'file')
        fprintf('Downloading FLIC from %s\n', FLIC_URL);
        websave(CACHE_PATH, FLIC_URL);
    end
    fprintf('Extracting FLIC data to %s\n', DEST_PATH);
    unzip(CACHE_PATH, dest_dir);
end

if ~exist(FLIC_PLUS_DEST, 'file')
    fprintf('Downloading FLIC+ annotations from %s\n', FLIC_PLUS_URL);
    websave(FLIC_PLUS_DEST, FLIC_PLUS_URL);
end

fp_loaded = load(FLIC_PLUS_DEST);
flic_plus_indices = sort(fp_loaded.tr_plus_indices);

flic_examples_s = load(fullfile(DEST_PATH, 'examples.mat')');
flic_examples = flic_examples_s.examples;
flic_data = struct(); % Silences Matlab warnings about growing arrays
for i=1:length(flic_examples)
    ex = flic_examples(i);
    flic_data(i).frame_no = ex.currframe;
    flic_data(i).movie_name = ex.moviename;
    file_name = sprintf('%s-%08i.jpg', flic_data(i).movie_name, flic_data(i).frame_no);
    flic_data(i).image_path = fullfile(DEST_PATH, 'images', file_name);
    flic_data(i).joint_locs = convert_joints(ex.coords, poselet);
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
train_inds = intersect(flic_plus_indices, find(~[flic_examples.istest]));
test_inds = intersect(flic_plus_indices, find([flic_examples.istest]));
train_pairs = find_pairs(train_inds, flic_data, FRAME_THRESHOLD);
test_pairs = find_pairs(test_inds, flic_data, FRAME_THRESHOLD);
end

function pairs = find_pairs(inds, flic_data, thresh)
fst_inds = inds(1:end-1);
snd_inds = inds(2:end);
names_eq = strcmp({flic_data(fst_inds).movie_name}, {flic_data(snd_inds).movie_name});
fdiffs = [flic_data(snd_inds).frame_no] - [flic_data(fst_inds).frame_no];
firsts = find(names_eq & fdiffs > 0 & fdiffs <= thresh);
pairs = cat(2, firsts', firsts'+1);
end

function locs = convert_joints(orig_locs, poselet)
% Return head, followed by L shoulder/elbow/wrist, followed by R
% shoulder/elbow/wrist
locs = orig_locs(:, poselet)';
end