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

function [flic_data, pairs] = get_flic(dest_dir, cache_dir)
FLIC_URL = 'http://vision.grasp.upenn.edu/video/FLIC.zip';
DEST_PATH = fullfile(dest_dir, 'FLIC/');
CACHE_PATH = fullfile(cache_dir, 'FLIC.zip');
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
end;

flic_examples_s = load(fullfile(DEST_PATH, 'examples.mat')');
flic_examples = flic_examples_s.examples;
flic_data = struct(); % Silences Matlab warnings about growing arrays
for i=1:length(flic_examples)
    ex = flic_examples(i);
    flic_data(i).frame_no = ex.currframe;
    flic_data(i).movie_name = ex.moviename;
    file_name = sprintf('%s-%08i.jpg', flic_data(i).movie_name, flic_data(i).frame_no);
    flic_data(i).image_path = fullfile(DEST_PATH, 'images', file_name);
    flic_data(i).joint_locs = convert_joints(ex.coords);
    flic_data(i).torso_box = ex.torsobox;
    flic_data(i).is_train = ex.istrain;
    flic_data(i).is_test = ex.istest;
end

% Find close pairs of frames
inds = 1:(length(flic_examples)-1);
names_eq = strcmp({flic_data(inds).movie_name}, {flic_data(inds+1).movie_name});
fdiffs = [flic_data(inds+1).frame_no] - [flic_data(inds).frame_no];
firsts = find(names_eq & fdiffs > 0 & fdiffs <= FRAME_THRESHOLD);
pairs = cat(2, firsts', firsts'+1);

function locs = convert_joints(orig_locs)
% Return head, followed by L shoulder/elbow/wrist, followed by R
% shoulder/elbow/wrist
locs = orig_locs(:, [17 1:6])';