function [all_data, all_pairs] = get_inria_person(dest_dir, cache_dir)
%GET_INRIA_PERSON Grab the INRIA negatives dataset
INRIA_PERSON_URL = 'http://pascal.inrialpes.fr/data/human/INRIAPerson.tar';
INRIA_DEST_PATH = fullfile(dest_dir, 'inria-person');
INRIA_CACHE_PATH = fullfile(cache_dir, 'INRIAPerson.tar');

% Download data
if ~exist(INRIA_DEST_PATH, 'dir')
    if ~exist(INRIA_CACHE_PATH, 'file')
        fprintf('Downloading INRIA Person dataset from %s\n', INRIA_PERSON_URL);
        websave(INRIA_CACHE_PATH, INRIA_PERSON_URL);
    end
    fprintf('Extracting basic INRIA Person data to %s\n', INRIA_DEST_PATH);
    untar(INRIA_CACHE_PATH, INRIA_DEST_PATH);
end

% Grab image names
real_dest = fullfile(INRIA_DEST_PATH, 'INRIAPerson');
val_files = read_file_list(fullfile(real_dest, 'Test', 'neg.lst'));
train_files = read_file_list(fullfile(real_dest, 'Train', 'neg.lst'));
assert(length(train_files) > 1 && length(val_files) > 1);
all_neg_paths = fullfile(real_dest, [train_files val_files]);
assert(iscell(all_neg_paths));
assert(length(all_neg_paths) == length(train_files) + length(val_files));

% Return data struct and pairs
% MPII data struct has the following names:
% frame_no, action_track_index, image_path, joint_locs, is_val, scene_num
% Really I only need image_path; everything else can go jump.
all_data = struct('image_path', all_neg_paths);
all_pairs = [1:length(all_data); 1:length(all_data)]';
end

function filenames = read_file_list(path)
fp = fopen(path, 'r');
filenames = {};
while true
    next_line = fgetl(fp);
    if ~ischar(next_line)
        break;
    end
    filenames{length(filenames)+1} = next_line; %#ok<AGROW>
end
fclose(fp);
end

% data_path = fullfile(cache_dir, 'inria_data.mat');
% regen_pairs = false;
% if ~exist(data_path, 'file')
%     fprintf('Regenerating data\n');
%     val_data = load_files_basic(INRIA_DEST_PATH);
%     
%     train_data = split_mpii_scenes(train_data, 0.2);
%     val_data = split_mpii_scenes(val_data, 0.1);
%     
%     save(data_path, 'train_data', 'val_data');
%     regen_pairs = true;
% else
%     fprintf('Loading data from file\n');
%     loaded = load(data_path);
%     train_data = loaded.train_data;
%     val_data = loaded.val_data;
% end
% 
% % Now split into training and validation sets TODO
% pair_path = fullfile(cache_dir, 'mpii_pairs.mat');
% if ~exist(pair_path, 'file') || regen_pairs
%     fprintf('Generating pairs\n');
%     train_pairs = find_pairs(train_data, TRAIN_FRAME_SKIP);
%     val_pairs = find_pairs(val_data, VAL_FRAME_SKIP);
%     save(pair_path, 'train_pairs', 'val_pairs');
% else
%     fprintf('Loading pairs from file\n');
%     loaded = load(pair_path);
%     % Yes, I know I could just load() without assigning anything, but I
%     % like this way better.
%     train_pairs = loaded.train_pairs;
%     val_pairs = loaded.val_pairs;
% end
% end
% 
% % XXX: This is ugly. I can probably combine load_files_{continuous,basic}.
% function cont_data = load_files_continuous(dest_path)
% pose_dir = fullfile(dest_path, 'data', 'gt_poses');
% pose_fns = dir(pose_dir);
% pose_fns = pose_fns(3:end); % Remove . and ..
% cont_data = struct(); % Silences Matlab warnings about growing arrays
% for i=1:length(pose_fns)
%     data_fn = pose_fns(i).name;
%     [track_index, index] = parse_continuous_fn(data_fn);
%     % "track_index" is the index within the dataset for the action
%     % track. "index" is the index within the dataset. They really could
%     % have picked a more optimal name :/
%     cont_data(i).frame_no = index;
%     cont_data(i).action_track_index = track_index;
%     file_name = sprintf('img_%06i_%06i.jpg', track_index, index);
%     cont_data(i).image_path = fullfile(dest_path, 'data', 'images', file_name);
%     loaded = load(fullfile(pose_dir, data_fn), 'pose');
%     cont_data(i).joint_locs = loaded.pose;
%     cont_data(i).is_val = false;
% end
% cont_data = sort_by_frame(cont_data);
% end
% 
% function basic_data = load_files_basic(dest_path)
% pose_dir = fullfile(dest_path, 'data', 'train_data', 'gt_poses');
% pose_fns = dir(pose_dir);
% pose_fns = pose_fns(3:end);
% basic_data = struct(); % Silences Matlab warnings about growing arrays
% for i=1:length(pose_fns)
%     data_fn = pose_fns(i).name;
%     frame_no = parse_basic_fn(data_fn);
%     basic_data(i).frame_no = frame_no;
%     file_name = sprintf('img_%06i.jpg', frame_no);
%     basic_data(i).image_path = fullfile(dest_path, 'data', 'train_data', 'images', file_name);
%     loaded = load(fullfile(pose_dir, data_fn), 'pose');
%     basic_data(i).joint_locs = loaded.pose;
%     basic_data(i).is_val = true;
% end
% basic_data = sort_by_frame(basic_data);
% end
% 
% function pairs = find_pairs(data, frame_skip)
% % Find pairs with frame_skip frames between them
% frame_nums = [data.frame_no];
% scene_nums = [data.scene_num];
% fst_inds = 1:(length(data)-frame_skip-1);
% snd_inds = fst_inds + frame_skip + 1;
% good_inds = (frame_nums(snd_inds) - frame_nums(fst_inds) == frame_skip + 1) ...
%     & (scene_nums(fst_inds) == scene_nums(snd_inds));
% pairs = cat(2, fst_inds(good_inds)', snd_inds(good_inds)');
% end
