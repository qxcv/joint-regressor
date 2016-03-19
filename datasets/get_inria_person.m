function neg_dataset = get_inria_person(dest_dir, cache_dir)
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

neg_dataset = unify_dataset(all_data, all_pairs, 'inria_neg_dataset');
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