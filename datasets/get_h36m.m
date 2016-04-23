function [train_dataset, test_seqs, tsize] = get_h36m(...
    dest_dir, cache_dir, subposes, step, template_scale)
%GET_H36M Fetches H3.6M pose estimation dataset (incl. videos)
% You'll have to download all of the necessary files yourself, since the
% dataset needs authentication & EULA acceptance for download (although
% there is a H3.6M downloader script written in Python stored in this
% directory, which may make the process marginally faster once you
% register).
FAKE_SUBJECTS = 1:7;
REAL_SUBJECTS = [1 5 6 7 8 9 11];
TEST_SUBJECT = 5;
% I can't figure out how to decode these videos
VIDEO_BLACKLIST = {'S11/Videos/Directions.54138969.mp4'};
% Number of frames to jump forward between pairs
FRAME_SKIP = 10;
assert(length(FAKE_SUBJECTS) == length(REAL_SUBJECTS));

h36m_dir = fullfile(dest_dir, 'h3.6m');
assert(~~exist(fullfile(h36m_dir, 'MD5SUM'), 'file'), ...
    ['Need ' dest_dir ' to exist and have MD5 sums!']);

% Cache data, since loading process is slooooow
save_path = fullfile(cache_dir, 'h36m_data.mat');
if exist(save_path, 'file')
    fprintf('Found existing H3.6M data, so I''ll just use that\n');
    [train_dataset, test_seqs, tsize] = parload(save_path, ...
        'train_dataset', 'test_seqs', 'tsize');
    return
else
    fprintf('Need to regenerate all H3.6M data :(\n');
end

% Jump into H36M dir and extract everything
old_dir = pwd;
cd_task = onCleanup(@() cd(old_dir));
cd(h36m_dir);

% Can check MD5s if you want to verify download, but it's slow
% fprintf('Checking MD5s in %s\n', dest_dir);
% assert(system('md5sum -c MD5SUM') == 0, 'Archive MD5s must match');

empty_data = @() struct('frame_no', {}, 'frame_time', {}, ...
    'joint_locs', {}, 'subject', {}, 'action', {}, 'camera', {}, ...
    'video_id', {});
train_frames = empty_data();
test_frames = empty_data();
all_video_paths = {};
for fake_subj=FAKE_SUBJECTS
    % Extract data if necessary
    real_subj = REAL_SUBJECTS(fake_subj);
    pose_dir = sprintf('S%i/MyPoseFeatures/D2_Positions', real_subj);
    if ~exist(pose_dir, 'dir')
        fprintf('Re-extracting subject %i poses\n', real_subj);
        untar(sprintf('PosesD2_PositionsSubjectSpecific_%i.tgz', fake_subj));
    end
    video_dir = sprintf('S%i/Videos', real_subj);
    if ~exist(video_dir, 'dir')
        fprintf('Re-extracting subject %i videos\n', real_subj);
        untar(sprintf('VideosSubjectSpecific_%i.tgz', fake_subj));
    end
    
    % Iterate over each scenario
    pose_dir_listing = dir(pose_dir);
    pose_fns = {pose_dir_listing(3:end).name};
    subj_frames = empty_data();
    subject_video_paths = cell([1 length(pose_fns)]);
    parfor pose_fn_idx=1:length(pose_fns)
        pose_fn = pose_fns{pose_fn_idx};
        
        [action, cam] = parse_fn(pose_fn);
        
        vid_path = fullfile(video_dir, sprintf('%s.%i.mp4', action, cam));
        subject_video_paths{pose_fn_idx} = fullfile(dest_dir, vid_path);
        video_id = length(all_video_paths) + pose_fn_idx;
        if any(strcmp(vid_path, VIDEO_BLACKLIST))
            continue
        end
        
        [poses, frame_times] = seq_data(real_subj, action, cam);
        
        rep_ft = @(val) repmat({val}, 1, length(frame_times));
        this_data = struct(...
            'frame_no', num2cell(int32(1:length(frame_times))), ...
            'frame_time', num2cell(single(frame_times)), ...
            'joint_locs', poses, ...
            'subject', rep_ft(int8(real_subj)), ...
            'action', rep_ft(action), ...
            'camera', rep_ft(int32(cam)), ...
            'video_id', rep_ft(int32(video_id)));
        subj_frames = [subj_frames, this_data];
    end
    all_video_paths = [all_video_paths subject_video_paths]; %#ok<AGROW>
    
    if real_subj ~= TEST_SUBJECT
        train_frames = [train_frames, subj_frames]; %#ok<AGROW>
    else
        assert(isempty(test_frames));
        test_frames = subj_frames;
    end
end

assert(~isempty(test_frames) && ~isempty(train_frames));

train_pairs = find_pairs(train_frames, FRAME_SKIP);
[train_frames, train_pairs] = trim_frames(train_frames, train_pairs);
test_pairs = find_pairs(test_frames, FRAME_SKIP);
[test_frames, test_pairs] = trim_frames(test_frames, test_pairs);

train_dataset = unify_dataset(train_frames, train_pairs, 'train_dataset_h36m');
train_dataset.video_paths = all_video_paths;
test_dataset = unify_dataset(test_frames, test_pairs, 'test_dataset_h36m_s5');
test_dataset.video_paths = all_video_paths;

[train_dataset, ~] = mark_scales(train_dataset, subposes, step, ...
    template_scale);
[test_dataset, tsize] = mark_scales(test_dataset, subposes, step, ...
    template_scale, [train_dataset.pairs.scale]);

% Extract sequences from test dataset
all_test_seqs = pairs2seqs(test_dataset, 10);
fprintf('Test set has %i seqs and %i frames\n', ...
    length(all_test_seqs), sum(cellfun(@length, all_test_seqs)));
test_seqs = make_test_set(test_dataset, all_test_seqs);

cd(old_dir);
fprintf('Saving to %s\n', save_path);
save(save_path, 'train_dataset', 'test_seqs', 'tsize');
end

function [poses, frame_times] = seq_data(subject, action, cam)
video_path = sprintf(...
    fullfile('S%i', 'Videos', '%s.%i.mp4'), subject, action, cam);
pose_path = sprintf(...
    fullfile('S%i', 'MyPoseFeatures', 'D2_Positions', '%s.%i.cdf'), ...
    subject, action, cam);
poses = load_cdf_poses(pose_path);
frame_times = mp4_frametimes(video_path);
% Sometimes there are more frames than poses for some reason.
% XXX: Investigate the above. Is it just because they are aligning poses
% for the three cameras? If so, why do some cameras have more frames than
% others for the same subject and action sequence?
assert(length(frame_times) >= length(poses));
if length(frame_times) ~= length(poses)
    min_len = min(length(frame_times), length(poses));
    frame_times = frame_times(1:min_len);
    poses = poses(1:min_len);
end
end

function [action, cam] = parse_fn(fn)
% Parse filename like 'Eating 2.60457274.mp4' to get action and camera name
tokens = regexp(fn, '([\w ])+.(\d+).(mp4|cdf)', 'tokens');
assert(length(tokens) == 1 && length(tokens{1}) == 3);
action = tokens{1}{1};
cam = str2double(tokens{1}{2});
end

function poses = load_cdf_poses(cdf_path)
% Load poses from NetCDF file. Return as cell array of J*2 matrices.
var_cell = cdfread(cdf_path, 'Variable', {'Pose'});
unshaped = var_cell{1};
new_size = [size(unshaped, 1), 2, size(unshaped, 2) / 2];
shaped = permute(reshape(unshaped, new_size), [3 2 1]);
poses = squeeze(num2cell(shaped, [1 2]))';
end

function all_pairs = find_pairs(frames, frame_skip)
% Find pairs from collected frames
all_vid_ids = [frames.video_id];
vid_idents = unique(all_vid_ids);
all_pairs = [];
parfor vid_idx=1:length(vid_idents)
    vid_ident = vid_idents(vid_idx);
    frame_idxs = find(all_vid_ids == vid_ident);
    fsts = frame_idxs(1:frame_skip:end);
    snds = fsts(2:end);
    fsts = fsts(1:end-1);
    pairs = cat(2, fsts', snds');
    all_pairs = [all_pairs; pairs];
end
end

function [frames, pairs] = trim_frames(frames, pairs)
% Remove data for frames not associated with a pair
% This saves a lot of space without breaking anything, but is pretty hacky.
assert(isstruct(frames) && ismatrix(pairs) && size(pairs, 2) == 2);
included = unique(pairs(:));
% Drop frames not in a pair
frames = frames(included);
% Renumber pair references
for incl_idx=1:length(included)
    pairs(pairs == included(incl_idx)) = incl_idx;
end
end

% Some info on H3.6M:
% Data is distributed either as subject-specific archives (one archive for
% video and one archive for poses, per subject) or as activity-specific
% archives. I'm downloading the subject-specific archives.
%
% Each video archive has the name "VideosSubjectSpecific_<N>.tgz", and
% contains only the directory "S<M>/Videos/" (where M != N in some cases).
% Each video has format "<action>.<camera ID>.mp4". Some of the cameras are
% forward-facing and some are not.
%
% Each pose archive has the name "PosesD2_PositionsSubjectSpecific_3.tgz"
% and has just the directory "S<M>/MyPoseFeatures/D2_Positions/". The files
% in that directory are named "<action>.<camera ID>.cdf", and are in
% CDF 3.3 format (*not* NetCDF or HDF5, despite both of those formats being
% vastly more popular).
%
% Possible camera IDs are:
% 54138969: aft port
% 55011271: fwd port
% 58860488: rear stbd
% 60457274: fwd stbd
%
% Possible action IDs (52 of them) are:
% _ALL, _ALL 1, Directions, Directions 1, Directions 2, Discussion,
% Discussion 1, Discussion 2, Discussion 3, Eating, Eating 1, Eating 2,
% Greeting, Greeting 1, Greeting 2, Phoning, Phoning 1, Phoning 2, Phoning
% 3, Photo, Photo 1, Photo 2, Posing, Posing 1, Posing 2, Purchases,
% Purchases 1, Sitting, Sitting 1, Sitting 2, SittingDown, SittingDown 1,
% SittingDown 2, Smoking, Smoking 1, Smoking 2, TakingPhoto, TakingPhoto 1,
% Waiting, Waiting 1, Waiting 2, Waiting 3, WalkDog, WalkDog 1, Walking,
% Walking 1, Walking 2, WalkingDog, WalkingDog 1, WalkTogether,
% WalkTogether 1, WalkTogether 2
%
% Per [0], I should use subject 5 as test subject and all others as train
% subjects. Will probably use all cameras (although I have a strong
% suspicion that will cause things to asplode hard; might need to fix to
% fwd {port, stbd} if it all goes south).
%
% [0] http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Fragkiadaki_Recurrent_Network_Models_ICCV_2015_paper.pdf
%
% Joint information
% 1: Pelvis --> Dupe of 12?
% 2: Hip (right)
% 3: Knee (right)
% 4: Ankle (right)
% 5: Foot arch (right)
% 6: Toes (right)
% 7: Hip (left)
% 8: Knee (left)
% 9: Ankle (left)
% 10: Foot arch (left)
% 11: Toes (left)
% 12: Pelvis --> Dupe of 1?
% 13: Torso
% 14: Base of neck --> Dupe of 17, 25?
% 15: Head low
% 16: Head high
% 17: Base of neck --> Dupe of 14, 25?
% 18: Shoulder (left)
% 19: Elbow (left)
% 20: Wrist (left) --> Dupe of 21?
% 21: Wrist (left) --> Dupe of 20?
% 22: ?? hand (left)
% 23: ?? hand (left) --> Dupe of 24? Unreliable?
% 24: ?? hand (left) --> Dupe of 23? Unreliable?
% 25: Base of neck --> Dupe of 14, 17?
% 26: Shoulder (right)
% 27: Elbow (right)
% 28: Wrist (right) --> Dupe of 29?
% 29: Wrist (wright) --> Dupe of 29?
% 30: ?? hand (right) --> Unreliable?
% 31: ?? hand (right) --> Dupe of 32? Unreliable?
% 32: ?? hand (right) --> Dupe of 31? Unreliable?
