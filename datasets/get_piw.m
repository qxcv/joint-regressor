function test_seqs = get_piw(ds_dir, cache_dir);
%GET_PIW Get Poses in the Wild dataset
% See
% https://github.com/qxcv/comp2560/blob/master/project/matlab/get_piw_data.m
% for example of sort of transformations which need to be done to PIW data.
piw_url = 'https://lear.inrialpes.fr/research/posesinthewild/dataset/poses_in_the_wild_public.tar.gz';
cache_path = fullfile(cache_dir, 'poses_in_the_wild_public.tar.gz');
dest_path = fullfile(ds_dir, 'poses_in_the_wild_public');

mkdir_p(cache_dir);

if ~exist(dest_path, 'dir')
    if ~exist(cache_path, 'file')
        fprintf('Downloading PIW from %s\n', piw_url);
        websave(cache_path, piw_url);
    end
    fprintf('Extracting PIW data to %s\n', dest_path);
    % Extract  to ds_dir and it will show up at dest_path
    untar(cache_path, ds_dir);
end

% Seqs are already split for us, so this will be really easy.
piw_data = parload(fullfile(dest_path, 'poses_in_the_wild_data.mat'), ...
    'piw_data');
num_data = length(piw_data);
make_empty = @() cell([1 num_data]);
data = struct('image_path', make_empty(), 'joint_locs', make_empty(), ...
    'visible', make_empty, 'seq', make_empty());

for data_idx=1:num_data
    orig_datum = piw_data(data_idx);
    datum.image_path = fullfile(dest_path, orig_datum.im);
    datum.joint_locs = orig_datum.point;
    datum.visible = orig_datum.visible;
    datum.seq = get_seq_num(datum.image_path);
    data(data_idx) = datum;
end

all_seq_nums = [data.seq];
seq_nums = unique(all_seq_nums);
seqs = cell([1 length(seq_nums)]);
for seq_idx=1:length(seq_nums)
    seq_num = seq_nums(seq_idx);
    seqs{seq_idx} = find(all_seq_nums == seq_num);
end

test_seqs.data = data;
test_seqs.seqs = seqs;
test_seqs.name = 'piw_test_seqs';
end

function seq_num = get_seq_num(filename)
[~, tokens, ~] = regexp(filename, '/selected_seqs/seq(\d+)/', ...
    'match', 'tokens');
assert(length(tokens) == 1 && length(tokens{1}) == 1);
num_str = tokens{1}{1};
seq_num = str2double(num_str);
end
