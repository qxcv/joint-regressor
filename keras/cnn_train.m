function cnn_train(cnnpar, cache_dir, train_h5s, val_h5s)
%CNN_TRAIN Train a CNN using Keras

assert(~isempty(val_h5s) && ~isempty(val_h5s), ...
    'Need some training and validation patches');

this_dir = fileparts(mfilename('fullpath'));

if exist(cnnpar.deploy_json, 'file') && exist(cnnpar.deploy_weights, 'file')
    fprintf('Already have weights, no need to re-train CNN\n');
    return
end

fprintf('Getting inital weights for fine-tuning\n');
ilsvrc_weight_path = fetch_vgg16_weights;

fprintf('Configuring virtualenv\n');
configure_env(cnnpar);

init_weight_path = fullfile(cache_dir, 'ilsvrc_ft_weights.h5');
if ~exist(init_weight_path, 'file')
    fprintf('Resizing initial weights to fit our architecture\n');
    init_script_path =  fullfile(this_dir, 'make_init_weights.py');
    model_name = 'vggnet16_poselet_class_flow';
    conv_cmdline = to_cmdline(init_script_path, ...
        ilsvrc_weight_path, model_name, train_h5s{1}, init_weight_path);
    display(['Running "' conv_cmdline '"']);
    assert(system(conv_cmdline, '-echo') == 0, 'Net conversion failed');
else
    fprintf('Resized weights exist already, will skip to training\n');
end

train_files = strjoin(train_h5s, ',');
val_files = strjoin(val_h5s, ',');
checkpoint_dir = fullfile(cache_dir, 'keras-checkpoints');
train_script_path =  fullfile(this_dir, 'train.py');
mp_path = fullfile(cache_dir, 'mean_pixel.mat');

train_cmdline = to_cmdline(train_script_path, ...
    '--model-name', 'vggnet16_poselet_class_flow', ...
    '--mean-pixel-mat', mp_path, ...
    '--learning-rate', '0.0001', ...
    '--decay', '0.00001', ...
    '--finetune', init_weight_path, ...
    '--write-fc-weights', cnnpar.deploy_weights, ...
    '--write-fc-json', cnnpar.deploy_json, ...
    ... TODO: Need better stopping rule than this :/
    '--max-iter', '50000', ...
    train_files, val_files, checkpoint_dir);
display(['Running "' train_cmdline '"']);
assert(system(train_cmdline, '-echo') == 0, 'CNN training failed');
end

function cmdline = to_cmdline(varargin)
% Converts args to something like '"<arg1>" "<arg2>" ...'. Unfortunately
% Matlab doesn't have a sane system() (one which lets you explicitly
% separate arguments), and this is the closest I'm willing to get to full
% shell escaping.
mapped = cellfun(@(p) ['"' p '"'], varargin, 'UniformOutput', false);
cmdline = strjoin(mapped, ' ');
end
