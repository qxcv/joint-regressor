% Grab deepflow and whatever else we need
function get_deps
% Build my little OpenCV CUDA Brox wrapper
this_file = mfilename('fullpath');
[this_dir, ~, ~] = fileparts(this_file);
flow_dir = fullfile(this_dir, 'flow');
old_dir = pwd;
cd(flow_dir);
if ~exist('.built', 'file');
    mex('mex_broxOpticalFlow.cpp', '-lopencv_core', '-lopencv_cudaoptflow');
    fclose(fopen('.built', 'w'));
end
addpath_full(flow_dir);
cd(old_dir);

% Add some other junk from the file exchange
misc_dir = fullfile(this_dir, 'misc');
addpath_full(misc_dir);
end