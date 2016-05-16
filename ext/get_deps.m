% Grab deepflow and whatever else we need
function get_deps
% Build my little OpenCV CUDA Brox wrapper
this_file = mfilename('fullpath');
[this_dir, ~, ~] = fileparts(this_file);
flow_dir = fullfile(this_dir, 'flow');
old_dir = pwd;
cd(flow_dir);
to_build = struct(...
    'source', {'mex_broxOpticalFlow.cpp',       'mex_cvGPUDevice.cpp'}, ...
    'dest',   {['mex_broxOpticalFlow.' mexext], ['mex_cvGPUDevice.' mexext]});
for fn=1:length(to_build)
    src = to_build(fn).source;
    dest_mex = to_build(fn).dest;
    if shouldrebuild(src, dest_mex);
        mex_cmd = ['mex CXXFLAGS=''$CXXFLAGS -std=gnu++11'' -lopencv_core -lopencv_cudaoptflow -output ' dest_mex ' ' src];
        eval(mex_cmd);
    end
end
addpath_full(flow_dir);
cd(old_dir);

% Add some other junk from the file exchange
misc_dir = fullfile(this_dir, 'misc');
addpath_full(misc_dir);
end
