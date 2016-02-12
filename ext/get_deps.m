% Grab deepflow and whatever else we need
function get_deps(ext_dir, cache_dir)

% First, build my little OpenCV CUDA Brox wrapper
this_file = mfilename('fullpath');
[this_dir, ~, ~] = fileparts(this_file);
flow_dir = fullfile(this_dir, 'flow');
old_dir = pwd;
cd(flow_dir);
if ~exist('.built', 'file');
    mex('mex_broxOpticalFlow.cpp', '-lopencv_core', '-lopencv_cudaoptflow');
    fclose(fopen('.built', 'w'));
end
addpath(flow_dir);
cd(old_dir);

% Now build the semaphore thing
sem_dir = fullfile(this_dir, 'semaphore');
cd(sem_dir);
if ~exist('.built', 'file');
    mex('semaphore.c');
    fclose(fopen('.built', 'w'));
end
addpath(sem_dir);
cd(old_dir);

caffe_matlab_path = fullfile(ext_dir, 'conscaffe', 'matlab');
if ~exist(fullfile(caffe_matlab_path), 'dir')
    error(['Please download https://github.com/qxcv/conscaffe, build ' ...
           'matcaffe in that repository, then put (or link to) the entire ' ...
           'repo at ' fullfile(ext_dir, 'conscaffe')]);
end

addpath(caffe_matlab_path);
end