% Grab deepflow and whatever else we need
function get_deps(ext_dir, cache_dir)

% First, build my little OpenCV CUDA Brox wrapper
mex -lopencv_core -lopencv_cudaoptflow _broxOpticalFlow.cpp

caffe_matlab_path = fullfile(ext_dir, 'conscaffe', 'matlab');
if ~exist(fullfile(caffe_matlab_path), 'dir')
    error(['Please download https://github.com/qxcv/conscaffe, build ' ...
           'matcaffe in that repository, then put (or link to) the entire ' ...
           'repo at ' fullfile(ext_dir, 'conscaffe')]);
end

addpath(caffe_matlab_path);
