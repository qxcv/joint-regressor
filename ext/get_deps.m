% Grab deepflow and whatever else we need
function get_deps(ext_dir, cache_dir)
% I'm using LDOF instead of the (newer, similar, probably better) DeepFlow
% because LDOF does not use ~60GB memory on the MPII continuous pose videos
% (from the cooking dataset)! It also does not segfault MATLAB on the
% regular.
LDOF_URL = 'http://www.cs.berkeley.edu/~katef/code/LDOF_src.zip';
ldof_dir = fullfile(ext_dir, 'LDOF_src/');
ldof_touch_file = fullfile(ldof_dir, '.built');

%% Get and build DeepMatching (required by DeepFlow)
if ~exist(ldof_touch_file, 'file')
    % First, make sure we have a copy of DeepMatching
    if ~exist(ldof_dir, 'dir')
        zip_path = fullfile(cache_dir, 'LDOF_src.zip');
        if ~exist(zip_path, 'file')
            fprintf('Downloading LDOF from %s\n', LDOF_URL);
            websave(zip_path, LDOF_URL);
        end
        fprintf('Extracting LDOF to %s\n', ldof_dir);
        unzip(zip_path, ext_dir);
    end
    
    % Now we can build LDOF
    last_folder = cd(ldof_dir);
    movefile('startup.m', 'startup_ldof.m');
    % For some reason the LDOF distribution is filled with shit, like the
    % .m~/.cpp~ backup files.
    delete('*.m~', '*.cpp~');
    startup_ldof;
    cd(last_folder);
    
    % Touch a file so that we don't have to do that again
    fclose(fopen(ldof_touch_file, 'w'));
end

addpath(ldof_dir);

caffe_matlab_path = fullfile(ext_dir, 'conscaffe', 'matlab');
if ~exist(fullfile(caffe_matlab_path), 'dir')
    error(['Please download https://github.com/qxcv/conscaffe, build ' ...
           'matcaffe in that repository, then put (or link to) the entire ' ...
           'repo at ' fullfile(ext_dir, 'conscaffe')]);
end

addpath(caffe_matlab_path);
