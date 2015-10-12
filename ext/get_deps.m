% Grab deepflow and whatever else we need
function get_deps(ext_dir, cache_dir)
DEEPMATCHING_URL = 'http://lear.inrialpes.fr/src/deepmatching/code/deepmatching_1.2.2.zip';
DEEPFLOW_URL = 'http://pascal.inrialpes.fr/data2/deepmatching/files/DeepFlow_release2.0.tar.gz';

deepflow_dir = fullfile(ext_dir, 'DeepFlow_release2.0/');
deepflow_touch_file = fullfile(deepflow_dir, '.built');
deepmatching_dir = fullfile(ext_dir, 'deepmatching_1.2.2_c++/');
deepmatching_touch_file = fullfile(deepmatching_dir, '.built');
make_flags = sprintf(' -j%i', feature('numCores'));

%% Get and build DeepMatching (required by DeepFlow)
if ~exist(deepmatching_touch_file, 'file')
    % First, make sure we have a copy of DeepMatching
    if ~exist(deepmatching_dir, 'dir')
        zip_path = fullfile(cache_dir, 'deepmatching.zip');
        if ~exist(zip_path, 'file')
            fprintf('Downloading DeepMatching from %s\n', DEEPMATCHING_URL);
            websave(zip_path, DEEPMATCHING_URL);
        end
        fprintf('Extracting DeepMatching to %s\n', deepmatching_dir);
        unzip(zip_path, ext_dir);
    end
    
    % Now we can build DeepMatching
    last_folder = cd(deepmatching_dir);
    % Don't bother with the makefile
    % NB: This REALLY requries
    % LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_rt.so before starting
    % Matlab. Otherwise, Matlab segfaults when producing flow and doesn't
    % bother writing anything to the crash dump. This makes no sense, and I
    % don't have time to fix it :P
    mex deepmatching_matlab.cpp deep_matching.cpp conv.cpp hog.cpp ...
        image.cpp io.cpp main.cpp maxfilter.cpp pixel_desc.cpp ...
        -output deepmatching CXXFLAGS="-fPIC -Wall -g -std=c++11 -O3" ...
        -lpng -ljpeg -lm -lmwblas
    cd(last_folder);
    
    % Touch a file so that we don't have to do that again
    fclose(fopen(deepmatching_touch_file, 'w'));
end

addpath(deepmatching_dir);

%% Get and built DeepFlow
if ~exist(deepflow_touch_file, 'file')
    % Download it
    if ~exist(deepflow_dir, 'dir')
        tar_path = fullfile(cache_dir, 'deepflow.tar');
        zip_path = fullfile(cache_dir, 'deepflow.tar.gz');
        if ~exist(zip_path, 'file')
            fprintf('Downloading DeepFlow from %s\n', DEEPFLOW_URL);
            websave(zip_path, DEEPFLOW_URL);
        end
        fprintf('Extracting DeepFlow to %s\n', deepflow_dir);
        gunzip(zip_path);
        untar(tar_path, ext_dir);
    end
    
    % Now let's build it
    last_folder = cd(deepflow_dir);
    % Add $(LIBFLAGS) to the end, since otherwise GCC freaks out because it
    % throws away the first -l flags when it finds it can't use them
    % immediately (but later realises that "Oh, hey, looks like I need some
    % weird symbols that I don't know how to get. Wacky, right?")
    !sed -i.bak -e 's/\(.\+$(LIBFLAGS).\+\).\+$/\1 $(LIBFLAGS)/' Makefile
    status = system(['make' make_flags]);
    if status ~= 0
        error('Could not build DeepFlow');
    end
    mex deepflow2_matlab.cpp image.o io.o opticalflow.o opticalflow_aux.o solver.o -ljpeg -lpng -lm -output deepflow2
    cd(last_folder);
    fclose(fopen(deepflow_touch_file, 'w'));
end

addpath(deepflow_dir);
