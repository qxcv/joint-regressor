% addpaths!
addpath_full ./mex/;
addpath_full ./eval/;  
addpath_full ./detect/;
addpath_full ./;

mex_fns = {'unsafe_ksp'};
for i=1:length(mex_fns)
    fn = mex_fns{i};
    src_path = fullfile('./mex', [fn '.cpp']);
    dest_path = fullfile('./mex', [fn '.' mexext]);
    if shouldrebuild(src_path, dest_path)
        mex(src_path, '-output', dest_path);
    end
end
