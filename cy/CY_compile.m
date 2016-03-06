function CY_compile()
buildfile('src/mex/distance_transform.cpp');
buildfile('external/qpsolver/qp_one_sparse.cc');
buildfile('external/qpsolver/score.cc');
buildfile('external/qpsolver/lincomb.cc');
end

function buildfile(path)
mexcmd = 'mex -outdir bin';
mexcmd = [mexcmd ' -O'];
mexcmd = [mexcmd ' -L/usr/lib -L/usr/local/lib'];

[~, fn, ~] = fileparts(path);
dest_path = fullfile('bin', [fn '.' mexext]);
if ~exist(dest_path, 'file')
    eval([mexcmd ' ' path]);
end
end