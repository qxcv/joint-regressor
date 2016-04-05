function CY_compile()
buildfile('src/mex/shiftdt.cpp');
buildfile('external/qpsolver/qp_one_sparse.cpp');
buildfile('external/qpsolver/score.cpp');
buildfile('external/qpsolver/lincomb.cpp');
end

function buildfile(src_path)
mexcmd = 'mex -outdir bin';
mexcmd = [mexcmd ' -O'];
mexcmd = [mexcmd ' -L/usr/lib -L/usr/local/lib'];

[~, fn, ~] = fileparts(src_path);
dest_path = fullfile('bin', [fn '.' mexext]);
if shouldrebuild(src_path, dest_path)
    eval([mexcmd ' ' src_path]);
end
end