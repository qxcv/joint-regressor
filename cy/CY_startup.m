function CY_startup
if ~exist('./bin', 'dir')
  mkdir('./bin');
end

if ~isdeployed
  addpaths({'dataio', 'bin', 'src', 'tools', 'external', 'external/qpsolver'});
end
end

function addpaths(paths)
for i=1:length(paths)
    path = paths{i};
    if ~isempty(path) && path(1) ~= '/'
        addpath(fullfile(pwd, path));
    else
        addpath(path);
    end
end
end