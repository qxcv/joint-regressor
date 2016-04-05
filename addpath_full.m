function addpath_full(varargin)
%ADDPATH_FULL Run addpath on all arguments, but resolve full path first
% Will raise error if a path doesn't exist.
here = pwd;
assert(isunix, 'This won''t work on Windows');
for i=1:nargin
    fn = varargin{i};
    assert(~isempty(fn), 'Filename can''t be empty');
    if fn(1) ~= '/'
        full_path = fullfile(here, fn);
    else
        full_path = fn;
    end
    assert(exist(full_path, 'dir') || exist(full_path, 'file'), ...
        '"%s" must exist', full_path);
    addpath(full_path);
end
end

