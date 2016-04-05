function res = shouldrebuild(src, dest)
%SHOULDREBUILD Should we rebuild src to make dest?
% Will return true iff (dest doesn't exist or src is newer than dest).
src_md = md(src);
assert(isdatetime(src_md), 'src (%s) must exist', src);
dest_md = md(dest);
res = ~isdatetime(dest_md) || src_md > dest_md;
end

function modified = md(fn)
finfo = dir(fn);
if ~isempty(finfo)
    finfo = finfo(1);
    modified = datetime(finfo.date);
else
    modified = NaN;
end
end
