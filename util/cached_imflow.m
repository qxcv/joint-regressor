function out_flow = cached_imflow(d1, d2, cache_dir)
%Compute optical flow between two images from the data set; cache if
%possible.

flow_dir = fullfile(cache_dir, 'flow');
if ~exist(flow_dir, 'dir')
    mkdir(flow_dir);
end

cache_name = sprintf('%s_to_%s.mat', dname(d1), dname(d2));
path = fullfile(flow_dir, cache_name);

try
    load(path, 'out_flow');
catch
    fprintf('Computing flow for %s->%s\n', dname(d1), dname(d2));
    out_flow = imflow(d1.image_path, d2.image_path);
    save(path, 'out_flow');
end