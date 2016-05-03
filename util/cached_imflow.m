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
    frame_1 = readim(d1);
    frame_2 = readim(d2);
    out_flow = imflow(frame_1, frame_2);
    save(path, 'out_flow');
end
