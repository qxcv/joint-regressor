function new_flow = smart_resize_flow(flow_data, new_size_in)
%SMART_RESIZE_FLOW Rescale flow data properly.
% This carefully handles the rescaling of flow vectors as will as
% upsampling of the flow data itself.
% Input data should be in [h w c] format, just like imread produces.
assert(ndims(flow_data) == 3);
assert(size(flow_data, 3) == 2);
assert(isvector(new_size_in));
assert(length(new_size_in) <= 3);

old_size = size(flow_data);
old_size = old_size(1:2);

if isscalar(new_size_in)
    new_size = new_size_in * old_size;
elseif ndims(new_size_in) == 3
    new_size = new_size_in(1:2);
end

new_flow = imresize(flow_data, new_size); 
new_flow = rescale_flow_mags(new_flow, old_size, new_size);
assert(all(size(new_flow) == [new_size 2]));
end