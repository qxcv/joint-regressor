function scaled_flow = rescale_flow_mags(flow, old_size, new_size)
%RESCALE_FLOW_MAGS Change flow magnitude to match change in patch size.
% If you perform a rescaling on a flow patch then you should remember to
% call this afterwards to ensure that flow is scaled up or down to the
% correct dimensions. Note that this WON'T change the shape of the flow
% matrix itself.

% Safety checks on dimensions
assert(ndims(flow) == 3, 'Need h*w*c flow');
assert(size(flow, 3) == 2, 'Need u,v channels');

scale_factors = new_size ./ old_size;
scale_factors = scale_factors(2:-1:1);
scaled_flow = bsxfun(@times, flow, reshape(scale_factors, [1 1 2]));
end