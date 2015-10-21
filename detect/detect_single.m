function acts = detect_single(stack, net)
%DETECT_SINGLE Output a single pose from a given RGB/RGB/flow stack
% Useful for checking whether the net is training properly.
assert(ndims(stack) == 3);
assert(size(stack, 3) == 8);
% Remember that we use the [h w c n] size required by Caffe (where n = 1)!
resp = net.forward({stack});
% Just return activations so that we can use show_stack
acts = resp{1};
end

