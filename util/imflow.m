function out_flow = imflow(frame_1, frame_2)
%IMFLOW Compute optical flow between two images, specified by their paths
% Note that we *only* open frame_1 to get its width and height
is_valid = @(f) ndims(f) == 3 && size(f, 3) == 3 && isa(f, 'uint8');
assert(is_valid(frame_1) && is_valid(frame_2), ...
    'Frames must be 3D RGB uint8 arrays');
out_flow = broxOpticalFlow(frame_1, frame_2);
end
