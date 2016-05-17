function flow = broxOpticalFlow(im1, im2)
%BROXOPTICALFLOW Compute optical flow with OpenCV's Brox et al. implementation
% This runs on the GPU, so it's blazing fast. Note that this function is a
% wrapper around a mex file which does the heavy lifting of calling OpenCV.
sim1 = convertIm(im1);
sim2 = convertIm(im2);
assert(all(size(sim1) == size(sim2)));
flow = mex_broxOpticalFlow(sim1, sim2);
end

function safeIm = convertIm(im)
% Make sure the given image is safe to pass to _broxOpticalFlow
assert(ndims(im) == 3 || ismatrix(im));

% Convert to grayscale
if size(im, 3) == 3
    grayscale = rgb2gray(im);
elseif size(im, 3) ~= 1
    error('JointRegressor:broxOpticalFlow:invalidImage', 'Expected grayscale image')
else
    grayscale = im;
end

safeIm = single(grayscale);

% Convert uint8/uint16 to single
if isa(grayscale, 'uint8')
    safeIm = safeIm / 255.0;
elseif isa(grayscale, 'uint16')
    safeIm = safeIm / 65535.0;
else
    assert(isa(grayscale, 'single') || isa(grayscale, 'double'));
end
end
