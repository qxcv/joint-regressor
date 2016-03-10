function poselet_probs = cnn_eval(model, images_data, flow_data, mean_pixels)
%CNN_EVAL Evaluates our CNN by calling into Python
% Our code uses Keras, so evaluating directly in Matlab (or a mex library)
% won't work here.
assert(ndims(images_data) == 4, 'Need 4D RGB tensor (N6HW)');
assert(ndims(flow_data) == 4, 'Need 4D flow tensor (N2HW)');
assert(size(images_data, 2) == 6, 'RGBRGB channels should be second axis');
assert(size(flow_data, 2) == 2, 'uv channels should be second axis');
assert(isvector(mean_pixels.images) && isvector(mean_pixels.flow));

% Format for Keras input data: N*C*H*W (see
% http://keras.io/layers/convolutional/#convolution2d). I've been feeding
% it N*C*S*S, so hopefully the first S is height and the second is width :)
im_mean_pixel = reshape(mean_pixels.images, [1 length(mean_pixels.images) 1 1]);
norm_images_data = bsxfun(@minus, double(images_data), im_mean_pixel);
flow_mean_pixel = reshape(mean_pixels.flow, [1 length(mean_pixels.flow) 1 1]);
norm_flow_data = bsxfun(@minus, double(flow_data), flow_mean_pixel);

% Build dict and evaluate
data_dict = py.dict();
data_dict.setdefault('images', matlab2numpy(norm_images_data));
data_dict.setdefault('flow', matlab2numpy(norm_flow_data));
results = model.predict(data_dict);

% Get results in NCHW format (assuming fully convolutional)
% In the poselet case, the result will be (num samples)*301*height*width.
poselet_probs = numpy2matlab(results.get('poselet'));
end