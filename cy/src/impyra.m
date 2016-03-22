function pyra = impyra(im, flow, cnn_model, mean_pixels, step, psize, ...
    interval, scale_factor)
% Compute feature pyramid.
%
% pyra.feat{i} is the i-th level of the feature pyramid.
% pyra.scales{i} is the scaling factor used for the i-th level.
% pyra.feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.

% These next checks might need to changed if it's easier to organise things
% some other way.
assert(ndims(im) == 3 && ndims(flow) == 3);
assert(size(im, 1) == 6, 'Need RGBRGB channels second');
assert(size(flow, 1) == 2, 'Need uv channels second');

flow_mp = reshape(mean_pixels.flow, [1 2 1 1]);
image_mp = reshape(mean_pixels.images, [1 6 1 1]);

imsize = size(im);
flowsize = size(flow);
assert(all(imsize(1:2) == flowsize(1:2)));

im = imresize(im, scale_factor);  % may upscale image to better handle small objects.
flow = smart_resize_flow(flow, scale_factor);
assert(all(size(im) == size(flow) | [0 0 1]));

% TODO: What is psize?
% Okay, it's the expected size of part, expressed in the side length of the
% region which contains it in the output volume. That's what it's computed
% as tsize*psize in build_model.m, where tsize is the expected size in
% input pixels of a part and psize is the downsampling factor (stride) of
% the fully convolutional network.

% the ceil((x-1)/2) seems to be division by 2 with round-down (and a max to
% clamp in [0, infty)).
% The following checks are necessary so that our subsampling is the same in
% each dimension.
assert(psize(1) == psize(2));
assert(step(1) == step(2));
pad = max(ceil((double(psize(1)-1)/2)), 0);
sc = 2 ^(1/interval);
imsize = [size(im, 1), size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/max(psize))/log(sc));

% pyra is structure
pyra = struct('feat', cell(max_scale, 1), 'sizs', cell(max_scale, 1), ...
    'scale', cell(max_scale, 1), 'padx', cell(max_scale, 1), ...
    'pady', cell(max_scale, 1));

% Change down max_batch_size if you don't have enough memory for your
% choice of scales
max_batch_size = interval;
for octave = 1:max_batch_size:max_scale
    scaled_im = imresize(im, 1/sc^(octave-1));
    scaled_flow = smart_resize_flow(flow, size(scaled_im));
    assert(all(size(scaled_im) == size(scaled_flow) | [0 0 1]));
    
    num = min(max_batch_size, max_scale-octave+1);
    % Order for cnn_eval is NCHW
    im_pyra = zeros(...
        num, 3, size(scaled_im, 1)   + 2*pad, size(scaled_im, 2)   + 2*pad, ...
        'single');
    flow_pyra = zeros(...
        num, 2, size(scaled_flow, 1) + 2*pad, size(scaled_flow, 2) + 2*pad, ...
        'single');
    for sub_scale = 0:num-1
        % Pad so that we get an output volume covering the whole image
        % Images first
        scaled_im_pad = padarray(scaled_im, [pad, pad, 0], 'replicate');
        scaled_im_pad = bsxfun(@minus, scaled_im_pad, image_mp);
        im_pyra(sub_scale+1, :, 1:size(scaled_im_pad,1), 1:size(scaled_im_pad,2)) = scaled_im_pad;
        % Flow second
        scaled_flow_pad = padarray(scaled_flow, [pad, pad, 0], 'replicate');
        scaled_flow_pad = bsxfun(@minus, scaled_flow_pad, flow_mp);
        flow_pyra(sub_scale+1, :, 1:size(scaled_flow_pad,1), 1:size(scaled_flow_pad,2)) = scaled_flow_pad;
        
        % This output size function was used in the original code because
        % it reflects how Caffe computes fully convolutional network output
        % volumes. It turns out that Keras does the same thing (hooray!),
        % so I can keep it :)
        pyra(octave+sub_scale).sizs = floor([size(scaled_im_pad, 1) - psize(1), ...
                                size(scaled_im_pad, 2) - psize(2)] / step) + 1;
        
        pyra(octave+sub_scale).scale = step / (scale_factor * 1/sc^(octave-1+sub_scale));
        pyra(octave+sub_scale).pad = pad / step;
        
        scaled_im = imresize(scaled_im, 1/sc);
        scaled_flow = smart_resize_flow(flow, size(scaled_im));
        assert(all(size(scaled_im) == size(scaled_flow) | [0 0 1]));
    end
    % Result is an array with dimensions N*C*H*W (where C is the number of
    % outputs).
    resp = cnn_eval(cnn_model, im_pyra, flow_pyra, mean_pixels);
    for sub_scale = 0:num-1
        pyra(octave+sub_scale).feat = resp(sub_scale+1, :, ...
            1:pyra(octave + sub_scale).sizs(1), ...
            1:pyra(octave + sub_scale).sizs(2));
    end
end