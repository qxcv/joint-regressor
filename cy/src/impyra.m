function pyra = impyra(im, flow, cnn_model, mean_pixels, step, cnn_size, ...
    interval, scale_factor)
% Compute feature pyramid.
%
% pyra.feat{i} is the i-th level of the feature pyramid.
% pyra.scales{i} is the scaling factor used for the i-th level.
% pyra.feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.

assert(ndims(im) == 3 && ndims(flow) == 3);
assert(size(im, 3) == 6, 'Need RGBRGB channels last');
assert(size(flow, 3) == 2, 'Need uv channels last');

imsize = size(im);
flowsize = size(flow);
assert(all(imsize(1:2) == flowsize(1:2)));

im = imresize(im, scale_factor);  % may upscale image to better handle small objects.
flow = smart_resize_flow(flow, scale_factor);
assert(all(size(im) == size(flow) | [0 0 1]));

% psize is the expected size of part, expressed in the side length of the
% region which contains it in the output volume. That's what it's computed
% as tsize*step in build_model.m, where tsize is the expected size in
% input pixels of a part and step is the downsampling factor (stride) of
% the fully convolutional network.

% the ceil((x-1)/2) seems to be division by 2 with round-down (and a max to
% clamp in [0, infty)).
% The following checks are necessary so that our subsampling is the same in
% each dimension.
assert(isscalar(cnn_size) && isscalar(step));
pad = max(ceil((double(cnn_size-1)/2)), 0);
sc = 2 ^(1/interval);
imsize = [size(im, 1), size(im, 2)];
max_scale = 1 + floor(log(min(imsize)/cnn_size)/log(sc));

% pyra is structure
pyra = struct('feat', cell(max_scale, 1), 'sizs', cell(max_scale, 1), ...
    'scale', cell(max_scale, 1), 'padx', cell(max_scale, 1), ...
    'pady', cell(max_scale, 1), ...
    'in_rgb', cell(max_scale, 1), 'in_flow', cell(max_scale, 1));

% Change down max_batch_size if you don't have enough memory for your
% choice of scales
max_batch_size = 1; % TODO: Intelligently decide what this should be based
                    % approximate memory constraints
for octave = 1:max_batch_size:max_scale
    scaled_im = imresize(im, 1/sc^(octave-1));
    scaled_flow = smart_resize_flow(flow, size(scaled_im));
    assert(all(size(scaled_im) == size(scaled_flow) | [0 0 1]));
    
    num = min(max_batch_size, max_scale-octave+1);
    % Order for cnn_eval is NCHW
    im_pyra = zeros(...
        num, 6, size(scaled_im, 1) + 2*pad, size(scaled_im, 2) + 2*pad, ...
        'single');
    flow_pyra = zeros(...
        num, 2, size(scaled_flow, 1) + 2*pad, size(scaled_flow, 2) + 2*pad, ...
        'single');
    for sub_scale = 0:num-1
        % Pad so that we get an output volume covering the whole image
        % Images first
        scaled_im_pad = padarray(scaled_im, [pad, pad, 0], 'replicate');
        height = size(scaled_im_pad, 1);
        width = size(scaled_im_pad, 2);
        % Move into channels-first order
        scaled_im_shuf = permute(scaled_im_pad, [3 1 2]);
        % scaled_im_shuf = reshape(scaled_im_shuf, [1 size(scaled_im_shuf)]);
        im_pyra(sub_scale+1, :, 1:height, 1:width) = scaled_im_shuf;
        
        % Flow second
        scaled_flow_pad = padarray(scaled_flow, [pad, pad, 0], 'replicate');
        assert(size(scaled_flow_pad, 1) == height ...
            && size(scaled_flow_pad, 2) == width);
        scaled_flow_shuf = permute(scaled_flow_pad, [3 1 2]);
        % scaled_flow_shuf = reshape(scaled_flow_shuf, [1 size(scaled_flow_shuf)]);
        flow_pyra(sub_scale+1, :, 1:height, 1:width) = scaled_flow_shuf;
        
        % This output size function was used in the original code because
        % it reflects how Caffe computes fully convolutional network output
        % volumes. It turns out that Keras does the same thing (hooray!),
        % so I can keep it :)
        % TODO: Should this be floor(x) + 1 or ceil(x)? Only makes a
        % difference when x is whole, but could still be problematic.
        pyra(octave+sub_scale).sizs = floor([height - cnn_size, ...
                                             width - cnn_size] / step) + 1;
        
        pyra(octave+sub_scale).scale = step / (scale_factor * 1/sc^(octave-1+sub_scale));
        pyra(octave+sub_scale).pad = pad / step;
        pyra(octave+sub_scale).in_rgb = scaled_im_shuf;
        pyra(octave+sub_scale).in_flow = scaled_flow_shuf;
        
        scaled_im = imresize(scaled_im, 1/sc);
        scaled_flow = smart_resize_flow(flow, size(scaled_im));
        assert(all(size(scaled_im) == size(scaled_flow) | [0 0 1]));
    end
    % Result is an array with dimensions N*C*H*W (where C is the number of
    % outputs).
    resp = cnn_eval(cnn_model, im_pyra, flow_pyra, mean_pixels);
    for sub_scale = 0:num-1
        feat = resp(sub_scale+1, :, ...
            1:pyra(octave + sub_scale).sizs(1), ...
            1:pyra(octave + sub_scale).sizs(2));
        assert(ndims(feat) == 4);
        assert(size(feat, 1) == 1);
        feat_size = size(feat);
        % Get rid of singleton leading dimension
        feat = reshape(feat, feat_size(2:end));
        % Put channels last
        feat = permute(feat, [2 3 1]);
        pyra(octave+sub_scale).feat = feat;
    end
end