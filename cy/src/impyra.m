function pyra = impyra(im, flow, cnn_model, mean_pixels, step, cnn_size, ...
    scales, add_debug_fields)
% Compute feature pyramid.
%
% pyra.feat{i} is the i-th level of the feature pyramid.
% pyra.scales{i} is the scaling factor used for the i-th level.
% pyra.feat{i+interval} is computed at exactly half the resolution of feat{i}.
% first octave halucinates higher resolution data.

assert(ndims(im) == 3 && ndims(flow) == 3);
assert(size(im, 3) == 6, 'Need RGBRGB channels last');
assert(size(flow, 3) == 2, 'Need uv channels last');
assert(issorted(scales) && isvector(scales(end:-1:1)));

imsize = size(im);
flowsize = size(flow);
assert(all(imsize(1:2) == flowsize(1:2)));

% the ceil((x-1)/2) seems to be division by 2 with round-down (and a max to
% clamp in [0, infty)).
% The following checks are necessary so that our subsampling is the same in
% each dimension.
assert(isscalar(cnn_size) && isscalar(step));
pad = max(ceil((double(cnn_size-1)/2)), 0);

% pyra is structure
empty_cells = @() cell(length(scales), 1);
pyra = struct('feat', empty_cells(), 'raw_sizs', empty_cells(), ...
    'sizs', empty_cells(),  'in_rgb', empty_cells(), ...
    'in_flow', empty_cells(), 'raw_pad', empty_cells(), ...
    'pad', empty_cells(), 'raw_scale', empty_cells(), ...
    'scale', empty_cells());
clear empty_cells;

% Change down max_batch_size if you don't have enough memory for your
% choice of scales
max_batch_size = 4; % TODO: Intelligently decide what this should be based
                    % approximate memory constraints
for octave = 1:max_batch_size:length(scales)
    batch_size = min(max_batch_size, length(scales)-octave+1);
    octave_scales = scales(octave:octave+batch_size-1);
    
    % im_pyra and flow_pyra will store RGB and flow input pyramids for CNN,
    % respectively. Order for cnn_eval is NCHW.
    im_pyra = [];
    flow_pyra = [];
    for sub_scale = 0:batch_size-1
        % Start by scaling appropriately
        sub_scale_factor = octave_scales(sub_scale+1);
        scaled_im = imresize(im, sub_scale_factor);
        scaled_flow = smart_resize_flow(flow, size(scaled_im));
        assert(all(size(scaled_im) == size(scaled_flow) | [0 0 1]));
        
        % If input feature pyramids are empty, we initialise them with
        % enough space to hold the largest sample (the first one!)
        if isempty(im_pyra)
            assert(isempty(flow_pyra) && sub_scale == 0);
            im_pyra = zeros(batch_size, 6, size(scaled_im, 1) + 2*pad, ...
                size(scaled_im, 2) + 2*pad, 'single');
            flow_pyra = zeros(batch_size, 2, size(scaled_im, 1) + 2*pad, ...
                size(scaled_im, 2) + 2*pad, 'single');
        end
        assert(~isempty(im_pyra) && ~isempty(flow_pyra));
        
        % Pad so that we get an output volume covering the whole image
        scaled_im_pad = padarray(scaled_im, [pad, pad, 0], 'replicate');
        height = size(scaled_im_pad, 1);
        width = size(scaled_im_pad, 2);
        % Move into channels-first order
        scaled_im_shuf = permute(scaled_im_pad, [3 1 2]);
        % scaled_im_shuf = reshape(scaled_im_shuf, [1 size(scaled_im_shuf)]);
        % mlint AGROW is spurious
        im_pyra(sub_scale+1, :, 1:height, 1:width) = scaled_im_shuf; %#ok<AGROW>
        
        % Same for flow
        scaled_flow_pad = padarray(scaled_flow, [pad, pad, 0], 'replicate');
        assert(size(scaled_flow_pad, 1) == height ...
            && size(scaled_flow_pad, 2) == width);
        scaled_flow_shuf = permute(scaled_flow_pad, [3 1 2]);
        % scaled_flow_shuf = reshape(scaled_flow_shuf, [1 size(scaled_flow_shuf)]);
        flow_pyra(sub_scale+1, :, 1:height, 1:width) = scaled_flow_shuf; %#ok<AGROW>
        
        % .raw_sizs is size of output volume
        % TODO: Should this be floor(x) + 1 or ceil(x)? Only makes a
        % difference when x is whole, but could still be problematic.
        pyra(octave+sub_scale).raw_sizs = floor([height - cnn_size, ...
                                                 width - cnn_size] / step) + 1;
        pyra(octave+sub_scale).raw_scale = step / sub_scale_factor;
        pyra(octave+sub_scale).raw_pad = pad / step;
        if add_debug_fields
            pyra(octave+sub_scale).in_rgb = scaled_im_shuf;
            pyra(octave+sub_scale).in_flow = scaled_flow_shuf;
        end
    end
    % Result is an array with dimensions N*C*H*W (where C is the number of
    % outputs).
    resp = cnn_eval(cnn_model, im_pyra, flow_pyra, mean_pixels);
    parfor sub_scale = 0:batch_size-1
        feat = resp(sub_scale+1, :, ...
            1:pyra(octave + sub_scale).raw_sizs(1), ...
            1:pyra(octave + sub_scale).raw_sizs(2)); %#ok<PFBNS>
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
