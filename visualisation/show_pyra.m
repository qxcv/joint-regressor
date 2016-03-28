function show_pyra(pyra_path)
%SHOW_PYRA Visualise output of imCNNdet
% Useful for debugging CNN.
% Current idea: in one figure, display first frame, second frame, and flow.
% Then at the end display one unary map feature. Specific feature should be
% selectable with a slider.
DEFAULT_PATH = 'cache/pos-pyra/pos-pyra-1-iter-1.mat';
if ~exist('pyra_path', 'var');
    fprintf('Using default path of %s\n', DEFAULT_PATH);
    pyra_path = DEFAULT_PATH;
end
loaded = load(pyra_path);
pyra = loaded.pyra;
unary_map = loaded.unary_map;
% TODO: Make the following two variables selectable with sliders
scale = 1;
subpose = 2;
map_idx = 50;

% Extract data
im_stack = unperm(pyra(scale).in_rgb);
assert(ndims(im_stack) == 3 && size(im_stack, 3) == 6);
im_size = size(im_stack);
im1 = im_stack(:, :, 1:3);
im2 = im_stack(:, :, 4:6);
flow = unperm(pyra(scale).in_flow);
assert(ndims(flow) == 3);
flow_size = size(flow);
assert(all(im_size(1:2) == flow_size(1:2)));

fprintf('**********\n');
fprintf('Scale %i, subpose %i, map index %i\n', scale, subpose, map_idx);
% First image
subtightplot(1,4,1);
imshow(im1);

% Second image
subtightplot(1,4,2);
imshow(im2);

% Flow
subtightplot(1,4,3);
imshow(pretty_flow(flow));

% Heatmap
subtightplot(1,4,4);
heatmap = pretty_heatmap(unary_map{scale}{subpose}(:, :, map_idx), im_size(1:2));
colormap('jet');
imagesc(heatmap);
hm_range = [min(heatmap(:)) max(heatmap(:))];
fprintf('Range: %f->%f (%f->%f exp)\n', [hm_range exp(hm_range)]);
axis image off;
axis equal;
end

function up = unperm(permed)
up = permute(permed, [2 3 1]);
end

function pretty = pretty_heatmap(map, new_size)
% Need [height width] for new_size
assert(numel(new_size) == 2);
hs_size = size(map);
assert(numel(hs_size) == 2);
scale_factor = min(new_size ./ hs_size);
fprintf('map size: [%d %d]; im size: [%d %d]; scale: %f\n', [hs_size, new_size, scale_factor]);
assert(isscalar(scale_factor) && scale_factor > 0);
pretty = imresize(map, scale_factor);
end