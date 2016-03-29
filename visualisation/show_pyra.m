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

global scale subpose map_idx pyra unary_map best_act;

loaded = load(pyra_path);
pyra = loaded.pyra;
unary_map = loaded.unary_map;
% Highest activation
best_act = highest_act(unary_map);
fprintf('Best activation: %f\n', best_act);

% Initial plot
scale = 1;
subpose = 1;
map_idx = 1;
do_plot;

% Set up uicontrol callbacks
conf = get_conf_mpii;
sp_names = {conf.subposes.name};
num_maps = conf.biposelet_classes;
num_scales = length(pyra);
uicontrol('Style', 'text',...
          'Position', [10 10, 80 15],...
          'String', 'Subpose');
uicontrol('Style', 'popup',...
          'String', sp_names,...
          ... left, bottom, width, height
          'Position', [100 10 100 15],...
          'Callback', @set_subpose);
uicontrol('Style', 'text',...
          'Position', [10 30, 80 15],...
          'String', 'Scale');
uicontrol('Style', 'slider',...
          'Min', 1, 'Max', num_scales, 'Value', scale,...
          'Position', [100 30 300 15],...
          'SliderStep', [1 1],...
          'Callback', @set_scale);
uicontrol('Style', 'text',...
          'Position', [10 50, 80 15],...
          'String', 'Poselet');
uicontrol('Style', 'slider',...
          'Min', 1, 'Max', num_maps, 'Value', map_idx,...
          'Position', [100 50 300 15],...
          'Callback', @set_map_idx);
end

function set_subpose(source, ~)
global subpose;
subpose = round(source.Value);
do_plot;
end

function set_map_idx(source, ~)
global map_idx;
map_idx = round(source.Value);
do_plot(false);
end

function set_scale(source, ~)
global scale
scale = round(source.Value);
do_plot;
end

function do_plot(redraw_input)
global scale subpose map_idx pyra unary_map best_act;
if nargin < 1
    redraw_input = true;
end
fprintf('**********\n');
fprintf('Scale %i, subpose %i, map index %i, redraw? %i\n', scale, subpose, map_idx, redraw_input);

% Extract data
im_stack = unperm(pyra(scale).in_rgb);
assert(ndims(im_stack) == 3 && size(im_stack, 3) == 6);
im_size = size(im_stack);

if redraw_input
    im1 = im_stack(:, :, 1:3);
    im2 = im_stack(:, :, 4:6);
    flow = unperm(pyra(scale).in_flow);
    assert(ndims(flow) == 3);
    flow_size = size(flow);
    assert(all(im_size(1:2) == flow_size(1:2)));
    
    % First image
    subtightplot(1,5,1);
    imshow(im1);
    
    % Second image
    subtightplot(1,5,2);
    imshow(im2);
    
    % Flow
    subtightplot(1,5,3);
    imshow(pretty_flow(flow));
end

% Heatmap 2 (all poselets)
subtightplot(1,5,4);
colormap('hot');
all_maps = unary_map{scale}{subpose};
best_map = pretty_heatmap(max(all_maps, [], 3), im_size(1:2));
imagesc(best_map, [0 exp(best_act)]);
bm_range = [min(best_map(:)) max(best_map(:))];
fprintf('Best range (all poselets): %f->%f\n', bm_range);
axis image off
axis equal;

subtightplot(1,5,5);
heatmap = pretty_heatmap(all_maps(:, :, map_idx), im_size(1:2));
imagesc(heatmap, [0 exp(best_act)]);
hm_range = [min(heatmap(:)) max(heatmap(:))];
fprintf('Poselet range: %f->%f\n', hm_range);
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
pretty = exp(imresize(map, scale_factor));
end

function best = highest_act(unary_map)
best = -Inf;
for i=1:length(unary_map)
    for j=1:length(unary_map{i})
        best = max([best unary_map{i}{j}(:)']);
    end
end
end