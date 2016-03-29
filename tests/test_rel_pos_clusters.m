function test_rel_pos_clusters
%TEST_REL_POS_CLUSTERS Visually check biposelet displacement means
% WARNING: I'm plotting using a reversed Y axis here; this emulates what
% imshow() does for you, and means that the joint coordinates (which also
% use a reversed y-axis) display properly.
bp_centroids = parload('cache/centroids.mat', 'centroids');
% Remember: indices are (subpose, child, parent, yx)
disps = parload('cache/subpose_disps.mat', 'subpose_disps');
conf = get_conf_mpii;
num_sp = length(conf.subpose_pa);
K = size(disps, 2);
assert(ndims(disps) == 4 && size(disps, 1) == num_sp && size(disps, 4) == 2);
colours = {'green', 'blue', 'red'};

% Choose some random types for our subposes
types = randi([1 K], [1 num_sp]);

clf;
hold on;
centers = zeros([2, num_sp]);
set(gca, 'Ydir', 'reverse');
for subpose_idx = 1:num_sp
    colour = colours{subpose_idx};
    fprintf('%s has colour %s\n', conf.subposes(subpose_idx).name, colour);
    
    poselet = types(subpose_idx);
    sp_coords = get_coords(bp_centroids, subpose_idx, poselet);
    
    pa = conf.subpose_pa(subpose_idx);
    if pa == 0
        midpoint = [0 0];
    else
        pa_poselet = types(pa);
        disp = squeeze(disps(subpose_idx, poselet, pa_poselet, :));
        midpoint = centers(:, pa)' - disp';
    end
    centers(:, subpose_idx) = midpoint;
    
    split = size(sp_coords, 1) / 2;
    fst = sp_coords(1:split, :); snd = sp_coords(split+1:end, :);
    
    % First frame
    subplot(1, 3, 1);
    axis equal;
    hold on;
    set(gca, 'Ydir', 'reverse');
    plot_subpose(fst, midpoint, conf.cnn.window, colour);
    
    % Second frame
    subplot(1, 3, 2);
    hold on;
    axis equal;
    set(gca, 'Ydir', 'reverse');
    plot_subpose(snd, midpoint, conf.cnn.window, colour);
    
    % Both at once
    subplot(1, 3, 3);
    hold on;
    axis equal;
    set(gca, 'Ydir', 'reverse');
    plot_subpose(sp_coords, midpoint, conf.cnn.window, colour);
end
hold off;
end

function shaped = get_coords(centroids, subpose, poselet)
raw = centroids{subpose}(poselet, :);
lr = length(raw);
assert(lr > 0 && mod(lr, 4) == 0);
num_coords = length(raw) / 2;
% coords = reshape(raw, [1 num_coords 2]);
shaped = reshape(raw, [2 num_coords])';
% midpoint = num_coords / 2;
% shaped = cat(1, coords(:, 1:midpoint, :), coords(:, midpoint+1:end, :));
% assert(all(size(shaped) == [2, lr/4, 2]));
end

function plot_subpose(coords, midpoint, box_size, colour)
shifted = bsxfun(@plus, coords, midpoint - box_size / 2);
assert(ismatrix(shifted) && size(shifted, 2) == 2);
plot_joints(shifted, colour);
end