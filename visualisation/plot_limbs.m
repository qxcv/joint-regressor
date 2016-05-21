function plot_limbs(joints, pa)
%PLOT_LIMBS Plot limbs defined by joint locations and parents array
assert(pa(1) <= 1);
cmap = colormap;
cmap_len = size(cmap, 1);
num_sticks = length(pa) - 1;
for this_joint=2:length(pa)
    cmap_idx = floor((this_joint - 2) * cmap_len / num_sticks) + 1;
    color = cmap(cmap_idx, :);
    parent = pa(this_joint);
    inds = [this_joint parent];
    line(joints(inds, 1), joints(inds, 2), 'Color', color);
end

