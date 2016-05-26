function plot_limbs(joints, pa)
%PLOT_LIMBS Plot limbs defined by joint locations and parents array

% Pick the colours. We want bold colours, so I'm using the prism (rainbow)
% colour scheme.
cmap = prism;
cmap = unique(cmap, 'rows');
rng(42);
cmap = cmap(randperm(size(cmap, 1)), :);
rng('shuffle');
cmap_len = size(cmap, 1);

% Choose which joints are children of some other joint (i.e. correspond to
% sticks)
sticks = ~isnan(pa) & pa ~= 1:length(pa);
num_sticks = sum(sticks);
stick_idxs = cumsum(sticks);

for this_joint=2:length(pa)
    parent = pa(this_joint);
    if isnan(parent) || parent == this_joint
        continue
    end
    stick_idx = stick_idxs(this_joint);
    cmap_idx = ceil(stick_idx * cmap_len / num_sticks);
    color = cmap(cmap_idx, :);
    inds = [this_joint parent];
    line(joints(inds, 1), joints(inds, 2), ...
        'Color', color, ...
        'LineWidth', 2);
end

