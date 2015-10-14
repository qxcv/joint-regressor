function show_stack(stack, labels)
%SHOW_STACK Show a stack of RGB/RGB/flow. Optionally show labels as well.
stack = permute(stack, [2 1 3]);
all_joints = reshape(labels, [2, numel(labels) / 2])';
per_set = size(all_joints, 1) / 2;
j1 = all_joints(1:per_set, :);
j2 = all_joints((per_set+1):end, :);

subplot(1, 4, 1);
title('First frame');
imshow(stack(:, :, 1:3));
hold on;
plot_joints(j1);
hold off;

subplot(1, 4, 2);
title('Second frame');
imshow(stack(:, :, 4:6));
hold on;
plot_joints(j2);
hold off;

subplot(1, 4, 3);
title('Flow');
% Have to flip the flow horizontally because of how quiver works
flow = stack(end:-1:1, :, 7:8);
flow = imresize(flow, 0.05);
quiver(flow(:, :, 1), flow(:, :, 2));
axis equal;
axis([0, size(flow, 2), 0, size(flow, 1)]);
axis off;

subplot(1, 4, 4);
title('Flow magnitude');
flow = stack(:, :, 7:8);
mags = sqrt(sum(flow.^2, 3));
norm_mags = mags / max(mags(:));
imshow(norm_mags);
% axis equal;
% axis([0, size(flow, 2), 0, size(flow, 1)]);
% axis off;
end