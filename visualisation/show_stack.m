function show_stack(stack, labels)
%SHOW_STACK Show a stack of RGB/RGB/flow. Optionally show labels as well.
if size(labels, 1) == 1
    % Use column vector
    labels = labels';
end
all_joints = reshape(labels, [numel(labels) / 2, 2]);
per_set = size(all_joints, 1) / 2;
j1 = all_joints(1:per_set, :);
j2 = all_joints((per_set+1):end, :);

imshow(stack(:, :, 1:3));
hold on;
plot_joints(j1);
hold off;

imshow(stack(:, :, 4:6));
hold on;
plot_joints(j2);
hold off;

quiver(stack(:, :, 7), stack(:, :, 8));
end

