function mean_pixel = compute_mean_pixel(filenames)
%COMPUTE_MEAN_PIXEL Compute the mean pixel of a list of HDF5 files
%(identified by cell array of filenames).
volumes = zeros(length(filenames), 1);
means = cell(length(filenames), 1);
assert(~isempty(filenames));
parfor i=1:length(filenames)
    fprintf('Reading %s\n', filenames{i});
    data = h5read(filenames{i}, '/data');
    % data is now an w * h * c * n (or h * w * c * n, not sure) matrix
    assert(ndims(data) == 4);
    volumes(i) = size(data, 1) * size(data, 2) * size(data, 4);
    means{i} = squeeze(mean(mean(mean(data, 2), 4), 1));
end
mean_pixel = zeros(size(means{1}));
weights = volumes / sum(volumes);
for i=1:length(filenames)
    mean_pixel = mean_pixel + means{i} * weights(i);
end
end

