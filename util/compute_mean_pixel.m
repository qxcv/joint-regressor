function mean_pixel = compute_mean_pixel(filenames, fieldname)
%COMPUTE_MEAN_PIXEL Compute the mean pixel of a list of HDF5 files
%(identified by cell array of filenames).
volumes = zeros(length(filenames), 1);
means = cell(length(filenames), 1);
assert(~isempty(filenames));
fixed_field = regexprep(fieldname, '^/+', '');
display(filenames);
for i=1:length(filenames)
    fn = filenames{i};
    fprintf('Reading %s\n', fn);
    info = h5info(fn);
    ds_index = find(strcmp({info.Datasets.Name}, fixed_field));
    assert(numel(ds_index) == 1);
    chunk_size = info.Datasets(ds_index).ChunkSize;
    ds_size = info.Datasets(ds_index).Dataspace.Size;
    % Should be w * h * c * n (or h * w * c * n, not sure)
    assert(numel(chunk_size) == 4);
    assert(numel(ds_size) == 4);
    if isempty(chunk_size)
        % No chunking, so read everything at once
        chunk_size = ds_size;
    end
    batch_size = chunk_size(4);
    total_elems = ds_size(4);
    num_batches = ceil(total_elems / batch_size);
    batch_sizes = batch_size * ones([1 num_batches]);
    batch_sizes(end) = total_elems - batch_size * (num_batches - 1);
    batch_weights = batch_sizes ./ sum(batch_sizes);
    batch_starts = cumsum([1 batch_sizes]);
    batch_starts = batch_starts(1:end-1);
    assert(length(batch_starts) == length(batch_sizes));
    assert(batch_starts(1) == 1);
    assert(batch_starts(end) + batch_sizes(end) - 1 == total_elems);
    % batch_results is channels*num_batches matrix
    batch_results = zeros([ds_size(3) num_batches]);
    parfor batch_num=1:num_batches
        batch_start = batch_starts(batch_num);
        start = [1 1 1 batch_start];
        count = [ds_size(1:3) batch_sizes(batch_num)]; %#ok<PFBNS>
        batch_data = h5read(fn, fieldname, start, count);
        % This mean(mean(mean())) looks complicated, but we're just
        % averaging over every dimension that's not a pixel dimension.
        batch_results(:, batch_num) = ...
            squeeze(mean(mean(mean(batch_data, 2), 4), 1));
    end
    means{i} = batch_results * batch_weights';
    assert(numel(means{i}) == ds_size(3));
    volumes(i) = ds_size(1) * ds_size(2) * ds_size(4);
end
mean_pixel = zeros(size(means{1}));
weights = volumes / sum(volumes);
for i=1:length(filenames)
    assert(isscalar(weights(i)));
    mean_pixel = mean_pixel + means{i} * weights(i);
end
end

