% This is like Caffe's store2hdf5, but better. Supports a variable number
% of datasets to write, and has some sensible defaults that only make sense
% when not using Caffe (e.g. it always uses the type of the data you give
% it, instead of converting everything to single).

function store3hdf6(filename, opts, varargin)
% *filename* is the path to the HDF5 file
%
% *opts* is a struct (or other data type, in which case it is ignored)
% containing zero or more of the fields listed below.
%
% *opts.chunksz* (used only in create mode), specifies number of samples
% to be stored per chunk (see HDF5 documentation on chunking) for
% creating HDF5 files with unbounded maximum size - TLDR; higher chunk
% sizes allow faster read-write operations
%
% *opts.deflate* is the compression level
%
% *varargin* should contain a series of pairs of arguments consisting of
% a dataset name (e.g. '/label') and some data to write. If the given
% dataset exists, the data to write will be appended to the existing data
% by concatenating the new data with the existing data along the first
% axis. If the given dataset does not exist (including if the file has
% only just been created), then a new dataset with unbounded first
% dimension and remaining dimensions just like the given data will be
% created with the appropriate name. Note that image data should be
% Width*Height*Channels*Number, where the last dimension is the number of
% samples you are storing (this is the axis along which samples will be
% stacked in append mode).
%
% Usage example:
%
% store3hdf6('my_file.h5', {}, '/label', [1; 2; 3])
%
% If my_file.h5 does not exist, then that will create a new file called
% my_file.h5 and add a 1D dataset with path '/label' and values (1, 2, 3).

if isstruct(opts) && hasfield(opts, 'chunksz')
    chunksz = opts.chunksz;
else
    chunksz = 1024;
end

if isstruct(opts) && hasfield(opts, 'deflate')
    deflate = opts.deflate;
else
    deflate = false;
end

ds_info_map = containers.Map('KeyType', 'char', 'ValueType', 'any');
if exist(filename, 'file')
    % h5info takes a long time on large, remote datasets, so we need to
    % make sure that we're only calling it when absolutely necessary.
    all_info = h5info(filename);
    for ds_idx=1:length(all_info.Datasets)
        ds_info_map(all_info.Datasets(ds_idx).Name) = all_info.Datasets(ds_idx);
    end
end

% Sanity check: because of our h5info caching shenanigans, we need to be
% careful with rewriting datasets, so no including datasets twice!
ds_names = varargin(1:2:end);
assert(length(unique(ds_names)) == length(varargin) / 2, ...
    ['Each piece of at must have a dataset name, and you can''t ' ...
     'specify dataset names twice']);

for i=1:2:length(varargin)
    dataset = varargin{i};
    % Strip leading slash
    ds_name = regexprep(dataset, '^/', '');
    data = varargin{i+1};
    data_dims = size(data);
    if ndims(data) == 3
        % If we don't do this, then we can't write out samples one at a
        % time, since Matlab won't let us have a trailing dimension of size
        % 1 :(
        data_dims = [data_dims 1]; %#ok<AGROW>
    end
    
    if ~ds_info_map.isKey(ds_name)
        % we'll need to create the dataset
        % chunk size format is  [width, height, channels, number]
        create_args = {filename, dataset, [data_dims(1:end-1) Inf], ...
            'Datatype', class(data), ...
            'ChunkSize', [data_dims(1:end-1) chunksz]};
        if deflate
            create_args{length(create_args)+1} = 'Deflate';
            create_args{length(create_args)+1} = deflate;
            % Shuffle=true really should be the default when Deflate>0,
            % but it looks like Matlab needs you to specify it yourself.
            create_args{length(create_args)+1} = 'Shuffle';
            create_args{length(create_args)+1} = true;
        end
        h5create(create_args{:});
        startloc = [ones(1, length(data_dims)-1), 1];
    else
        % otherwise, we can just write to the file
        ds_struct = ds_info_map(ds_name);
        prev_size = ds_struct.Dataspace.Size;
        assert(all(prev_size(1:end-1) == data_dims(1:end-1)), ...
            'Data dimensions must match existing dimensions in dataset');
        startloc = [ones(1, length(data_dims)-1), prev_size(end)+1];
    end
    
    h5write(filename, dataset, data, startloc, data_dims);
end
end
