% This is like Caffe's store2hdf5, but better :)

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

for i=1:2:length(varargin)
    dataset = varargin{i};
    data = varargin{i+1};
    data_dims = size(data);
    if length(data_dims) == 3
        % If we don't do this, then we can't write out samples one at a
        % time, since Matlab won't let us have a trailing dimension of size
        % 1 :(
        data_dims = [data_dims 1];
    end
    
    if ~exist('last_size', 'var')
        last_size = length(data_dims);
    else
        if length(data_dims) ~= last_size
            warning('Some inputs to store3hdf6 have different lengths');
        end
    end
    
    if ~hdf5_location_exists(filename, dataset)
        % we'll need to create the file
        % chunk size format is  [width, height, channels, number]
        h5create(filename, dataset, [data_dims(1:end-1) Inf], ...
            'Datatype', class(data), ...
            'ChunkSize', [data_dims(1:end-1) chunksz]);
        startloc = [ones(1, length(data_dims)-1), 1];
    else
        % otherwise, we can just write to the file
        info = h5info(filename);
        % strip leading slash
        ds_name = dataset(2:end);
        ds_idx = strcmp(ds_name, {info.Datasets.Name});
        prev_size = info.Datasets(ds_idx(1:1)).Dataspace.Size;
        assert(all(prev_size(1:end-1) == data_dims(1:end-1)), ...
            'Data dimensions must match existing dimensions in dataset');
        startloc = [ones(1, length(data_dims)-1), prev_size(end)+1];
    end
    
    h5write(filename, dataset, data, startloc, data_dims);
end
end