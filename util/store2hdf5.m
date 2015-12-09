% This file pulled from BVLC/caffe master at d3d7d07be981ab046da99e1ad5d84217bf0ec4c3
% It will probably end up with some changes for my purposes :-)

function [curr_dat_sz, curr_lab_sz] = store2hdf5(filename, data, joint_locs, create, startloc, chunksz)  
  % *data* is W*H*C*N matrix of images should be normalized (e.g. to lie
  % between 0 and 1) beforehand
  % *joint_locs* is D*N matrix of joint locations (D joint locations per
  % sample)
  % *create* [0/1] specifies whether to create file newly or to append to
  % previously created file, useful to store information in batches when a
  % dataset is too big to be held in memory  (default: 1)
  % *startloc* (point at which to start writing data). By default, if
  % create=1 (create mode), startloc.data=[1 1 1 1], and startloc.lab=[1
  % 1]; if create=0 (append mode), startloc.data=[1 1 1 K+1], and
  % startloc.lab = [1 K+1]; where K is the current number of samples stored
  % in the HDF
  % *chunksz* (used only in create mode), specifies number of samples to be
  % stored per chunk (see HDF5 documentation on chunking) for creating HDF5
  % files with unbounded maximum size - TLDR; higher chunk sizes allow
  % faster read-write operations

  % verify that format is right
  dat_dims=size(data);
  lab_dims=size(joint_locs);
  if length(dat_dims) == 3
      % If we don't do this, then we can't write out samples one at a time,
      % since Matlab won't let us have a trailing dimension of size 1 :(
      dat_dims = [dat_dims 1];
  end
  num_samples=dat_dims(end);

  assert(lab_dims(end)==num_samples, 'Number of samples should be matched between data and labels');

  if ~exist('create','var')
    create=true;
  end

  
  if create
    %fprintf('Creating dataset with %d samples\n', num_samples);
    if ~exist('chunksz', 'var')
      chunksz=1000;
    end
    if exist(filename, 'file')
      fprintf('Warning: replacing existing file %s \n', filename);
      delete(filename);
    end      
    h5create(filename, '/data', [dat_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [dat_dims(1:end-1) chunksz]); % width, height, channels, number 
    h5create(filename, '/label', [lab_dims(1:end-1) Inf], 'Datatype', 'single', 'ChunkSize', [lab_dims(1:end-1) chunksz]); % width, height, channels, number 
    if ~exist('startloc','var') 
      startloc.dat=[ones(1,length(dat_dims)-1), 1];
      startloc.lab=[ones(1,length(lab_dims)-1), 1];
    end 
  else  % append mode
    if ~exist('startloc','var')
      info=h5info(filename);
      prev_dat_sz=info.Datasets(1).Dataspace.Size;
      prev_lab_sz=info.Datasets(2).Dataspace.Size;
      assert(all(prev_dat_sz(1:end-1)==dat_dims(1:end-1)), 'Data dimensions must match existing dimensions in dataset');
      assert(all(prev_lab_sz(1:end-1)==lab_dims(1:end-1)), 'Label dimensions must match existing dimensions in dataset');
      startloc.dat=[ones(1,length(dat_dims)-1), prev_dat_sz(end)+1];
      startloc.lab=[ones(1,length(lab_dims)-1), prev_lab_sz(end)+1];
    end
  end

  if ~isempty(data)
    h5write(filename, '/data', single(data), startloc.dat, dat_dims);
    h5write(filename, '/label', single(joint_locs), startloc.lab, lab_dims);
  end

  if nargout
    info=h5info(filename);
    curr_dat_sz=info.Datasets(1).Dataspace.Size;
    curr_lab_sz=info.Datasets(2).Dataspace.Size;
  end
end
