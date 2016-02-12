function rv = hdf5_location_exists(filename, dataset)
%HDF5_LOCATION_EXISTS Check that the given HDF5 file and location within it
%(dataset, group, whatever) exist.
try
    h5info(filename, dataset);
    rv = true;
catch ex
    if strcmp(ex.identifier, 'MATLAB:imagesci:h5info:fileOpenErr') ...
            || strcmp(ex.identifier, 'MATLAB:imagesci:h5info:libraryError')
        rv = false;
    else
        rethrow(ex);
    end
end
end