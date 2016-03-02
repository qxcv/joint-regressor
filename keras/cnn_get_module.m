function module = cnn_get_module(modname)
%CNN_GET_MODULE Import Python modules necessary for evaluating CNN.
% This will fail if you haven't activated the current virtualenv!
%
% Importing Python modules from Matlab, like I'm doing here, is fraught
% with danger. This is because Matlab always links with its own version of
% the libraries it uses, but some of those libraries are also used by
% Python modules. Since Matlab has already loaded its own version, the
% Python modules might be exposed to a different version to the one they
% were compiled with.

% I don't think most modules bother telling you this, but h5py certainly
% does! If you get an error, you might need to recompile h5py, use
% LD_PRELOAD appropriately, or do HDF5_DISABLE_VERSION_CHECK=1 when
% launching Matlab.
% See https://au.mathworks.com/help/matlab/matlab_external/call-user-defined-custom-module.html
if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

module = py.importlib.import_module(modname);
end

