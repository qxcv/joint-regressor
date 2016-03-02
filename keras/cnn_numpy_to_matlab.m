function matlab_array = cnn_numpy_to_matlab(np_array)
%CNN_NUMPY_TO_MATLAB Convert ndarray to Matlab double matrix
numpy = cnn_get_module('numpy');
assert(py.isinstance(np_array, numpy.ndarray));
darray = np_array.astype('double');
fortran_array = numpy.asfortranarray(darray);
py_array = py.array.array('d');
py_array.fromstring(fortran_array.data);
matlab_array = double(py_array);
matlab_array = reshape(matlab_array, cellfun(@double, cell(np_array.shape)));
end

