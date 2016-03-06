function np_data = matlab2numpy(some_data)
%MATLAB2NUMPY Convert Matlab array to Numpy one.
%Doesn't handle vectors properly because hey, neither does Matlab.
assert(isnumeric(some_data));
np = cnn_get_module('numpy');
% Only using int32 because array.array doesn't support int64
real_shape = py.tuple(int32(size(some_data)));
np_data_unshaped = np.array(some_data(:)');
np_data = np_data_unshaped.reshape(real_shape, pyargs('order', 'F'));
end