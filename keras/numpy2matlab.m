function matlab_data = numpy2matlab(np_data)
%NUMPY2MATLAB Reverse of matlab2numpy

% Grab correct dimensions
real_shape = cell2mat(py.list(np_data.shape).cell);
if length(real_shape) == 1
    real_shape = [1 real_shape];
end
assert(ndims(real_shape) >= 2);

% Convert data via an array.array
double_data = np_data.astype('double');
array_data = py.array.array('d', double_data.flatten('F'));
matlab_data = reshape(array_data.double, real_shape);
end