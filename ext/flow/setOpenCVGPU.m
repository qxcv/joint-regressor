function new_gpu = setOpenCVGPU(gpu_num)
%SETOPENCVGPU Set GPU used by OpenCV
gpu = uint32(gpu_num);
assert(gpu == gpu_num, 'Need non-negative integral value');
num_devices = mex_cvGPUDevice('getCudaEnabledDeviceCount');
assert(gpu < num_devices, 'Only have %i devices', num_devices);
if gpu ~= mex_cvGPUDevice('getDevice')
    mex_cvGPUDevice('resetDevice');
    mex_cvGPUDevice('setDevice', gpu);
end
new_gpu = mex_cvGPUDevice('getDevice');
end

