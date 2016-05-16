#include <cstdint>

#include "opencv2/cuda.hpp"
#include "mex.h"

/* Quick-and-dirty MEx API to run the calls described in the OpenCV 2 manual:
 *
 * http://docs.opencv.org/2.4/modules/gpu/doc/initalization_and_information.html
 */

enum class Command {GET_DEVICE_COUNT, GET_DEVICE, SET_DEVICE, RESET_DEVICE};

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Initial argument validation
    if (nlhs > 1) {
        mexErrMsgIdAndTxt(
            "JointRegressor:cvGPUDevice:invalidArgs", "Only one output is produced"
        );
        return;
    }
    if (nrhs < 1 || mxGetClassID(prhs[0]) != mxCHAR_CLASS
            || mxGetNumberOfDimensions(prhs[0]) > 2) {
        mexErrMsgIdAndTxt(
            "JointRegressor:cvGPUDevice:invalidArgs", "Need command as first input"
        );
        return;
    }
    
    // Parse command string
    char *commandString = mxArrayToString(prhs[0]);
    Command cmd;
    if (0 == strcmp(commandString, "setDevice")) {
        cmd = Command::SET_DEVICE;
    } else if (0 == strcmp(commandString, "resetDevice")) {
        cmd = Command::RESET_DEVICE;
    }  else if (0 == strcmp(commandString, "getDevice")) {
        cmd = Command::GET_DEVICE;
    }  else if (0 == strcmp(commandString, "getCudaEnabledDeviceCount")) {
        cmd = Command::GET_DEVICE_COUNT;
    } else {
        mexErrMsgIdAndTxt(
            "JointRegressor:cvGPUDevice:invalidArgs", "Unknown command"
        );
        return;
    }
    
    // Validate arguments (was going to be a switch, but realised that
    // current approach with enums and stuff is overkill while writing the
    // function)
    if (cmd == Command::SET_DEVICE) {
        if (nrhs != 2 || !mxIsUint32(prhs[1])) {
            mexErrMsgIdAndTxt(
                "JointRegressor:cvGPUDevice:invalidArgs", "Need uint32 device number to set"
            );
            return;
        }
    } else {
        if (nrhs != 1) {
            mexErrMsgIdAndTxt(
                "JointRegressor:cvGPUDevice:invalidArgs", "This command doesn't take any arguments"
            );
            return;
        }
    }
    
    double out_arg = -1;
    uint32_t deviceNum, numDevices;
    switch (cmd) {
        case Command::GET_DEVICE:
            out_arg = cv::cuda::getDevice();
            break;
        case Command::SET_DEVICE:
            deviceNum = *(uint32_t*)mxGetData(prhs[1]);
            numDevices = (uint32_t)cv::cuda::getCudaEnabledDeviceCount();
            if (deviceNum >= numDevices) {
                mexErrMsgIdAndTxt(
                    "JointRegressor:cvGPUDevice:invalidArgs", "Not enough devices"
                );
            } else {
                cv::cuda::setDevice(deviceNum);
            }
            break;
        case Command::RESET_DEVICE:
            cv::cuda::resetDevice();
            break;
        case Command::GET_DEVICE_COUNT:
            out_arg = (double)cv::cuda::getCudaEnabledDeviceCount();
            break;
        default:
            mexErrMsgIdAndTxt(
                "JointRegressor:cvGPUDevice:invalidArgs", "Unhandled command"
            );
    }
    
    // Output handling
    switch (cmd) {
        case Command::GET_DEVICE:
        case Command::GET_DEVICE_COUNT:
            plhs[0] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
            double *output = mxGetPr(plhs[0]);
            output[0] = out_arg;
            break;
    }
}
