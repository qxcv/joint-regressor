#include <vector>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "mex.h"

/* This is a mex wrapper for Matlab's CUDA implementation of Brox et al.'s
 * optical flow implementation[0], which is apparently insanely fast.
 *
 * Usage:
 *
 *  flow_field = brox_optical_flow_mex(first_frame, second_frame);
 *
 * [0] http://docs.opencv.org/3.0-beta/modules/cudaoptflow/doc/optflow.html */

bool verifyMatrix(const mxArray *mat) {
    // uint8s are returned by imread() (which we want to support out of the box)
    return mxGetClassID(mat) == mxSINGLE_CLASS && mxGetNumberOfDimensions(mat) == 2;
}

bool dimensionsMatch(const mxArray *mat1, const mxArray *mat2) {
    mwSize ndim = mxGetNumberOfDimensions(mat1);
    if (ndim != mxGetNumberOfDimensions(mat2)) {
        return false;
    }
    const mwSize *dim1 = mxGetDimensions(mat1);
    const mwSize *dim2 = mxGetDimensions(mat2);
    for (mwSize i = 0; i < ndim; i++) {
        if (dim1[i] != dim2[i]) {
            return false;
        }
    }
    return true;
}

cv::cuda::GpuMat toGPUMat(const mxArray *m) {
    // XXX: mxGetData will return something in column-major order (so the first
    // m cells represent the rows of the first column), but I think openCV
    // expects row-major order. Looks like I'm going to have to copy in the data
    // manually, or perhaps do a transpose in Matlab (at best) :(
    float *columnMajor = (float*)mxGetData(m);
    size_t rows = mxGetM(m);
    size_t cols = mxGetN(m);
    cv::Mat cvMatrix = cv::Mat((int)rows, (int)cols, CV_32F);
    for (size_t row = 0; row < rows; row++) {
        for (size_t col = 0; col < cols; col++) {
            cvMatrix.at<float>(row, col) = columnMajor[col * rows + row];
        }
    }
    return cv::cuda::GpuMat(cvMatrix);
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if (nlhs > 1) {
        mexErrMsgIdAndTxt(
            "JointRegressor:brox_optical_flow_mex:nlhs", "Only one output is produced"
        );
        return;
    }

    if (nrhs != 2) {
        mexErrMsgIdAndTxt(
            "JointRegressor:brox_optical_flow_mex:nrhs", "Two inputs required"
        );
        return;
    }

    if (!verifyMatrix(prhs[0]) || !verifyMatrix(prhs[1])) {
        mexErrMsgIdAndTxt(
            "JointRegressor:brox_optical_flow_mex:prhs", "Inputs should be 2D single arrays"
        );
        return;
    }

    if (!dimensionsMatch(prhs[0], prhs[1])) {
        mexErrMsgIdAndTxt(
            "JointRegressor:brox_optical_flow_mex:prhs", "Input dimensions don't match"
        );
        return;
    }

    /* Now we can read the images and convert them to CV_32FC1 (grayscale image,
     * single-precision float).
     * This is straight from the optical_flow.cpp example in the repository. */
    cv::Ptr<cv::cuda::BroxOpticalFlow> brox = cv::cuda::BroxOpticalFlow::create();
    cv::cuda::GpuMat im0 = toGPUMat(prhs[0]);
    cv::cuda::GpuMat im1 = toGPUMat(prhs[1]);
    cv::cuda::GpuMat outFlow(im0.size(), CV_32FC2);
    brox->calc(im0, im1, outFlow);
    cv::Mat result;
    outFlow.download(result);
    std::vector<cv::Mat> channels(2);
    cv::split(result, channels);

    /* Finally, output the m*n*2 flow field */
    cv::Size outSize = result.size();
    mwSize outDim[3] = {outSize.height, outSize.width, 2};
    mwSize rows = outDim[0], cols = outDim[1];
    plhs[0] = mxCreateNumericArray(3, outDim, mxSINGLE_CLASS, mxREAL);
    float *outMatrix = (float*)mxGetData(plhs[0]);

    for (int chan = 0; chan < outDim[2]; chan++) {
        for (int col = 0; col < outDim[1]; col++) {
            for (int row = 0; row < outDim[0]; row++) {
                int index = chan * cols * rows + col * rows + row;
                outMatrix[index] = channels[chan].at<float>(row, col);
            }
        }
    }
}
