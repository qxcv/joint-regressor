#define INF 1E20
#include <math.h>
#include <sys/types.h>
#include "mex.h"

/*
 * shiftdt.cpp
 * Generalized distance transforms based on Felzenswalb and Huttenlocher.
 * This applies computes a min convolution of an arbitrary quadratic
 * function ax^2 + bx
 * This outputs results on an shifted, subsampled grid (useful for passing
 * messages between variables in different domains)
 */

static inline int square(int x) { return x*x; }

// dt1d(source,destination_val,destination_ptr,source_step,source_length,
//      a,b,dest_shift,dest_length,dest_step)
void dt1d(double *src, double *dst, int *ptr, int step, int len,
          double a, double b, double dshift, int dlen, double dstep) {
    /* Informal documentation: This function computes a single 1D distance
     * transform, with several parameters for controlling array access
     * strides (useful for processing matrices where you want to do several
     * 1D distance transforms) and steps for subsampled grids. Parameters:
     *
     * src: original function value at each location (for PSM, this will be
     *      the value of the unary).
     * dst: value of lower envelope at each location.
     * ptr: pointer to parabola which forms lower envelope at location
     * step: stride to use when accessing src.
     * len: number of cells in the 1D array you want to distance transform.
     * a, b: parameters for deformation quadratic ax^2 + bx
     **** WARNING: Next three definitions could be totally wrong. ****
     * dshift: initial shift for output grid. Useful with dstep.
     * dlen: size of destination array (might be bigger or smaller than
     *       source array, which allows for {super,sub}sampling)
     * dstep: lets you adapt for subsampling of output grid or something.
     *        Can't think of a practical goal this would achieve.
     */
    int   *v = new int[len];
    float *z = new float[len+1];
    int k = 0;
    int q = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    
    for (q = 1; q <= len-1; q++) {
        float s = ((src[q*step] - src[v[k]*step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
        while (s <= z[k]) {
            k--;
            s  = ((src[q*step] - src[v[k]*step]) - b*(q - v[k]) + a*(square(q) - square(v[k]))) / (2*a*(q-v[k]));
        }
        k++;
        v[k]   = q;
        z[k]   = s;
        z[k+1] = +INF;
    }
    
    k = 0;
    double dq = dshift;
    
    for (int i=0; i <= dlen-1; i++) {
        while (z[k+1] < dq)
            k++;
        dst[i*step] = a*square(dq-v[k]) + b*(dq-v[k]) + src[v[k]*step];
        ptr[i*step] = v[k];
        dq += dstep;
    }
    
    delete [] v;
    delete [] z;
}



// matlab entry point
// [M, Ix, Iy] = shiftdt(score, ax, bx, ay, by, offx, offy, lenx, leny, step)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 5)
        mexErrMsgTxt("Wrong number of inputs");
    if (nlhs != 3)
        mexErrMsgTxt("Wrong number of outputs");
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS)
        mexErrMsgTxt("Invalid unaries");
    if (mxGetClassID(prhs[1]) != mxDOUBLE_CLASS
            || mxGetNumberOfElements(prhs[1]) != 4)
        mexErrMsgTxt("Invalid Gaussian weights");
    if (mxGetClassID(prhs[2]) != mxDOUBLE_CLASS
            || mxGetNumberOfElements(prhs[2]) != 2)
        mexErrMsgTxt("Invalid offsets");
    if (mxGetClassID(prhs[3]) != mxINT32_CLASS
            || mxGetNumberOfElements(prhs[3]) != 2)
        mexErrMsgTxt("Invalid lengths");
    
    // Read in deformation coefficients, negating to define a cost
    // Read in offsets for output grid, fixing MATLAB 0-1 indexing
    double *vals = mxGetPr(prhs[0]);
    int sizx  = mxGetN(prhs[0]);
    int sizy  = mxGetM(prhs[0]);
    double *gauw = mxGetPr(prhs[1]);
    double ax = -gauw[0];
    double bx = -gauw[1];
    double ay = -gauw[2];
    double by = -gauw[3];
    double *offs = mxGetPr(prhs[2]);
    int *lens = (int *)mxGetData(prhs[3]);
    double offx  = offs[0] - 1;
    double offy  = offs[1] - 1;
    int lenx  = lens[0];
    int leny  = lens[1];
    double step = mxGetScalar(prhs[4]);
    
    mxArray  *mxM = mxCreateNumericMatrix(leny,lenx,mxDOUBLE_CLASS, mxREAL);
    mxArray *mxIy = mxCreateNumericMatrix(leny,lenx,mxINT32_CLASS, mxREAL);
    mxArray *mxIx = mxCreateNumericMatrix(leny,lenx,mxINT32_CLASS, mxREAL);
    double   *M = (double *)mxGetPr(mxM);
    int32_t *Iy = (int32_t *)mxGetPr(mxIy);
    int32_t *Ix = (int32_t *)mxGetPr(mxIx);
    
    double   *tmpM =  (double *)mxCalloc(leny*sizx, sizeof(double));
    int32_t *tmpIy = (int32_t *)mxCalloc(leny*sizx, sizeof(int32_t));
    
    //printf("(%d,%d),(%d,%d),(%d,%d)\n",offx,offy,lenx,leny,sizx,sizy);
    
    // dt1d(source,destination_val,destination_ptr,source_step,source_length,
    //      a,b,dest_shift,dest_length,dest_step)
    for (int x = 0; x < sizx; x++)
        dt1d(vals+x*sizy, tmpM+x*leny, tmpIy+x*leny, 1, sizy, ay, by, offy, leny, step);
    
    for (int y = 0; y < leny; y++)
        dt1d(tmpM+y, M+y, Ix+y, leny, sizx, ax, bx, offx, lenx, step);
    
    // get argmins and adjust for matlab indexing from 1
    for (int x = 0; x < lenx; x++) {
        for (int y = 0; y < leny; y++) {
            int p = x*leny+y;
            Iy[p] = tmpIy[Ix[p]*leny+y]+1;
            Ix[p] = Ix[p]+1;
        }
    }
    
    mxFree(tmpM);
    mxFree(tmpIy);
    plhs[0] = mxM;
    plhs[1] = mxIx;
    plhs[2] = mxIy;
    return;
}
