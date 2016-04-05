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
          double a, double b, double mean, int dlen) {
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
     * mean: mean displacement for limb in current dimension
     * dlen: size of destination array (might be bigger or smaller than
     *       source array, which allows for {super,sub}sampling)
     */
    // v[k] gives root of k-th parabola included in lower envelope
    // [z[k], z[k+1]] is the region in which parabola k of the lower
    // envelope actually "counts" towards the lower envelope
    int   *v = new int[len];
    double *z = new double[len+1];
    int k = 0;
    int q = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = +INF;
    
    for (q = 1; q <= len-1; q++) {
        // For all locations in the source grid (except the first)...
        auto intersect = [&src,&v,q,step,a,b,mean](int k) {return
            // Calculates where k-th parabola in the lower envelope
            // (located at v[k]) intersections with parabola at q
            ((src[q*step] - src[v[k]*step])
              - b*(q - v[k])
              + a*(square(q) - square(v[k])))
            / (2*a*(q-v[k])) + mean;
        };
        
        // while the intersection of the last (k-th) parabola in the lower
        // envelope with the parabola we are currently considering (at q)
        // is to the left of the last parabola's root (i.e. the current
        // parabola is strictly lower than the last parabola in the lower
        // envelope at all points in which the last parabola in the lower
        // envelope actually counts towards the lower envelope), we delete
        // the last parabola from the array of parabolas which define the
        // lower envelope (effectively replacing it with the current
        // parabola)
        double s = intersect(k);
        while (s <= z[k]) {
            k--;
            s  = intersect(k);
        }
        // Now append the current parabola to the lowest envelope. It's the
        // farthest to the right of any parabola we've considered so far,
        // so we know it will eventually form a part of the lower envelope
        // if we go right far enough. Of course, we might consider another
        // parabola later which removes the supersedes the one we're about
        // to add, but we'll remove the current parabola when/if that
        // happens.
        k++;
        v[k]   = q;
        z[k]   = s;
        z[k+1] = +INF;
    }
    
    k = 0;
    for (q=0; q <= dlen-1; q++) {
        // Now, for each output location q...
        while (z[k+1] < q)
            // skip forward to find the parabola which forms the lower
            // envelope at q (which will be rooted at v[k])
            k++;
        dst[q*step] = a*square(q - v[k] - mean) + b*(q - v[k] - mean) + src[v[k]*step];
        ptr[q*step] = v[k];
    }
    
    delete [] v;
    delete [] z;
}

// matlab entry point
// [M, Ix, Iy] = shiftdt(score, ax, bx, ay, by, offx, offy, lenx, leny, step)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 4)
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
        mexErrMsgTxt("Invalid means");
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
    double *means = mxGetPr(prhs[2]);
    double meanx  = means[0];
    double meany  = means[1];
    int *lens = (int *)mxGetData(prhs[3]);
    int lenx  = lens[0];
    int leny  = lens[1];
    
    mxArray  *mxM = mxCreateNumericMatrix(leny,lenx,mxDOUBLE_CLASS, mxREAL);
    mxArray *mxIy = mxCreateNumericMatrix(leny,lenx,mxINT32_CLASS, mxREAL);
    mxArray *mxIx = mxCreateNumericMatrix(leny,lenx,mxINT32_CLASS, mxREAL);
    double   *M = (double *)mxGetPr(mxM);
    int32_t *Iy = (int32_t *)mxGetPr(mxIy);
    int32_t *Ix = (int32_t *)mxGetPr(mxIx);
    
    double   *tmpM =  (double *)mxCalloc(leny*sizx, sizeof(double));
    int32_t *tmpIy = (int32_t *)mxCalloc(leny*sizx, sizeof(int32_t));
    
    // dt1d(source,destination_val,destination_ptr,source_step,source_length,
    //      a,b,mean_disp,dest_length)
    for (int x = 0; x < sizx; x++)
        dt1d(vals+x*sizy, tmpM+x*leny, tmpIy+x*leny, 1, sizy, ay, by, meany, leny);
    
    for (int y = 0; y < leny; y++)
        dt1d(tmpM+y, M+y, Ix+y, leny, sizx, ax, bx, meanx, lenx);
    
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
