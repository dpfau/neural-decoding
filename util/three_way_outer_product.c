#include <string.h>
#include <omp.h>
#include "mex.h"

/**
* three_way_outer_product.c - given three two-d arrays, take the outer
*   product along each row, and inner product along the columns.
*   Equivalent to tprod(tprod(x,[2 3],y,[1 3]),[1 3 -1],z,[2 -1]), but
*   doesn't require allocating a size(x,1),size(y,1),size(y,2) array, in
*   case size(y,2) is extremely large, as in our applications
* 
* David Pfau, 2012
*
* Input arguments:
* prhs[0], prhs[1], prhs[2] - three 2D arrays with the same number of columns
*
* Output arguments:
* plhs[0] - 3D array 
**/
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
	if( nrhs != 3 ) {
		mexErrMsgTxt("Incorrect number of input arguments!");
	}
	if( !mxIsDouble( prhs[0] ) || !mxIsDouble( prhs[1] ) || !mxIsDouble( prhs[2] ) ) {
		mexErrMsgTxt("Inputs must be matrices!");
	}
    if( mxGetN( prhs[0] ) != mxGetN( prhs[1] ) || mxGetN( prhs[0] ) != mxGetN( prhs[2] ) ) {
        mexErrMsgTxt("Input arguments must have same number of columns!");
    }

	mwSize m, n, p, q;
	m = mxGetM( prhs[0] );
	n = mxGetM( prhs[1] );
    p = mxGetM( prhs[2] );
    q = mxGetN( prhs[0] );
    
    mwSize dims[] = {m,n,p};
	plhs[0] = mxCreateNumericArray( 3, &dims[0], mxDOUBLE_CLASS, mxREAL );

	double *x = mxGetPr( prhs[0] );
    double *y = mxGetPr( prhs[1] );
    double *z = mxGetPr( prhs[2] );
	double *result = mxGetPr( plhs[0] );
    
	mwIndex i, j, k, l;
    #pragma omp parallel for
    for( i = 0; i < m; i++ ) {
        for( j = 0; j < n; j++ ) {
            for( k = 0; k < p; k++ ) {
                for( l = 0; l < q; l++ ) {
                    result[i + m*j + m*n*k] += x[i + m*l] * y[j + n*l] * z[k + p*l];
                }
            }
        }
    }
}