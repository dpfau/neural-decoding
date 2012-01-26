#include <string.h>
#include "mex.h"

/**
* column_squared_norm.c - for some reason, Matlab doesn't make it easy to
*   just compute in place the squared norm of each column of a matrix.
* 
* David Pfau, 2012
*
* Input arguments:
* prhs[0] - matrix for which we calculate the squared norm of each row
*
* Output arguments:
* plhs[0] - row vector of squared norms
**/
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
	if( nrhs != 1 ) {
		mexErrMsgTxt("Incorrect number of input arguments!");
	}
	if( !mxIsDouble( prhs[0] ) ) {
		mexErrMsgTxt("First input must be matrix!");
	}

	mwSize m, n;
	m = mxGetM( prhs[0] );
	n = mxGetN( prhs[0] );
    
	plhs[0] = mxCreateDoubleMatrix( 1, n, mxREAL );

	double *matrix = mxGetPr( prhs[0] );
	double *result = mxGetPr( plhs[0] );
    
	mwIndex i, j;
    for( i = 0; i < m; i++ ) {
        for( j = 0; j < n; j++ ) {
            result[j] += matrix[i + m*j]*matrix[i + m*j];
        }
    }
}