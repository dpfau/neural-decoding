#include <string.h>
#include "mex.h"

/**
* add_vector.c - subtract a vector from every column of a matrix.
*   hopefully faster than the matlab conventions:
*   x + y*ones(1,size(x,2)) or x + y(:,ones(1,size(x,2)))
* 
* David Pfau, 2012
*
* Input arguments:
* prhs[0] - matrix to which we add the vector repeatedly 
* prhs[1] - vector which we add to each column of the first input
*
* Output arguments:
* plhs[0] - matrix with all elements subtracted 
**/
void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) {
	if( nrhs != 2 ) {
		mexErrMsgTxt("Incorrect number of input arguments!");
	}
	if( !mxIsDouble( prhs[0] ) ) {
		mexErrMsgTxt("First input must be matrix!");
	}
	if( !mxIsDouble( prhs[1] ) ) {
		mexErrMsgTxt("Second input must be vector!");
	}

	mwSize m, n, x, y;
	m = mxGetM( prhs[0] );
	n = mxGetN( prhs[0] );
    x = mxGetM( prhs[1] );
    y = mxGetN( prhs[1] );
    
    if( m != x ) {
        mexErrMsgTxt("Column lengths must be equal!");
    }
    if( y != 1 ) {
        mexErrMsgTxt("Second input must be vector!");
    }
	plhs[0] = mxCreateDoubleMatrix( m, n, mxREAL );

	double *matrix = mxGetPr( prhs[0] );
	double *vector = mxGetPr( prhs[1] );
	double *result = mxGetPr( plhs[0] );
    
	mwIndex i, j;
    for( i = 0; i < m; i++ ) {
        for( j = 0; j < n; j++ ) {
            result[i + m*j] = matrix[i + m*j] + vector[i];
        }
    }
}