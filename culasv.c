
 
#include <stdlib.h>
#include <stdio.h>

#include "culapack.h"
#include "mex.h"
//#include <cula.h>
//#include <culastatus.h>
//#include <culatypes.h>


void checkStatus(culaStatus status)
{
    if(!status)
        return;

    if(status == culaArgumentError)
        mexPrintf("Invalid value for parameter %d\n", culaGetErrorInfo());
    else if(status == culaRuntimeError)
        mexPrintf("Runtime error (%d)\n", culaGetErrorInfo());
    else
        mexPrintf("%s\n", culaGetStatusString(status));

    culaShutdown();
	mexErrMsgTxt("CULA error!");
}


void mexFunction(int			nlhs, 		/* number of expected outputs */
				 mxArray		*plhs[],	/* mxArray output pointer array */
				 int			nrhs, 		/* number of inputs */
				 const mxArray	*prhs[]		/* mxArray input pointer array */)
{
    
    
    float *A,*B,*X;
    int *IPIV;
    int M,N;
    //int info;
          culaStatus status;
    int NRHS;
    if (nrhs != 2) {
        mexErrMsgTxt("Need two input argument");
    }
	if (nlhs != 1) {
		mexErrMsgTxt("Only single argument allowed.");
	}

    
    if (mxGetClassID(prhs[0]) != mxSINGLE_CLASS) {
		mexErrMsgTxt("Only single precision allowed");
	}
	if (mxGetNumberOfDimensions(prhs[0]) != 2) {
		mexErrMsgTxt("2D matrix required");
	}

    
    M = mxGetM(prhs[0]);
	N = mxGetN(prhs[0]);
    NRHS=mxGetN(prhs[1]);
    
    
    A = (float*)mxGetPr(prhs[0]);
    B = (float*)mxGetPr(prhs[1]);
    
    
    
    plhs[0] = mxDuplicateArray(prhs[1]);
	X = (float*)mxGetPr(plhs[0]);
   // info = (int*)mxGetPr(plhs[1]);
    IPIV = (int*)malloc(N*sizeof(int));
    
    status = culaInitialize();
    checkStatus(status);



    status = culaSgesv(N, NRHS, A, N, IPIV, X, N);
    
	checkStatus(status);

culaShutdown();

}
    
