
 /* =======================================================================
 * Copyright (c) 2010, Institute for Microelectronics, TU Vienna.
 * http://www.iue.tuwien.ac.at
 * -----------------
 * Matlab interface for
 * ViennaCL - The Vienna Computing Library
 * -----------------
 *
 * authors:    Karl Rupp                          rupp@iue.tuwien.ac.at
 * Florian Rudolf                     flo.rudy+viennacl@gmail.com
 * Josef Weinbub                      weinbub@iue.tuwien.ac.at
 *
 * license:    MIT (X11), see file LICENSE in the ViennaCL base directory
 *
 * file changelog: - June 10, 2010   New from scratch for first release
 * ======================================================================= */

#include <math.h>
#include <cmath>
#include <stdio.h>
#include "mex.h"
#include <time.h>
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/bicgstab.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/forwards.h"

#if !defined(MAX)
#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#endif

void viennacl_bicgstab(double	* result,   //output vector
        mwIndex	* cols,    //input vector holding column jumpers
        mwIndex	* rows,    //input vector holding row indices
        double *entries,
        double *rhs,
        mwIndex *rhs_rows,
        mwIndex *rhs_cols,
        mwSize     num_cols,
        mwSize     nnzmax,
        mwSize     k,
        mwIndex *pre_cols,
        mwIndex *pre_rows,
        double *pre,
        mwSize pre_n
        ) {

    viennacl::vector<double>             vcl_rhs(num_cols);
    viennacl::vector<double>               vcl_result(num_cols);
    viennacl::compressed_matrix<double>    vcl_matrix(num_cols, num_cols);
    viennacl::compressed_matrix<double>    vcl_pre1(num_cols, num_cols);

    std::vector< std::map< unsigned int, double > >  stl_matrix(num_cols);
    
    std::vector< std::map< unsigned int, double > >  stl_matrix_pre(num_cols);
    
    std::vector< std::map< unsigned int, double > >  stl_matrix_rhs(num_cols);
   

    float exec_time1=0;
    
    for (mwSize j=0; j<num_cols; ++j) {
        for (mwSize i = cols[j]; i<cols[j+1]; ++i)
            stl_matrix[rows[i]][j] = entries[i];
    }
    time_t timer1;
    time_t  start_time = clock();
    copy(stl_matrix, vcl_matrix);
    float exec_time = (float) (clock() - start_time) / CLOCKS_PER_SEC;  
     stl_matrix.clear(); //clean up this temporary storage
   
    //now copy matrix to GPU:
    for (mwSize j=0; j<pre_n; ++j) {
        for (mwSize i = pre_cols[j]; i<pre_cols[j+1]; ++i)
            stl_matrix_pre[pre_rows[i]][j] = pre[i];
    }
 
    //now copy matrix to GPU:
     start_time = clock();
     
    copy(stl_matrix_pre, vcl_pre1);
    
    exec_time = exec_time+((float) (clock() - start_time) / CLOCKS_PER_SEC); 

    stl_matrix_pre.clear();
    
    viennacl::linalg::ilut_precond< viennacl::compressed_matrix<double> > vcl_pre(vcl_pre1, viennacl::linalg::ilut_tag());
    //solve it:

    for(mwSize i=0;i<k;++i) {
         
        start_time = clock();
        copy(rhs + (i*num_cols) , rhs + ((i+1)*num_cols), vcl_rhs.begin());
        exec_time = exec_time+((float) (clock() - start_time) / CLOCKS_PER_SEC); 

        vcl_result = solve(vcl_matrix,
                vcl_rhs,
                viennacl::linalg::bicgstab_tag(1e-22, 300), vcl_pre);  //relative tolerance of 1e-15, at most 300 iterations
 
        timer1 = clock();
        copy(vcl_result.begin(), vcl_result.end(), result + i*num_cols  );
        exec_time1 = exec_time1+((float) (clock() -timer1 ) / CLOCKS_PER_SEC); 

    }
mexPrintf("-------------------CPU TO GPU . time: ------------------------\n %f \n",exec_time) ;
mexPrintf("-------------------GPU TO CPU . time: ------------------------\n %f \n",exec_time1) ;
    
    return;
}

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] )
        
{

    double *result;
    double *rhs;
    double *pre;
    mwSize m, n, k, nnzmax, pre_n;
    mwIndex * cols, *pre_cols, *pre_rows, *rhs_rows;
    mwIndex * rows, *rhs_cols;
    double * entries;
    
    
    /* Check for proper number of arguments */
    
    //mexPrintf("checking the number of arguments\n\n");
    
    if (nrhs != 3) {
        mexErrMsgTxt("Three input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Wrong number of output arguments.");
    }
    
    //mexPrintf("Before checking for sparse and dimension of Y\n\n");
    
    
    /* Check the dimensions of Y.  Y can be 4 X 1 or 1 X 4. */
    
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    k = mxGetN(prhs[1]);
    //mexPrintf("%d\n\n\n\n",k);
    pre_n = mxGetN(prhs[2]);
    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||  !mxIsSparse(prhs[0]) ) {
        mexErrMsgTxt("viennacl_bicgstab requires a double precision real sparse matrix.");
        return;
    }
    if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
        mexErrMsgTxt("viennacl_bicgstab requires a double precision real right hand side vector.");
        return;
    }
    

    
    /* Create a vector for the return argument (the solution :-) ) */
    plhs[0] = mxCreateDoubleMatrix(MAX(m, n), k, mxREAL); //return vector with 5 entries
    
    

    
    /* Assign pointers to the various parameters */
    result = mxGetPr(plhs[0]);

    
    
    cols    = mxGetJc(prhs[0]);
    rows    = mxGetIr(prhs[0]);
    entries = mxGetPr(prhs[0]);
    nnzmax  = mxGetNzmax(prhs[0]);
    rhs_cols    = mxGetJc(prhs[1]);
    rhs_rows    = mxGetIr(prhs[1]);
    rhs     = mxGetPr(prhs[1]);
    pre     = mxGetPr(prhs[2]);
    pre_cols    = mxGetJc(prhs[2]);
    pre_rows    = mxGetIr(prhs[2]);

    /* Do the actual computations in a subroutine */
    viennacl_bicgstab(result, cols, rows, entries, rhs, rhs_rows, rhs_cols, n, nnzmax, k, pre_cols, pre_rows, pre, pre_n);
    
 
    return;
    
}


