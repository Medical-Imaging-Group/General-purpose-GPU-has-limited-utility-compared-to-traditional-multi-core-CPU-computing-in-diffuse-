# General purpose GPU has limited utility compared to traditional multi core CPU computing in diffuse 

# Installing CULA (txt|pdf) -for using mex files

Matlab Executables:

JTJ Routine (Update Equation 3): calc_update_gpu_JTJ.mexa64

JJT Routine (Update Equation 4): calc_update_gpu_JJT.mexa64

# Installing ViennaCL (txt|pdf) -for using mex files

Matlab Executable:

Sparse Finite Element Solver: viennacl_bicgstab_precon.mexa64

# Creation of MEX files using CULA functions (txt|pdf)

Example: Mex wrapper for culasv code: culasv.c

Mex wrapper for ViennaCL code: viennacl_bicgstab_precon.cpp

# Matlab Code for Image reconstruction (requires NIRFAST)

On GPU:

Jacobian computation: get_field_GPU_inv.m

Reconstruction computation: reconstruction_stnd_GPU_inv.m

Note: These instructions are for linux operating system only.

This Matlab code is used as part of the work presented in:

J. Prakash, V. Desai, S. Srinivasan, and P. K. Yalavarthy, "Multi-core computers have high scalability than graphics processing units for diffuse optical tomographic image reconstruction," SPIE/OSA European Conference on Biomedical Optics (ECBO-2013), May 12-16, 2013, Munich, Germany

**Created on**: October 13, 2013.
