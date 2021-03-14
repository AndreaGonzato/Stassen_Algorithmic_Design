# 1
The implementation of the *strassen_matrix_mult* function can be found in the file [matrix.py](matrix.py)
# 2
The implementation of the general case can be found in the function: *matrix_mult_generalization*. Which can perform the multiplication of two matrices of any size (nxp)(pxm). 
The idea for this function is to up scale the dimensions of the couple of matrices to the closest power of 2. Then with these new matrix we can compute the Strassen algorithm and at the end down scale the matrix obtain after the multiplication.
