# 1
The implementation of the *strassen_matrix_mult* function can be found in the file [matrix.py](matrix.py)
# 2
The implementation of the general case can be found in the function *matrix_mult_generalization*. Which can perform the multiplication of two matrices of any size (nxp)(pxm). 

The idea for this function is to up scale the dimensions of the couple of matrices to the closest power of 2. To fills ehach of those bigger matrices we put the original matrix in the top left corner and fills the rest of the free space whit zeros. Then with these new matrices we can compute the Strassen algorithm and at the end we just need to down scale the matrix obtain after the multiplication.

Follow the demonstration of the time complexity of the algorithm:
