from __future__ import annotations

from numbers import Number
from typing import List, Tuple
from random import random, seed
from sys import stdout
from timeit import timeit



def next_power_of_2(n: int):
    '''
    determine the next power of two of a given number

    Parameters
    ----------
    n: an int number

    Returns
    -------
    int value
    '''
    count = 0

    # this condition is for the case when n is 0
    if (n and not (n & (n - 1))):
        return n

    while (n != 0):
        n >>= 1
        count += 1

    return 1 << count


def gauss_matrix_mult(A: Matrix, B: Matrix, check_null_matrix: bool = False) -> Matrix:
    """
    Multiply two matrices by using Gauss's algorithm

    Parameters
    ----------
    A: Matrix
        The first matrix to be multiplied
    B: Matrix
        The second matrix to be multiplied
    check_null_matrix: bool
        is a flag that define if it is necessary to manage differently the base case of null Matrix.

    Returns
    -------
    Matrix
        The row-column multiplication of the matrices passed as parameters

    Raises
    ------
    ValueError
        If the number of columns of `A` is different from the number of
        rows of `B`
    """
    # a precondition for this function is that B.num_of_rows = A.num_of_cols
    if(A.num_of_cols != B.num_of_rows):
        raise ValueError("The two matrix can not be multiplied")

    if check_null_matrix and (A.is_null() or B.is_null()):
            return Matrix([[0 for x in range(B.num_of_cols)] for y in range(A.num_of_rows)],
                          clone_matrix=False)

    result = [[0 for col in range(B.num_of_cols)]
              for row in range(A.num_of_rows)]

    for row in range(A.num_of_rows):
        for col in range(B.num_of_cols):
            value = 0
            for k in range(A.num_of_cols):
                value += A[row][k] * B[k][col]
            result[row][col] = value

    return Matrix(result, clone_matrix=False)

def get_matrix_quadrands(A: Matrix) -> Tuple[Matrix, Matrix, Matrix, Matrix]:

    half_num_of_row = A.num_of_rows//2
    if A.num_of_rows/2%1 > 0:
        half_num_of_row += 1
    half_num_of_cols = A.num_of_cols//2
    if A.num_of_cols/2%1 > 0:
        half_num_of_cols += 1
    A11 = A.submatrix(0, half_num_of_row, 0, half_num_of_cols)
    A12 = A.submatrix(0, half_num_of_row, A.num_of_cols - A.num_of_cols//2, A.num_of_cols//2)
    A21 = A.submatrix(half_num_of_row, A.num_of_rows - A.num_of_rows//2, 0, half_num_of_cols)
    A22 = A.submatrix(half_num_of_row, A.num_of_rows - A.num_of_rows//2, A.num_of_cols - A.num_of_cols//2, A.num_of_cols//2)
    return A11, A12, A21, A22


def matrix_mult_generalization(A: Matrix, B:Matrix, check_null_matrix: bool = True, improve_space_complexity: bool = True) -> Matrix:
    '''
    compute the matrix multiplication of two matrix

    Parameters
    ----------
    A: Matrix
        The first matrix used for the multiplication operation having dimension (nxp)
    B: Matrix
        The second matrix used for the multiplication operation having dimension (pxm)
    check_null_matrix: bool
        is a flag that define if it is necessary to manage differently the base case of null Matrix.
    improve_space_complexity: bool
        is a flag that define which Strassen function to call. (strassen_matrix_mult_improved VS strassen_matrix_mult).
         If it true it call strassen_matrix_mult_improved

    Returns
    -------
    Matrix
        The matrix corresponding to the multiplication of the two matrices

    Raises
    ------
    ValueError
        If the number of columns of `A` is different from the number of
        rows of `B`
    '''
    # a precondition for this function is that B.num_of_rows = A.num_of_cols
    if(A.num_of_cols != B.num_of_rows):
        raise ValueError("The two matrix can not be multiplied")

    size = max(A.num_of_rows, A.num_of_cols, B.num_of_cols)
    size = next_power_of_2(size)

    # build an up scaling matrix of A
    up_scaling_A = Matrix([[0 for x in range(size)] for y in range(size)],
                    clone_matrix=False)

    up_scaling_A.assign_submatrix(0, 0, A)
    up_scaling_A.assign_submatrix(0, A.num_of_cols, NullMatrix(A.num_of_rows, size - A.num_of_cols))
    up_scaling_A.assign_submatrix(A.num_of_rows, 0, NullMatrix(size - A.num_of_rows, A.num_of_cols))
    up_scaling_A.assign_submatrix(A.num_of_rows, A.num_of_cols, NullMatrix(size-A.num_of_rows, size-A.num_of_cols))

    # build an up scaling matrix of B
    up_scaling_B = Matrix([[0 for x in range(size)] for y in range(size)],
                    clone_matrix=False)

    up_scaling_B.assign_submatrix(0, 0, B)
    up_scaling_B.assign_submatrix(0, B.num_of_cols, NullMatrix(B.num_of_rows, size - B.num_of_cols))
    up_scaling_B.assign_submatrix(B.num_of_rows, B.num_of_cols, NullMatrix(size-B.num_of_rows, size-B.num_of_cols))
    up_scaling_A.assign_submatrix(B.num_of_rows, B.num_of_cols, NullMatrix(size - B.num_of_rows, size - B.num_of_cols))

    # compute the multiplication
    if improve_space_complexity:
        result = strassen_matrix_mult_improved(up_scaling_A, up_scaling_B, check_null_matrix=check_null_matrix)
    else:
        result = strassen_matrix_mult(up_scaling_A, up_scaling_B, check_null_matrix=check_null_matrix)

    # return the down scaled matrix
    return result.submatrix(0, A.num_of_rows, 0, B.num_of_cols)

def strassen_matrix_mult(A: Matrix, B:Matrix, check_null_matrix: bool = False) -> Matrix:
    '''
    compute the matrix multiplication of two square matrices

    Parameters
    ----------
    A: Matrix
        The first matrix used for the multiplication operation having dimension (nxn) where n = 2^i
    B: Matrix
        The second matrix used for the multiplication operation having dimension (nxn) where n = 2^i
    check_null_matrix: bool
        is a flag that define if it is necessary to manage differently the base case of null Matrix.

    Returns
    -------
    Matrix
        The matrix corresponding to the multiplication of the two matrices. Its size is (nxn)

    Raises
    ------
    ValueError
        If the number of columns of `A` is different from the number of
        rows of `B`
    '''
    # a precondition for this function is that B.num_of_rows = A.num_of_cols
    if(A.num_of_cols != B.num_of_rows):
        raise ValueError("The two matrix can not be multiplied")

    # this is an improvement of the original implementation to try to reduce the number of auxiliary matrices
    # (code for the question 3)
    if check_null_matrix and (A.is_null() or B.is_null()):
            return Matrix([[0 for x in range(B.num_of_cols)] for y in range(A.num_of_rows)],
                          clone_matrix=False)

    # Base case
    if max(A.num_of_rows, B.num_of_cols, A.num_of_cols) < 32:
        return gauss_matrix_mult(A, B)

    A11, A12, A21, A22 = get_matrix_quadrands(A)
    B11, B12, B21, B22 = get_matrix_quadrands(B)

    # sum Theta(n^2)
    S1 = B12 - B22
    S2 = A11 + A12
    S3 = A21 + A22
    S4 = B21 - B11
    S5 = A11 + A22
    S6 = B11 + B22
    S7 = A12 - A22
    S8 = B21 + B22
    S9 = A11 - A21
    S10 = B11 + B12

    # recursion calls
    P1 = strassen_matrix_mult(A11, S1, check_null_matrix=check_null_matrix)
    P2 = strassen_matrix_mult(S2, B22, check_null_matrix=check_null_matrix)
    P3 = strassen_matrix_mult(S3, B11, check_null_matrix=check_null_matrix)
    P4 = strassen_matrix_mult(A22, S4, check_null_matrix=check_null_matrix)
    P5 = strassen_matrix_mult(S5, S6, check_null_matrix=check_null_matrix)
    P6 = strassen_matrix_mult(S7, S8, check_null_matrix=check_null_matrix)
    P7 = strassen_matrix_mult(S9, S10, check_null_matrix=check_null_matrix)

    # second batch of sums Theta(n^2)
    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7

    # build the resulting matrix
    result = Matrix([[0 for x in range(B.num_of_cols)] for y in range(A.num_of_rows)],
                    clone_matrix=False)

    # copying Cij into the resulting matrix
    result.assign_submatrix(0, 0, C11)
    result.assign_submatrix(0, result.num_of_cols//2, C12)
    result.assign_submatrix(result.num_of_rows//2, 0, C21)
    result.assign_submatrix(result.num_of_rows//2, result.num_of_cols//2, C22)

    return result


def strassen_matrix_mult_improved(A: Matrix, B:Matrix, check_null_matrix: bool = False) -> Matrix:
    '''
    Compute the matrix multiplication of two square matrices.
    This function is better than strassen_matrix_mult() regarding the space complexity

    Parameters
    ----------
    A: Matrix
        The first matrix used for the multiplication operation having dimension (nxn) where n = 2^i
    B: Matrix
        The second matrix used for the multiplication operation having dimension (nxn) where n = 2^i
    check_null_matrix: bool
        is a flag that define if it is necessary to manage differently the base case of null Matrix.

    Returns
    -------
    Matrix
        The matrix corresponding to the multiplication of the two matrices. Its size is (nxn)

    Raises
    ------
    ValueError
        If the number of columns of `A` is different from the number of
        rows of `B`
    '''
    # a precondition for this function is that B.num_of_rows = A.num_of_cols
    if(A.num_of_cols != B.num_of_rows):
        raise ValueError("The two matrix can not be multiplied")

    # this is an improvement of the original implementation to try to reduce the number of auxiliary matrices
    # (code for the question 3)
    if check_null_matrix and (A.is_null() or B.is_null()):
            return Matrix([[0 for x in range(B.num_of_cols)] for y in range(A.num_of_rows)],
                          clone_matrix=False)


    # Base case
    if max(A.num_of_rows, B.num_of_cols, A.num_of_cols) < 32:
        return gauss_matrix_mult(A,B)

    A11, A12, A21, A22 = get_matrix_quadrands(A)
    B11, B12, B21, B22 = get_matrix_quadrands(B)

    # build the resulting matrix
    result = Matrix([[0 for x in range(B.num_of_cols)] for y in range(A.num_of_rows)],
                    clone_matrix=False)


    # recursion calls
    S = strassen_matrix_mult_improved(A11, B12 - B22, check_null_matrix=check_null_matrix)
    C12 = S
    C22 = S
    S = strassen_matrix_mult_improved(A21 + A22, B11, check_null_matrix=check_null_matrix)
    C21 = S
    C22 = C22 - S
    S = strassen_matrix_mult_improved(A22, B21 - B11, check_null_matrix=check_null_matrix)
    C11 = S
    C21 = C21 + S
    S = strassen_matrix_mult_improved(A11 + A22, B11 + B22, check_null_matrix=check_null_matrix)
    C11 = C11 + S
    C22 = C22 + S
    S = strassen_matrix_mult_improved(A12 - A22, B21 + B22, check_null_matrix=check_null_matrix)
    C11 = C11 + S
    S = strassen_matrix_mult_improved(A11 - A21, B11 + B12, check_null_matrix=check_null_matrix)
    C22 = C22 - S
    # S2
    S = strassen_matrix_mult_improved(A11 + A12, B22, check_null_matrix=check_null_matrix)
    C12 = C12 + S
    C11 = C11 - S

    # copying Cij into the resulting matrix
    result.assign_submatrix(0, 0, C11)
    result.assign_submatrix(0, result.num_of_cols//2, C12)
    result.assign_submatrix(result.num_of_rows//2, 0, C21)
    result.assign_submatrix(result.num_of_rows//2, result.num_of_cols//2, C22)

    return result



class Matrix(object):
    '''
    A simple naive matrix class

    Members
    -------
    _A: List[List[Number]]
        The list of rows that store all the matrix values

    Parameters
    ----------
    A: List[List[Number]]
        The list of rows that store all the matrix values
    clone_matrix: Optional[bool]
        A flag to require a full copy of `A`'s data structure.

    Raises
    ------
    ValueError
        If there are two lists having a different number of values
    '''
    def __init__(self, A: List[List[Number]], clone_matrix: bool = True):
        num_of_cols = None

        for i, row in enumerate(A):
            if num_of_cols is not None:
                if num_of_cols != len(row):
                    raise ValueError('This is not a matrix')
            else:
                num_of_cols = len(row)

        if clone_matrix:
            self._A = [[value for value in row] for row in A]
        else:
            self._A = A

    @property
    def num_of_rows(self) -> int:
        return len(self._A)

    @property
    def num_of_cols(self) -> int:
        if len(self._A) == 0:
            return 0

        return len(self._A[0])

    def copy(self):
        A = [[value for value in row] for row in self._A]

        return Matrix(A, clone_matrix=False)

    def __getitem__(self, y: int):
        ''' Return one of the rows

        Parameters
        ----------
        y: int
            the index of the rows to be returned

        Returns
        -------
        List[Number]
            The `y`-th row of the matrix
        '''
        return self._A[y]

    def __iadd__(self, A: Matrix) -> Matrix:
        ''' Sum a matrix to this matrix and update it

        Parameters
        ----------
        A: Matrix
            The matrix to be summed up

        Returns
        -------
        Matrix
            The matrix corresponding to the sum between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''

        if (self.num_of_cols != A.num_of_cols or
                self.num_of_rows != A.num_of_rows):
            raise ValueError('The two matrices have different sizes')

        for y in range(self.num_of_rows):
            for x in range(self.num_of_cols):
                self[y][x] += A[y][x]

        return self

    def __add__(self, A: Matrix) -> Matrix:
        ''' Sum a matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix to be summed up

        Returns
        -------
        Matrix
            The matrix corresponding to the sum between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''
        res = self.copy()

        res += A

        return res

    def __isub__(self, A: Matrix) -> Matrix:
        ''' Subtract a matrix to this matrix and update it

        Parameters
        ----------
        A: Matrix
            The matrix to be subtracted up

        Returns
        -------
        Matrix
            The matrix corresponding to the subtraction between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''

        if (self.num_of_cols != A.num_of_cols or
                self.num_of_rows != A.num_of_rows):
            raise ValueError('The two matrices have different sizes')

        for y in range(self.num_of_rows):
            for x in range(self.num_of_cols):
                self[y][x] -= A[y][x]

        return self

    def __sub__(self, A: Matrix) -> Matrix:
        ''' Subtract a matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix to be subtracted up

        Returns
        -------
        Matrix
            The matrix corresponding to the subtraction between this matrix and
            that passed as parameter

        Raises
        ------
        ValueError
            If the two matrices have different sizes
        '''
        res = self.copy()

        res -= A

        return res

    def __mul__(self, A: Matrix) -> Matrix:
        """ Multiply one matrix to this matrix

        Parameters
        ----------
        A: Matrix
            The matrix which multiplies this matrix

        Returns
        -------
        Matrix
            The row-column multiplication between this matrix and that passed
            as parameter

        Raises
        ------
        ValueError
            If the number of columns of this matrix is different from the
            number of rows of `A`
        """
        return gauss_matrix_mult(self, A)

    def __rmul__(self, value: Number) -> Matrix:
        ''' Multiply one matrix by a numeric value

        Parameters
        ----------
        value: Number
            The numeric value which multiplies this matrix

        Returns
        -------
        Matrix
            The multiplication between `value` and this matrix

        Raises
        ------
        ValueError
            If `value` is not a number
        '''

        if not isinstance(value, Number):
            raise ValueError('{} is not a number'.format(value))

        return Matrix([[value*elem for elem in row] for row in self._A],
                      clone_matrix=False)

    def submatrix(self, from_row: int, num_of_rows: int,
                  from_col: int, num_of_cols: int) -> Matrix:
        """
        Return a sub-matrix of this matrix

        Parameters
        ----------
        from_row: int
            The first row to be included in the submatrix to be returned
        num_of_rows: int
            The number of rows to be included in the submatrix to be returned
        from_col: int
            The first col to be included in the submatrix to be returned
        num_of_cols: int
            The number of cols to be included in the submatrix to be returned

        Returns
        -------
        Matrix
            A submatrix of this matrix
        """
        A = [row[from_col:from_col+num_of_cols]
             for row in self._A[from_row:from_row+num_of_rows]]

        return Matrix(A, clone_matrix=False)

    def assign_submatrix(self, from_row: int, from_col: int, A: Matrix):
        for y, row in enumerate(A):
            self_row = self[y + from_row]
            for x, value in enumerate(row):
                self_row[x + from_col] = value

    def __repr__(self):
        return '\n'.join('{}'.format(row) for row in self._A)

    def absolute_sum_of_elements(self) -> int:
        """
        Compute the absolute sum of all the elements of the matrix

        Returns
        -------
        int: a value corresponding to the absolute sum

        """
        absolute_sum = abs(self._A[0][0])
        for y in range(0, self.num_of_rows):
            for x in range(0, self.num_of_cols):
                absolute_sum += abs(self._A[y][x])
        return absolute_sum

    def is_null(self) -> bool:
        """Check if the matrix is a null Matrix

        Returns
        -------
        bool: that indicate if a Matrix is a null Matrix or not. A Null Matrix is a matrix where all elements are zeros
        """
        for y in range(0, self.num_of_rows):
            for x in range(0, self.num_of_cols):
                if self._A[y][x] != 0:
                    return False
        return True


class IdentityMatrix(Matrix):
    """
    A class for identity matrices

    Parameters
    ----------
    size: int
        The size of the identity matrix
    """
    def __init__(self, size: int):
        A = [[1 if x == y else 0 for x in range(size)]
             for y in range(size)]

        super().__init__(A)


class NullMatrix(Matrix):
    '''
    A class for null matrices

    Parameters
    ----------
    size_row: int
        The number of row of the null matrix
    size_col: int
        The number of columns of the null matrix
    '''
    def __init__(self, size_row: int, size_col: int):
        A = [[ 0 for x in range(size_col)]
             for y in range(size_row)]

        super().__init__(A)



if __name__ == '__main__':

    seed(0)


    '''
    This code is a test to evaluate the effect of the improvement in the reduction of the number of auxiliary matrices.
    '''
    stdout.write('   n    p    m |  standard  |lim. space|null check|both improvements\n')
    for i in range(2, 9):
        m = 2 ** i + 1
        n = m // 2 + 1
        p = m // 3
        stdout.write(str(n).rjust(4))
        stdout.write(str(p).rjust(5))
        stdout.write(str(m).rjust(5))
        A = Matrix([[random() for x in range(p)] for y in range(n)])
        B = Matrix([[random() for x in range(m)] for y in range(p)])
        for values in [(False, False), (False, True), (True, False), (True, True)]:
            T = timeit(f'{"matrix_mult_generalization"}(A,B,'+str(values[0])+','+str(values[1])+')', globals=locals(), number=1)
            stdout.write('{:.3f}'.format(T).rjust(11))
            stdout.flush()
        stdout.write('\n')

    '''
    This code give a concrete comparison of time effort of the two different program gauss_matrix_mult VS. strassen_matrix_mult
    '''
    stdout.write('n\tgauss\tstrassen\n')
    for i in range(2, 10):
        size = 2 ** i
        stdout.write(f'{size}')
        A = Matrix([[random() for x in range(size)] for y in range(size)])
        B = Matrix([[random() for x in range(size)] for y in range(size)])

        for funct in ['gauss_matrix_mult', 'strassen_matrix_mult']:
            T = timeit(f'{funct}(A,B)', globals=locals(), number=1)
            stdout.write('\t{:.3f}'.format(T))
            stdout.flush()
        stdout.write('\n')


    '''
    Follow a demo of how to make a matrix multiplication of A*B where A is a matrix (nxp) and B is a matrix (pxm)
    '''
    stdout.write('----------------------\n')
    n = 63
    p = 109
    m = 81
    A = Matrix([[random() for x in range(p)] for y in range(n)])
    B = Matrix([[random() for x in range(m)] for y in range(p)])
    R = matrix_mult_generalization(A, B)
    # error due to approximation
    error = gauss_matrix_mult(A, B) - R
    print("Sum of the absolute error: ", error.absolute_sum_of_elements())





