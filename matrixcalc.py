import numpy as np
import copy

class Matrix(object):
    def __init__(self, ls):
        self.matrix = ls
        if self.matrix == []:
            raise ValueError
        self.n_rows = len(self.matrix)
        if self.n_rows == 1:
            if type(self.matrix[0]) == int or type(self.matrix[0]) == float:
                self.n_col = 1
            elif type(self.matrix[0]) == list:
                count = 0
                for value in self.matrix[0]:
                    count += 1
                self.n_col = count
        else:
            self.n_col = len(self.matrix[0])
            for row in self.matrix[1:]:
                if len(row) != self.n_col:
                    raise ValueError
                
    def __eq__(self,other):
        if isinstance(other,Matrix):
            return (self.matrix == other.matrix)
        else:
            raise TypeError
    
    def issquare(self): #good
        if self.n_rows == self.n_col:
            return True
        else:
            return False
    
    def rows(self):
        """Returns number of rows in the Matrix """
        return len(self.matrix)
    
    def columns(self): 
        """Returns number of columns in the Matrix """
        return len(self.matrix[0])

    def duplicate(self): #good 
        """Returns a duplicated version of the given matrix"""
        A = self.matrix.copy()
        return Matrix(A)
                    
    def transpose(self): #good
        """ Transposes the Matrix that the user provides"""
        return Matrix([list(row) for row in zip(*self.matrix)])

    def __add__(self,*args): #good
        """ Adds however many other that the user provides """
        for other in args:
            if not isinstance(other, Matrix):
                raise TypeError("the object is not a matrix") 
            else:
                if (self.n_col != other.n_col) or (self.n_rows != other.n_rows):
                    raise ValueError("The size of the other are not the same")
                else:
                    new_matrix = []
                    for i in range(self.n_rows):
                        new_matrix.append([])
                    for row in new_matrix:
                        for amount_row in range(other. n_col):
                            row.append(0)
                    for i in range(self.n_rows):
                        for j in range(self.n_col):
                            new_matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
                    return Matrix(new_matrix)

    def __sub__(self,*args): #good
        """ Subtracts however many other that the user providew"""
        for other in args:
            if not isinstance(other, Matrix):
                raise TypeError("the object is not a matrix") 
            else:
                if (self.n_col != other.n_col) or (self.n_rows != other.n_rows):
                    raise ValueError("The size of the other are not the same")
                else:
                    new_matrix = []
                    for i in range(self.n_rows):
                        new_matrix.append([])
                    for row in new_matrix:
                        for amount_row in range(other. n_col):
                            row.append(0)
                    for i in range(self.n_rows):
                        for j in range(self.n_col):
                            new_matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
                    return Matrix(new_matrix)

    def __mul__(self, *args): #good
        """Returns the product of infinite matrices"""
        for other in args:            
            if isinstance(other, int or float):
                new_matrix = self.duplicate()
                for other in args:
                    for row in new_matrix:
                        for value in row:
                            value *= (other) #multiply by inverse
                    return Matrix(new_matrix)
            else:
                    if self.n_col != other.n_rows: 
                        raise ValueError ("The matrices do not follow the right format for multiplication: mxn x nxm")
                    else :
                        if not isinstance(other, Matrix): # If there are values that are not numbers
                            raise ValueError("Input Invalid: All values must be numbers")
                        else:
                            new_matrix = []
                            for i in range(self.n_rows):
                                new_matrix.append([])
                            for row in new_matrix:
                                for amount_row in range(other.n_col):
                                    row.append(0)
                            for column in range(self.n_rows): # in column of the self matrix
                                for row in range(other.n_col): # in the row of the other matrix
                                    for rows in range(other.n_rows): #rows of other matrix
                                        new_matrix[column][row] += self.matrix[column][rows] * other.matrix[rows][row] # multiply matrice
                            return Matrix(new_matrix)
                            
    def __truediv__(self, *args): #done
        new_matrix = self.duplicate().matrix
        for other in args:
            if not isinstance(other, int or float): #if it's not a number
                raise TypeError("the object is not a number")
            else:
                for row in range(self.n_rows):
                    for value in range(self.n_col):
                        new_matrix[row][value] *= 1/other #multiply by inverse
                return Matrix(new_matrix)

    def __str__(self): 
      if len(self.matrix) == 1:
        if type(self.matrix[0]) == int or type(self.matrix[0]) == float:
          print(float(self.matrix[0]))
        elif type(self.matrix[0]) == list:
          result = ""
          for value in self.matrix[0]:
              result += '{:10.2f}'.format(float(value))
          print(result)       
      else:
        for row in self.matrix: 
          result = ''
          for value in row:
              result += '{:10.2f}'.format(float(value))
          print(result)
      return str("")
    
    def identity(self): #done
        """Returns the identity matrix associated with the given Matrix"""
        if self.issquare():
            I = []
            for number in range(self.n_rows):
                I.append([])
            for row in I:
                for amount_row in range(self.n_col):
                    row.append(0)
            for i in range(len(self.matrix)):
                I[i][i] += 1
            return Matrix(I)
            
    def rank(self): #done
        """Returns the Rank of the given matrix"""
        rank_matrix = Matrix(self.ref().tolist())
        if rank_matrix.issquare():
            if rank_matrix == rank_matrix.identity():
                return len(self.matrix) # leading one on each row
            else:
                count = 0
                for i in range(len(self.matrix)):
                    rank_matrix.matrix[i][i] == 1
                    count += 1
        else:
            count = 0
            for i in range(len(self.matrix)):
                rank_matrix.matrix[i][i] == 1
                count += 1
                if IndexError:
                    print("It is impossible to find the Rank of this matrix")
                    break
        return count

    def ref(self):
        matr = np.array(self.matrix, dtype=np.float64)
        r = 0 
        c = 0 
        for k in range(len(self.matrix)):
            while all(matr.transpose()[c] == 0.0):
                c += 1
                if c == len(matr[0]) - 1 : 
                    break
            if matr[r][c] == 0:
                c_ = r
                while matr[c_][c] == 0: 
                    if c_ == len(matr) - 1: 
                        break
                    r += 1
                matr[[r, c_]] = matr[[c_, r]]
            if matr[r][c] != 0:
                matr[r] = matr[r] / matr[r][c]
            else: 
                continue
            for c_ in range(len(matr)):
                if c_ != r:
                    if matr[r][c] != 0:
                        matr[c_] = matr[c_] - matr[r] * matr[c_][c] / matr[r][c]
                    else:
                        continue
            if (r == len(matr) - 1) or (c == len(matr[0]) - 1): 
                break
            r += 1
            c += 1
            return matr

    def invert_matrix(self):
        '''Returns the inverse of the passed in matrix.'''
        # Make sure matrix can be inverted.
        if not self.issquare():
            raise ValueError('The matrix is not square')
        n = self.n_rows #because it's square, n_col == n_rows
        copy_A = copy.deepcopy(self.matrix)
        copy_I = copy.deepcopy(self.identity().matrix)

        try:
            for diagonal in range(n): # identity diagonal places that are present in identity matrix
                diagonal_scalar = 1 / copy_A[diagonal][diagonal]
                for j in range(n): # To make all the diagonal places = 1
                    copy_A[diagonal][j] *= diagonal_scalar
                    copy_I[diagonal][j] *= diagonal_scalar
                indices = list(range(n))
                for i in indices[0:diagonal] + indices[diagonal+1:]: #for all rows except the one with diagonal that is a part of the identity matrix
                    row_scaler = copy_A[i][diagonal]
                    for j in range(n): 
                        copy_A[i][j] = copy_A[i][j] - row_scaler * copy_A[diagonal][j] #Makes every non-identity diagonal part = 0
                        copy_I[i][j] = copy_I[i][j] - row_scaler * copy_I[diagonal][j]
            if copy_A == Matrix(copy_I).identity().matrix: #If it worked, return the inverse
                return Matrix(copy_I)
            else:
                raise TypeError("Matrix is singular")
        except ZeroDivisionError:
            raise ZeroDivisionError("Matrix is singular. Two identical rows/row of zeros")

    def nullity(self):
        return (self.n_col - self.rank())
        
    def size(self):
        a = "The Matrix is a " + str(len(self.matrix)) + "x" + str(len(self.matrix[0]))
        return(a)

    def info(self): 
        """Returns size, if identity, rank(num of leading ones), nullity(num of free variables"""
        print(self.size())
        print("This is the identity: \n", Matrix.identity(self))
        print("This is the rank:", Matrix.rank(self))
        print("This is the nullity:", Matrix.nullity(self))

