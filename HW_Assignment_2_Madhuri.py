
# coding: utf-8

# In[107]:



from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[152]:



import numpy as np


print("################################\n")
print("HW          : Assignment 2\n")
print("Student Name: Madhuri Kalani\n")
print("Date        : ", np.datetime64('2018-06-08'))
print("\n################################\n\n\n")

# Creating matrix A containing random numbers 
A=np.matrix(np.random.random(15))

# Matrix A with size (3,5) 
A = A.reshape(3,5)

print("Matrix A of random numbers and size (3,5):\n\n", A)


# The length of matrix A
print("\nThe length of matrix A : ",A.size)
# The size of matrix A
print("\nThe size of matrix A : ",A.nbytes)

# Slicing matrix A to size (3,4)
A=A[:3,:4]
print("\nMatrix A after slicing:\n\n",A)

# Transpose of matrix A and assigning it to B
B=A.T
print("\n\nMatrix B :\n\n",B)

# The minimum value in column 1 of matrix B
print("\nThe minimum value in column 1 (index 0) of matrix B : ",B[:,0].min())

# The minimum value for the entire matrix A
print("\n\nThe minimum value for the entire matrix A : ",A.min())
# The maximum value for the entire matrix A
print("\nThe maximum value for the entire matrix A : ",A.max())

# Vector X (an array) with 4 random numbers
X= np.array(np.random.random(4))
print("\n\nVector X (an array) with 4 random numbers:\n\n",X)


# Function definition with arguments vector X and matrix A 
def func_one(X,A):
    
    # Product of vector X with matrix A 
    D=np.dot(A,X) 
    return D
    
     

# Function Call
print("\n\nCalling function func_one and passing arguments vector X & matrix A :")
D = func_one(X,A)
print("\nStoring the product of vector X with matrix A as D :\n",D)
print("\nEnd of function func_one!\n\n")

# Complex number Z with absolute and real parts != 0
Z= np.complex(3,4) 
print("\nComplex number Z : ",Z)
    
# Real part  of Z
print("\nReal part  of Z : ", Z.real)
# Imaginary part of Z
print("\nImaginary part of Z : ",Z.imag)
# Absolute value of Z
print("\nAbsolute value of Z : ",np.abs(Z))
    
# Multiplying result and recording it to C
C=np.abs(Z)*D 
print("\nProduct of D with the absolute value of Z is C :\n\n ",C)

# Converting matrix B from a matrix to a string and overwrite B
B=np.str(B) 
print("\n\nB is of type : ",type(B))
print ("\nValue of string B : \n\n",B)


print("\n\nMadhuri Kalani is done with HW2\n\n") 

# End of HW2

