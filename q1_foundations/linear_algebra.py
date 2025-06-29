import numpy as np
import matplotlib.pyplot as plt

a = np.arange(15).reshape(3, 5)
b = np.array([6, 7, 8])
# print(a)
# print(type(a))
# print(a.shape)
# print(a.dtype)

# print(b)
# print(type(b))
# print(b.shape)
# print(b.dtype)

# Practice vector addition, subtraction, and scalar multiplication.
# Create some sample vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition
# print("\nVector addition:")
# print(f"v1 + v2 = {v1 + v2}")

# Vector subtraction 
# print("\nVector subtraction:")
# print(f"v1 - v2 = {v1 - v2}")

# Scalar multiplication
scalar = 2
# print("\nScalar multiplication:")
# print(f"{scalar} * v1 = {scalar * v1}")

def manual_dot_product(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Vectors must be of the same length")
    return sum(x * y for x, y in zip(v1, v2))

def vector_dot_product(v1, v2):
    return v1 @ v2

# Write a function that takes a matrix (list of lists) and returns its transpose.
def manual_matrix_transpose(matrix):
    return np.array([[row[i] for row in matrix] for i in range(len(matrix[0]))])

def matrix_transpose(matrix):
    return matrix.T

# print("\nManual dot product:")
# print(manual_dot_product(v1, v2))
# print("\nVector dot product:")
# print(vector_dot_product(v1, v2))
# print("\nMatrix transpose:")
# print(manual_matrix_transpose(a))
# print(matrix_transpose(a))


# Write a function to calculate the determinant of a 2x2 and a 3x3 matrix from scratch
def manual_determinant_2x2(matrix):
    if matrix.shape != (2, 2):
        raise ValueError("Matrix must be a 2x2")
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

def determinant_2x2(matrix):
    return np.linalg.det(matrix)

def manual_determinant_3x3(matrix):
    if matrix.shape != (3, 3):
        raise ValueError("Matrix must be a 3x3")
    return (
        matrix[0, 0] * matrix[1, 1] * matrix[2, 2] +
        matrix[0, 1] * matrix[1, 2] * matrix[2, 0] +
        matrix[0, 2] * matrix[1, 0] * matrix[2, 1] -
        matrix[0, 2] * matrix[1, 1] * matrix[2, 0] -
        matrix[0, 1] * matrix[1, 0] * matrix[2, 2] -
        matrix[0, 0] * matrix[1, 2] * matrix[2, 1]
    )

def determinant_3x3(matrix):
    return np.linalg.det(matrix)

a2x2 = np.array([[1, 2], [3, 4]])
print("\n2x2 matrix:")
print(a2x2)
a3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n3x3 matrix:")
print(a3x3)
print("\nManual determinant 2x2:")
print(manual_determinant_2x2(a2x2))
print(determinant_2x2(a2x2))
print("\nManual determinant 3x3:")
print(manual_determinant_3x3(a3x3))
print(determinant_3x3(a3x3))


# Write a function to calculate the inverse of a 2x2 and a 3x3 matrix from scratch
def manual_inverse_2x2(matrix):
    if matrix.shape != (2, 2):
        raise ValueError("Matrix must be a 2x2")
    determinant = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    if determinant == 0:
        raise ValueError("Matrix is singular, cannot be inverted")
    return np.array([[matrix[1, 1] / determinant, -matrix[0, 1] / determinant], [-matrix[1, 0] / determinant, matrix[0, 0] / determinant]])

def inverse_2x2(matrix):
    return np.linalg.inv(matrix)

def inverse_3x3(matrix):
    return np.linalg.inv(matrix)

print("\nManual inverse 2x2:")
print(manual_inverse_2x2(a2x2))
print(inverse_2x2(a2x2))


# a singular (non-invertible) matrix
a_singular = np.array([[1, 2], [2, 4]])
print("\nSingular matrix:")
print(a_singular)
#print("\nManual inverse 2x2:")
#print(manual_inverse_2x2(a_singular))
print("\nDeterminant of singular matrix:")
print(determinant_2x2(a_singular))
print("\nInverse of singular matrix:")
#print(inverse_2x2(a_singular))

# column space
print("\nColumn space of a_singular:")
print(np.linalg.matrix_rank(a_singular))

# Explore the concepts of column space and null space using NumPy. Can you find a vector `v` that is in the column space of a matrix `A`? (Hint: `Ax=v` has a solution). Can you find a vector `x` (other than zero) that is in the null space of `A`? (Hint: `Ax=0`)
# Find a vector v in the column space of A
# Any vector v that can be written as a linear combination of A's columns is in the column space
A = a_singular  # Using our singular matrix as an example
x = np.array([1, 0])  # Try this vector
v = A @ x  # v will be in the column space by construction
print("\nVector v in column space of A:")
print(f"A = \n{A}")
print(f"x = {x}")
print(f"v = Ax = {v}")
print("v is in column space since it's a linear combination of A's columns")

# Find a vector in the null space of A
# For singular matrix, there exists non-zero x where Ax = 0
# For our singular matrix where second row is 2× first row
# If x = [2, -1], then Ax = 0:
x_null = np.array([2, -1])
result = A @ x_null
print("\nVector x in null space of A:")
print(f"x = {x_null}")
print(f"Ax = {result}")
print("x is in null space since Ax = 0")

# Verify this is indeed in null space
is_zero = np.allclose(result, np.zeros_like(result))
print(f"Is Ax zero? {is_zero}")

x = np.linspace(0, 2, 100)
y = x ** 2
fig = plt.figure(figsize=(10, 5), dpi=100)
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])      #add_axes([bottom x, bottom y, width, height])
ax1.plot(x, y, 'b')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Example figure');
plt.show()