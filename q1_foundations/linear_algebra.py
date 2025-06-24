import numpy as np

a = np.arange(15).reshape(3, 5)
b = np.array([6, 7, 8])
print(a)
print(type(a))
print(a.shape)
print(a.dtype)

print(b)
print(type(b))
print(b.shape)
print(b.dtype)

# Practice vector addition, subtraction, and scalar multiplication.
# Create some sample vectors
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Vector addition
print("\nVector addition:")
print(f"v1 + v2 = {v1 + v2}")

# Vector subtraction 
print("\nVector subtraction:")
print(f"v1 - v2 = {v1 - v2}")

# Scalar multiplication
scalar = 2
print("\nScalar multiplication:")
print(f"{scalar} * v1 = {scalar * v1}")

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

print("\nManual dot product:")
print(manual_dot_product(v1, v2))
print("\nVector dot product:")
print(vector_dot_product(v1, v2))
print("\nMatrix transpose:")
print(manual_matrix_transpose(a))
print(matrix_transpose(a))


