import timeit
import numpy as np

setup_code = """
import numpy as np
n = 960
# Initialize matrices
A = np.random.random((n, n))  # Example initialization
B = np.random.random((n, n))  # Example initialization
C = np.zeros((n, n))          # Important to initialize with zeros
"""
test_code = """
for i in range(n):
    for j in range(n):
        for k in range(n):
            C[i][j] += A[i][k] * B[k][j]
"""

number = 1

time_taken = timeit.timeit(stmt=test_code, setup=setup_code, number=number)
print(f"Execution time: {time_taken:.5f} seconds")