import numpy as np
import scipy as sp

L = np.array([[1,0,0,0,-1], [0,1,0,0,-1], [0,0,1,0,-1], [0,0,0,0,0], [-1,-1,-1,0,3]]).astype(np.float64)

values, vectors = np.linalg.eig(L)

print("Eigenvalues:")
print(values)
print("\nEigenvectors:")
print(vectors)

# for vector in vectors:
#     for val in vector:
#         print(f"{val:.4f}", end=" & ")
#     print()
