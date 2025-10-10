from sympy import var, Matrix
T, S, m= var('T S m')
G = Matrix([[1, (1.5*T)/m, (T)/m],
            [(1.5*T)/m, (2.25*S)/m, (1.5*S)/m],
            [(T)/m, (1.5*S)/m, (S)/m]])

eigenvalues = G.eigenvals()

eigenvectors = G.eigenvects()

print("Eigenvalues:")
for i,eigenvalue in enumerate(eigenvalues):
    print(f"eigenvalue {i+1}:")
    print(eigenvalue)
print("Eigenvectors:")
for i,eigenvector in enumerate(eigenvectors):
    print(f"Eigenvector of eigenvalue {eigenvector[0]}:")
    print(f"\t {eigenvector[2]}")
