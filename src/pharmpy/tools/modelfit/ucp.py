from pharmpy.deps import numpy as np


def scale_matrix(A):
    # Create a scale/transformation matrix S for initial parameter matrix A
    # The scale matrix will be used in descaling of ucps
    L = np.linalg.cholesky(A)
    v1 = np.diag(L)
    v2 = v1 / np.exp(0.1)
    M2 = np.diag(v1)
    M3 = np.diag(v2)
    S = np.abs(10.0 * (L - M2)) + M3
    # Convert lower triangle to symmetric
    irows, icols = np.triu_indices(len(S), 1)
    S[irows, icols] = S[icols, irows]
    return S


def descale_matrix(A, S):
    # Descales the lower triangular ucp matrix A using the scaling matrix S
    # The purpose of the scaling/transformation is threefold:
    # 1. Remove constraint on positive definiteness
    # 2. Remove constraint of diagonals (variances) being positive
    # 3. Scale so that all initial values are 0.1 on the ucp scale
    exp_diag = np.exp(np.diag(A))
    M = A.copy()
    np.fill_diagonal(M, exp_diag)
    M2 = M * S
    L = np.tril(M2)
    return L @ L.T
