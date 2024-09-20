import pharmpy.deps.scipy as scipy
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


def build_initial_values_matrix(rvs, parameters):
    # Only omegas/sigmas to estimate will be included
    # so fixed omegas/sigmas will not be included and
    # omegas for IOV will only be included once
    blocks = []
    seen_parameters = []
    for dist in rvs:
        if len(dist) == 1:
            parameter = parameters[dist.variance.name]
            if parameter.name in seen_parameters:
                continue
            seen_parameters.append(parameter.name)
            if parameter.fix:
                continue
            block = parameter.init
        else:
            var = dist.variance
            parameter = parameters[var[0, 0].name]
            if parameter.name in seen_parameters:
                continue
            seen_parameters.append(parameter.name)
            if parameter.fix:
                continue
            block = var.subs(parameters.inits).to_numpy()
        blocks.append(block)
    return scipy.linalg.block_diag(*blocks)


def build_parameter_coordinates(A):
    # From an initial values matrix list tuples of
    # coordinates of estimated parameters
    # only consider the lower triangle
    coords = []
    for row in range(len(A)):
        for col in range(row + 1):
            if A[row, col] != 0.0:
                coords.append((row, col))
    return coords
