import math

import pharmpy.deps.scipy as scipy
from pharmpy.deps import numpy as np
from pharmpy.deps import symengine


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


def unpack_ucp_matrix(x, coords):
    # Create an n times n matrix
    # Put each value in the vector x at the position of coords
    # Let rest of elements be zero
    n = coords[-1][0] + 1
    A = np.zeros((n, n))
    for val, (row, col) in zip(x, coords):
        A[row, col] = val
    return A


def split_ucps(x, omega_coords, sigma_coords):
    # Split the ucp vector into a theta vector, an omega matrix and a sigma matrix
    # All still on the ucp scale
    nomegas = len(omega_coords)
    nsigmas = len(sigma_coords)
    nthetas = len(x) - nomegas - nsigmas
    theta_ucp = x[0:nthetas]
    omega_ucp = unpack_ucp_matrix(x[nthetas : nthetas + nomegas], omega_coords)
    sigma_ucp = unpack_ucp_matrix(x[nthetas + nomegas :], sigma_coords)
    return theta_ucp, omega_ucp, sigma_ucp


def scale_thetas(parameters):
    # parameters should only contain non-fix thetas
    # returns vectors of scaled theta, lower bound (or None if no bounds)
    # range_ul, the range between lower and upper bound (or None if no bounds)
    theta = []
    lb = []
    range_ul_vec = []

    for p in parameters:
        if p.lower <= -1000000 and p.upper >= 1000000:
            # Unbounded
            theta.append(p.init / 0.1)
            lb.append(None)
            range_ul_vec.append(None)
        else:
            # Bounded in both or one direction
            upper = p.upper if p.upper < 1000000 else 1000000
            lower = p.lower if p.lower > -1000000 else -1000000
            range_ul = upper - lower
            range_prop = (p.init - lower) / range_ul
            scaled = 0.1 - math.log(range_prop / (1.0 - range_prop))
            theta.append(scaled)
            lb.append(lower)
            range_ul_vec.append(range_ul)
    return (theta, lb, range_ul_vec)


def descale_thetas(x, scale):
    # Descale thetas in vector x given theta scale tuple
    # * scale theta if no bounds
    descaled = []
    for ucp, scale_theta, lb, range_ul in zip(x, scale[0], scale[1], scale[2]):
        if lb is None:
            descaled.append(ucp * scale_theta)
        else:
            diff_scale = ucp - scale_theta
            prop_scale = np.exp(diff_scale) / (1.0 + np.exp(diff_scale))
            descaled.append(prop_scale * range_ul + lb)
    return np.array(descaled)


def build_starting_ucp_vector(theta_scale, omega_coords, sigma_coords):
    n = len(theta_scale[0]) + len(omega_coords) + len(sigma_coords)
    x = np.full(n, 0.1)
    return x


def calculate_matrix_gradient_scale(x, scale, coords):
    # For diagonal elements: scale(x_ii) = 2 * s_ii * exp(2 * x_ii)
    # For off diagonals: scale(x_ij) = s_ij * s_jj * exp(x_jj)
    values = []
    for row, col in coords:
        if row == col:
            val = 2.0 * scale[row, col] ** 2 * math.exp(2.0 * x[row, col])
        else:
            val = 2.0 * scale[row, col] * scale[col, col] * math.exp(x[col, col])
        values.append(val)
    return np.array(values)


def calculate_theta_gradient_scale(x, scale):
    # scale value for unbounded
    # For bounded:
    #     -s + x        -2⋅s + 2⋅x
    # rul⋅ℯ         rul⋅ℯ
    # ─────────── - ───────────────
    # -s + x                     2
    # ℯ       + 1    ⎛ -s + x    ⎞
    #                ⎝ℯ       + 1⎠
    s, _, range_ul = scale
    values = []

    for ucp, selt, rul in zip(x, s, range_ul):
        if rul is None:
            val = selt
        else:
            pw = math.exp(ucp - selt)
            val = rul * pw / (pw + 1.0) - rul * math.exp(2.0 * ucp - 2.0 * selt) / (pw + 1.0) ** 2
        values.append(val)
    return np.array(values)


def calculate_gradient_scale(
    theta_ucp,
    omega_ucp,
    sigma_ucp,
    theta_scale,
    omega_scale,
    sigma_scale,
    omega_coords,
    sigma_coords,
):
    # This is the inner derivative of the transformation
    # derivative of the function to go from ucp to normal parameter space
    theta = calculate_theta_gradient_scale(theta_ucp, theta_scale)
    omega = calculate_matrix_gradient_scale(omega_ucp, omega_scale, omega_coords)
    sigma = calculate_matrix_gradient_scale(sigma_ucp, sigma_scale, sigma_coords)
    return np.concatenate((theta, omega, sigma))


def get_parameter_symbols(model):
    symbols = (symengine.Symbol(s) for s in model.parameters.nonfixed.names)
    return tuple(symbols)
