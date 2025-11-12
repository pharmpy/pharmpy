import math
from itertools import count
from typing import Optional

from pharmpy.deps import numpy as np
from pharmpy.deps import sympy
from pharmpy.internals.expr.subs import subs

# This module could probably be made private.


def triangular(n: int):
    """The n:th triangular number"""
    return n * (n + 1) // 2


def triangular_root(x):
    """Calculate the triangular root of x. I.e. if x is a triangular number T_n what is n?"""
    return math.floor(math.sqrt(2 * x))


def symmetric_to_flattened(A):
    """Flatten a symmetric matrix into a vector using the lower triangle"""
    return np.concatenate([A[i, : i + 1] for i in range(len(A))])


def flattened_to_symmetric(x):
    """Convert a vector containing the elements of a lower triangular matrix into a full symmetric
    matrix
    """
    n = triangular_root(len(x))
    new = np.zeros((n, n))
    inds = np.tril_indices_from(new)
    new[inds] = x
    new[(inds[1], inds[0])] = x
    return new


def cov2corr(cov):
    """Convert covariance matrix to correlation matrix"""
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    corr = cov / outer_v
    corr[cov == 0] = 0
    return corr


def corr2cov(corr, sd):
    """Convert correlation matrix to covariance matrix

    Parameters
    ----------
    corr
        Correlation matrix (ones on diagonal)
    sd
        One dimensional array of standard deviations
    """
    sd_matrix = np.diag(sd)
    cov = sd_matrix @ corr @ sd_matrix
    return cov


def round_and_keep_sum(x, s):
    """Round values in Series x and their sum must be s

    Algorithm: Floor all elements in series. If sum not correct add one to element with
               highest fractional part until sum is reached.
    """
    sorted_fractions = x.apply(lambda x: math.modf(x)[0]).sort_values(ascending=False)
    rounded_sample_sizes = x.apply(lambda x: math.modf(x)[1])
    for group_index, _ in sorted_fractions.items():
        num_samples = rounded_sample_sizes.sum()
        diff = s - num_samples
        if diff == 0:
            break
        step = math.copysign(1, diff)
        rounded_sample_sizes[group_index] += step

    return rounded_sample_sizes.astype('int64')


def se_delta_method(expr, values, cov):
    """Use the delta method to estimate the standard error of a function of parameters with covariance
    matrix available.

    Parameters
    ----------
    expr
        A sympy expression for the function of parameters
    cov
        Dataframe with symbol names as indices must include at least all parameters needed for expr
    values
        dict/series parameter estimates. Must include at least all parameters needed for expr

    Returns
    -------
    Approximation of the standard error
    """
    symbs = expr.free_symbols
    names_unsorted = [s.name for s in symbs]
    # Sort names according to order in cov
    names = [y for x in cov.columns for y in names_unsorted if y == x]
    cov = cov[names].loc[names]
    symb_gradient = [sympy.diff(expr, sympy.Symbol(name)) for name in names]
    num_gradient = np.array([float(subs(x, values, simultaneous=True)) for x in symb_gradient])
    se = np.sqrt(num_gradient @ cov.values @ num_gradient.T)
    return se


def is_symmetric(A):
    return (A == A.T).all().all()


def is_positive_semidefinite(A, is_hermitian: Optional[bool] = None):
    """Checks whether a matrix is positive semi-definite"""
    if is_hermitian or (is_hermitian is None and is_symmetric(A)):
        eigvals = np.linalg.eigvalsh
    else:
        eigvals = np.linalg.eigvals
    return all(eigvals(A) >= 0)


def is_posdef(A):
    """Checks whether a matrix is positive definite"""
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def toeplitz_hermitian_part(A):
    return (A + A.T) / 2


def nearest_positive_semidefinite(A):
    """Return the nearest positive semidefinite matrix in the Frobenius norm to a matrix

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    [3] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    """
    # Check positive semidefinite instead of positive definite since it seems it can happen
    # that a matrix is deemed not positive semidefinite but positive definite, which causes
    # issues when validating and adjusting initial estimates
    if is_symmetric(A):
        if is_positive_semidefinite(A, is_hermitian=True):
            return A
        else:
            B = A

    else:
        if is_positive_semidefinite(A, is_hermitian=False):
            return A
        else:
            B = toeplitz_hermitian_part(A)

    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = toeplitz_hermitian_part(A2)

    eigvals = np.linalg.eigvalsh(A3)
    if all(eigvals >= 0):
        return A3

    spacing = np.spacing(np.linalg.norm(A3))
    # The above is different from [1] because taking increments of `eps(mineig)`
    # is too slow.
    Id = np.eye(A.shape[0])

    for k in count(1):
        mineig = np.min(eigvals)
        A3 += Id * (-mineig * k**2 + spacing)

        eigvals = np.linalg.eigvalsh(A3)
        if all(eigvals >= 0):
            break

    return A3


def nearest_positive_definite(A):
    # Find the (almost) nearest positive definite matrix given a positive semidefinite matrix
    A = A.copy()
    while not is_posdef(A):
        epsilon = np.nextafter(np.diag(A), np.inf)
        np.fill_diagonal(A, epsilon)
    return A


def conditional_joint_normal(mu, sigma, a):
    """Give parameters of the conditional joint normal distribution

    The condition is the last len(a) values

    See https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    """

    # partition mu and sigma
    nfirst = len(mu) - len(a)
    S11 = sigma[0:nfirst, 0:nfirst]
    S12 = sigma[0:nfirst, nfirst:]
    S21 = sigma[nfirst:, 0:nfirst]
    S22 = sigma[nfirst:, nfirst:]
    M1 = mu[0:nfirst]
    M2 = mu[nfirst:]

    S22_inv = np.linalg.inv(S22)
    S12_at_S22_inv = S12 @ S22_inv

    mu_bar = M1 + S12_at_S22_inv @ (a - M2)
    sigma_bar = S11 - S12_at_S22_inv @ S21

    return mu_bar, sigma_bar


def conditional_joint_normal_lambda(mu, sigma, n):
    # NOTE: Same as conditional_joint_normal but for fixed mu, sigma, and len(a)
    S11 = sigma[0:n, 0:n]
    S12 = sigma[0:n, n:]
    S21 = sigma[n:, 0:n]
    S22 = sigma[n:, n:]
    M1 = mu[0:n]
    M2 = mu[n:]

    S22_inv = np.linalg.inv(S22)
    S12_at_S22_inv = S12 @ S22_inv

    sigma_bar = S11 - S12_at_S22_inv @ S21

    def _cjn_eval(a):
        mu_bar = M1 + S12_at_S22_inv @ (a - M2)
        return mu_bar, sigma_bar

    return _cjn_eval


def round_to_n_sigdig(x, n):
    if x == 0:
        return x
    else:
        return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
