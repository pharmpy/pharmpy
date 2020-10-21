import math

import numpy as np
import sympy

# This module could probably be made private.


def triangular_root(x):
    """Calculate the triangular root of x. I.e. if x is a triangular number T_n what is n?"""
    return math.floor(math.sqrt(2 * x))


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
    for (group_index, _) in sorted_fractions.iteritems():
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
    num_gradient = np.array([float(x.subs(values)) for x in symb_gradient])
    se = np.sqrt(num_gradient @ cov.values @ num_gradient.T)
    return se


def is_posdef(A):
    """Checks whether a matrix is positive definite"""
    try:
        _ = np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def nearest_posdef(A):
    """Return the nearest positive definite matrix in the Frobenius norm to a matrix

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    [3] https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd
    """
    if is_posdef(A):
        return A

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if is_posdef(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    Id = np.eye(A.shape[0])
    k = 1
    while not is_posdef(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += Id * (-mineig * k ** 2 + spacing)
        k += 1

    return A3


def sample_truncated_joint_normal(mu, sigma, a, b, n):
    """Give an array of samples from the truncated joint normal distributon using sample rejection
    - mu, sigma - parameters for the normal distribution
    - a, b - vectors of lower and upper limits for each random variable
    - n - number of samples
    """
    kept_samples = np.empty((0, len(mu)))
    remaining = n
    while remaining > 0:
        samples = np.random.multivariate_normal(mu, sigma, size=remaining, check_valid='raise')
        in_range = np.logical_and(samples > a, samples < b).all(axis=1)
        kept_samples = np.concatenate((kept_samples, samples[in_range]))
        remaining = n - len(kept_samples)
    return kept_samples


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

    mu_bar = M1 + S12 @ S22_inv @ (a - M2)
    sigma_bar = S11 - S12 @ S22_inv @ S21

    return mu_bar, sigma_bar


def round_to_n_sigdig(x, n):
    if x == 0:
        return x
    else:
        return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))


def is_near_target(x, target, zero_limit, significant_digits):
    if target == 0:
        return abs(x) < abs(zero_limit)
    else:
        return round_to_n_sigdig(x, n=significant_digits) == round_to_n_sigdig(
            target, n=significant_digits
        )
