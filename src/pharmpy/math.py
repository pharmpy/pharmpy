import math

import numpy as np
import sympy

# This module could probably be made private.


def triangular_root(x):
    '''Calculate the triangular root of x. I.e. if x is a triangular number T_n what is n?
    '''
    return math.floor(math.sqrt(2 * x))


def flattened_to_symmetric(x):
    '''Convert a vector containing the elements of a lower triangular matrix into a full symmetric
       matrix
    '''
    n = triangular_root(len(x))
    new = np.zeros((n, n))
    inds = np.tril_indices_from(new)
    new[inds] = x
    new[(inds[1], inds[0])] = x
    return new


def cov2corr(cov):
    """Convert covariance matrix to correlation matrix
    """
    v = np.sqrt(np.diag(cov))
    outer_v = np.outer(v, v)
    corr = cov / outer_v
    corr[cov == 0] = 0
    return corr


def round_and_keep_sum(x, s):
    '''Round values in Series x and their sum must be s

       Algorithm: Floor all elements in series. If sum not correct add one to element with
                  highest fractional part until sum is reached.
    '''
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
    """ Use the delta method to estimate the standard error
        of a function of parameters with covariance matrix
        available.

        expr - A sympy expression for the function of parameters
        cov - dataframe with symbol names as indices
              must include at least all parameters needed for expr
        values - dict/series parameter estimates. Must include at least
                 all parameters needed for expr
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
