import math
import numpy as np

# This module could probably be made private.

def triangular_root(x):
    '''Calculate the triangular root of x. I.e. if x is a triangular number T_n what is n?
    '''
    return math.floor(math.sqrt(2 * x))


def flattened_to_symmetric(x):
    '''Convert a vector containing the elements of a lower triangular matrix into a full symmetric matrix
    '''
    n = triangular_root(len(x))
    new = np.zeros((n, n))
    inds = np.tril_indices_from(new)
    new[inds] = x
    new[(inds[1], inds[0])] = x
    return new


def round_and_keep_sum(x, s):
    '''Round values in Series x and their sum must be s

       Algorithm: Floor all elements in series. If sum not correct add one to element with highest fractional part
                  until sum is reached.
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
