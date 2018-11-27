import math
import numpy as np


def triangular_root(x):
    '''Calculate the triangular root of x. I.e. if x is a triangular number T_n what is n?
    '''
    return math.floor(math.sqrt(2 * x))


def flattened_to_symmetric(x):
    '''Convert a vector containing the elements of a lower triangular matrix into a full symmetric matrix
    '''
    n = triangular_root(len(x))
    new = np.zeros((n, n))
    inds = np.triu_indices_from(new)
    new[inds] = x
    new[(inds[1], inds[0])] = x
    return new
