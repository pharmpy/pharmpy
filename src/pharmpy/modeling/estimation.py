import math
from typing import Dict, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import Model


class UCPScale:
    def __init__(self, theta, omega, sigma, lb, range_ul):
        self.theta = theta
        self.omega = omega
        self.sigma = sigma
        self.lb = lb
        self.range_ul = range_ul

    def __repr__(self):
        return "<UCPScale object>"


def calculate_ucp_scale(model: Model):
    """Calculate a scale for unconstrained parameters for a model

    The UCPScale object can be used to calculate unconstrained parameters
    back into the normal parameter space.

    Parameters
    ----------
    model : Model
        Model for which to calculate an ucp scale

    Returns
    -------
    UCPScale
        A scale object

    See Also
    --------
    calculate_parameters_from_ucp : Calculate parameters from ucp:s

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> scale = calculate_ucp_scale(model)

    """
    omega_symbolic = model.random_variables.etas.covariance_matrix
    omega = omega_symbolic.subs(model.parameters.inits)
    omega = np.array(omega).astype(np.float64)
    scale_omega = _scale_matrix(omega)

    sigma_symbolic = model.random_variables.epsilons.covariance_matrix
    sigma = sigma_symbolic.subs(model.parameters.inits)
    sigma = np.array(sigma).astype(np.float64)
    scale_sigma = _scale_matrix(sigma)

    theta = []
    lb = []
    range_ul_vec = []
    for p in model.parameters:
        if not p.fix:
            if p.symbol not in model.random_variables.free_symbols:
                range_ul = p.upper - p.lower
                range_prop = (p.init - p.lower) / range_ul
                scaled = 0.1 - math.log(range_prop / (1.0 - range_prop))
                theta.append(scaled)
                lb.append(p.lower)
                range_ul_vec.append(range_ul)

    return UCPScale(np.array(theta), scale_omega, scale_sigma, np.array(lb), np.array(range_ul_vec))


def _scale_matrix(A):
    chol = np.linalg.cholesky(A)
    M1 = np.triu(chol)
    v1 = np.diag(M1)
    v2 = v1 / np.exp(0.1)
    M2 = np.diag(v1)
    M3 = np.diag(v2)
    m_scale = np.abs(10 * (M1 - M2)) + M3
    return m_scale.T


def calculate_parameters_from_ucp(
    model: Model, scale: UCPScale, ucps: Union[pd.Series, Dict[str, float]]
):
    """Scale parameter values from ucp to normal scale

    Parameters
    ----------
    model : Model
        Pharmpy model
    scale : UCPScale
        A parameter scale
    ucps : pd.Series or dict
        Series of parameter values

    Returns
    -------
    pd.Series
        Parameters on the normal scale

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> scale = calculate_ucp_scale(model)
    >>> values = {'PTVCL': 0.1, 'PTVV': 0.1, 'THETA_3': 0.1, \
                  'IVCL': 0.1, 'IVV': 0.1, 'SIGMA_1_1': 0.1}
    >>> calculate_parameters_from_ucp(model, scale, values)
    PTVCL                    0.004693
    PTVV                      1.00916
    THETA_3                       0.1
    IVCL                    0.0309626
    IVV          0.031127999999999996
    SIGMA_1_1    0.013241000000000001
    dtype: object

    See also
    --------
    calculate_ucp_scale : Calculate the scale for conversion from ucps
    """
    omega_symbolic = model.random_variables.etas.covariance_matrix
    omega = omega_symbolic.subs(dict(ucps))
    omega = np.array(omega).astype(np.float64)
    descaled_omega = _descale_matrix(omega, scale.omega)
    omega_dict = {}
    for symb, numb in zip(omega_symbolic, np.nditer(descaled_omega)):
        if symb.is_Symbol:
            omega_dict[symb] = numb

    sigma_symbolic = model.random_variables.epsilons.covariance_matrix
    sigma = sigma_symbolic.subs(dict(ucps))
    sigma = np.array(sigma).astype(np.float64)
    descaled_sigma = _descale_matrix(sigma, scale.sigma)
    sigma_dict = {}
    for symb, numb in zip(sigma_symbolic, np.nditer(descaled_sigma)):
        if symb.is_Symbol:
            sigma_dict[symb] = numb

    theta = []
    for p in model.parameters:
        if not p.fix:
            if p.symbol not in model.random_variables.free_symbols:
                theta.append(ucps[p.name])
    theta = np.array(theta)

    diff_scale = theta - scale.theta
    prop_scale = np.exp(diff_scale) / (1.0 + np.exp(diff_scale))
    descaled = prop_scale * scale.range_ul + scale.lb

    d = {}
    i = 0
    for p in model.parameters:
        if not p.fix:
            if p.symbol not in model.random_variables.free_symbols:
                d[p.name] = descaled[i]
                i += 1
            elif p.symbol in omega_symbolic.free_symbols:
                d[p.name] = omega_dict[p.symbol]
            else:
                d[p.name] = sigma_dict[p.symbol]

    return pd.Series(d)


def _descale_matrix(A, S):
    exp_diag = np.exp(np.diag(A))
    M = A.copy()
    np.fill_diagonal(M, exp_diag)
    M2 = M * S
    M3 = np.tril(M2)
    return M3 @ M3.T
