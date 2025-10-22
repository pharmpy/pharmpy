from __future__ import annotations

import warnings
from collections.abc import Mapping
from functools import partial
from typing import Literal, Optional, Union

from pharmpy import DEFAULT_SEED
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.math import is_posdef, nearest_positive_semidefinite
from pharmpy.model import Model


def create_rng(seed: Union[np.random.Generator, int] = DEFAULT_SEED):
    """Create a new random number generator

    Pharmpy functions that use random sampling take a random number generator or seed as input.
    This function can be used to create a default new random number generator.

    Parameters
    ----------
    seed : int or rng
        Seed for the random number generator or None (default) for a randomized seed. If seed
        is generator it will be passed through.

    Returns
    -------
    Generator
        Initialized numpy random number generator object

    Examples
    --------
    >>> from pharmpy.modeling import create_rng
    >>> rng = create_rng(23)
    >>> rng.standard_normal()
    0.5532605888887387

    """
    if isinstance(seed, np.random.Generator):
        rng = seed
    elif isinstance(seed, float) and int(seed) == seed:
        # Case to support int-like floats in pharmr
        rng = np.random.default_rng(int(seed))
    else:
        rng = np.random.default_rng(seed)
    return rng


def _sample_truncated_joint_normal(sigma, mu, a, b, n, rng):
    """Give an array of samples from the truncated joint normal distribution using sample rejection
    - mu, sigma - parameters for the normal distribution
    - a, b - vectors of lower and upper limits for each random variable
    - n - number of samples
    """
    if not is_posdef(sigma):
        raise ValueError("Covariance matrix not positive definite")
    kept_samples = np.empty((0, len(mu)))
    remaining = n
    while remaining > 0:
        samples = rng.multivariate_normal(
            mu, sigma, size=remaining, method="cholesky", check_valid='ignore'
        )
        in_range = np.logical_and(samples > a, samples < b).all(axis=1)
        kept_samples = np.concatenate((kept_samples, samples[in_range]))
        remaining = n - len(kept_samples)
    return kept_samples


def _sample_from_function(
    model,
    parameter_estimates,
    samplingfn,
    seed,
    force_posdef_samples=None,
    n=1,
):
    """Sample parameter vectors using a general function

    The sampling function will be given three arguments:

    - estimated parameter values
    - lower - lower bounds of parameters
    - upper - upper bounds of parameters
    - n - number of samples
    """
    rng = create_rng(seed)

    parameter_estimates = parameter_estimates[
        parameter_estimates.index.isin(model.parameters.nonfixed.names)
    ]
    pe = parameter_estimates.to_numpy()

    parameter_summary = model.parameters.to_dataframe().loc[parameter_estimates.keys()]
    parameter_summary = parameter_summary[~parameter_summary['fix']]
    lower = parameter_summary.lower.astype('float64').to_numpy()
    upper = parameter_summary.upper.astype('float64').to_numpy()

    # reject non-posdef
    kept_samples = pd.DataFrame()
    remaining = n

    if force_posdef_samples == 0:
        force_posdef = True
    else:
        force_posdef = False

    i = 0
    while remaining > 0:
        samples = samplingfn(pe, lower, upper, n=remaining, rng=rng)
        df = pd.DataFrame(samples, columns=parameter_estimates.keys())
        if not force_posdef:
            selected = df[df.apply(model.random_variables.validate_parameters, axis=1)]
        else:
            rvs = model.random_variables
            selected = df.transform(
                lambda row: pd.Series(rvs.nearest_valid_parameters(row)), axis=1
            )
        assert isinstance(selected, pd.DataFrame)
        kept_samples = pd.concat((kept_samples, selected))
        remaining = n - len(kept_samples)
        i += 1
        if not force_posdef and force_posdef_samples is not None and i >= force_posdef_samples:
            force_posdef = True

    return kept_samples.reset_index(drop=True)


def sample_parameters_uniformly(
    model: Model,
    parameter_estimates: Mapping[str, float],
    fraction: float = 0.1,
    force_posdef_samples: Optional[int] = None,
    n: int = 1,
    seed: Union[np.random.Generator, int] = DEFAULT_SEED,
    scale: Literal['UCP', 'normal'] = 'normal',
):
    """Sample parameter vectors using uniform sampling

    Each parameter value will be randomly sampled from a uniform distribution
    with the bounds being estimate ± estimate * fraction.

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_estimates : pd.Series
        Parameter estimates for parameters to use
    fraction : float
        Fraction of estimate value to use for distribution bounds
    force_posdef_samples : int
        Number of samples to reject before forcing variability parameters to give
        positive definite covariance matrices.
    n : int
        Number of samples
    seed : int or rng
        Random number generator or seed
    scale : str
        Scale to perform sampling on. Valid options are 'normal' and 'UCP'

    Returns
    -------
    pd.DataFrame
        samples

    Example
    -------
    >>> from pharmpy.modeling import create_rng, sample_parameters_uniformly, load_example_model
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> rng = create_rng(23)
    >>> pe = results.parameter_estimates
    >>> sample_parameters_uniformly(model, pe, n=3, seed=rng)
         POP_CL    POP_VC   COVAPGR    IIV_CL    IIV_VC     SIGMA
    0  0.004878  0.908216  0.149441  0.029179  0.025472  0.012947
    1  0.004828  1.014444  0.149958  0.028853  0.027653  0.013348
    2  0.004347  1.053837  0.165804  0.028465  0.026798  0.013727

    See also
    --------
    sample_parameters_from_covariance_matrix : Sample parameter vectors using the
        uncertainty covariance matrix
    sample_individual_estimates : Sample individual estiates given their covariance

    """

    def fn(pe, lower, upper, n, rng):
        samples = np.empty((n, len(lower)))
        for i, (x, a, b) in enumerate(zip(pe, lower, upper)):
            if scale == 'normal':
                lower = max(a, x - x * fraction)
                upper = min(b, x + x * fraction)
            elif scale == 'UCP':
                lower = 0.1 - 0.1 * fraction
                upper = 0.1 + 0.1 * fraction
            samples[:, i] = rng.uniform(lower, upper, n)
        return samples

    samples = _sample_from_function(
        model, parameter_estimates, fn, force_posdef_samples=force_posdef_samples, n=n, seed=seed
    )
    return samples


def sample_parameters_from_covariance_matrix(
    model: Model,
    parameter_estimates: Mapping[str, float],
    covariance_matrix: pd.DataFrame,
    force_posdef_samples: Optional[int] = None,
    force_posdef_covmatrix: bool = False,
    n: int = 1,
    seed: Union[np.random.Generator, int] = DEFAULT_SEED,
):
    """Sample parameter vectors using the covariance matrix

    If parameters is not provided all estimated parameters will be used

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_estimates : pd.Series
        Parameter estimates to use as means in sampling
    covariance_matrix : pd.DataFrame
        Parameter uncertainty covariance matrix
    force_posdef_samples : int
        Set to how many iterations to do before forcing all samples to be positive definite. None is
        default and means never and 0 means always
    force_posdef_covmatrix : bool
        Set to True to force the input covariance matrix to be positive definite
    n : int
        Number of samples
    seed : Generator
        Random number generator

    Returns
    -------
    pd.DataFrame
        A dataframe with one sample per row

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> rng = create_rng(23)
    >>> cov = results.covariance_matrix
    >>> pe = results.parameter_estimates
    >>> sample_parameters_from_covariance_matrix(model, pe, cov, n=3, seed=rng)
         POP_CL    POP_VC   COVAPGR    IIV_CL    IIV_VC     SIGMA
    0  0.004887  1.000761  0.198184  0.034860  0.031391  0.013750
    1  0.004631  1.024746  0.071056  0.031726  0.026824  0.012597
    2  0.004631  0.991088  0.130841  0.027464  0.024589  0.013215

    See also
    --------
    sample_parameters_uniformly : Sample parameter vectors using uniform distribution
    sample_individual_estimates : Sample individual estiates given their covariance

    """
    sigma = covariance_matrix.loc[parameter_estimates.keys(), parameter_estimates.keys()].to_numpy()
    if not is_posdef(sigma):
        if force_posdef_covmatrix:
            old_sigma = sigma
            sigma = nearest_positive_semidefinite(sigma)
            delta_frobenius = np.linalg.norm(old_sigma) - np.linalg.norm(sigma)
            delta_max = np.abs(old_sigma).max() - np.abs(sigma).max()
            warnings.warn(
                f'Covariance matrix was forced to become positive definite.\n'
                f'    Difference in the frobenius norm: {delta_frobenius:.3e}\n'
                f'    Difference in the max norm: {delta_max:.3e}\n'
            )
        else:
            raise ValueError("Uncertainty covariance matrix not positive-definite")

    fn = partial(_sample_truncated_joint_normal, sigma)
    samples = _sample_from_function(
        model, parameter_estimates, fn, force_posdef_samples=force_posdef_samples, n=n, seed=seed
    )
    return samples


def sample_individual_estimates(
    model: Model,
    individual_estimates: pd.DataFrame,
    individual_estimates_covariance: pd.DataFrame,
    parameters: Optional[list[str]] = None,
    samples_per_id: int = 100,
    seed: Union[np.random.Generator, int] = DEFAULT_SEED,
):
    """Sample individual estimates given their covariance.

    Parameters
    ----------
    model : Model
        Pharmpy model
    individual_estimates : pd.DataFrame
        Individual estimates to use
    individual_estimates_covariance : pd.DataFrame
        Uncertainty covariance of the individual estimates
    parameters : list
        A list of a subset of individual parameters to sample. Default is None, which means all.
    samples_per_id : int
        Number of samples per individual
    seed : rng or int
        Random number generator or seed

    Returns
    -------
    pd.DataFrame
        Pool of samples in a DataFrame

    Example
    -------
    >>> from pharmpy.modeling import create_rng, load_example_model, sample_individual_estimates
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> rng = create_rng(23)
    >>> ie = results.individual_estimates
    >>> iec = results.individual_estimates_covariance
    >>> sample_individual_estimates(model, ie, iec, samples_per_id=2, seed=rng)
                 ETA_CL    ETA_VC
    ID sample
    1  0      -0.127941  0.037273
       1      -0.065492 -0.182851
    2  0      -0.263323 -0.265849
       1      -0.295883 -0.060346
    3  0      -0.012108  0.219967
    ...             ...       ...
    57 1      -0.034279 -0.040988
    58 0      -0.187879 -0.143184
       1      -0.088845 -0.034655
    59 0      -0.187779 -0.014214
       1      -0.019953 -0.151151
    <BLANKLINE>
    [118 rows x 2 columns]

    See also
    --------
    sample_parameters_from_covariance_matrix : Sample parameter vectors using the
        uncertainty covariance matrix
    sample_parameters_uniformly : Sample parameter vectors using uniform distribution

    """
    rng = create_rng(seed)
    assert rng is not None
    ests = individual_estimates
    covs = individual_estimates_covariance
    if parameters is None:
        parameters = list(ests.columns)
    ests = ests[parameters]
    assert isinstance(ests, pd.DataFrame)
    samples = pd.DataFrame()
    for (idx, mu), sigma in zip(ests.iterrows(), covs):
        sigma = sigma.loc[parameters, parameters]
        sigma = nearest_positive_semidefinite(sigma)
        id_samples = rng.multivariate_normal(mu.to_numpy(), sigma.to_numpy(), size=samples_per_id)
        id_df = pd.DataFrame(id_samples, columns=ests.columns)
        id_df['ID'] = idx
        id_df['sample'] = list(range(0, samples_per_id))
        id_df.set_index(['ID', 'sample'], drop=True, inplace=True)
        samples = pd.concat((samples, id_df))
    return samples
