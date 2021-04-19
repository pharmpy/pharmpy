import warnings
from functools import partial

import numpy as np
import pandas as pd

from pharmpy.math import is_posdef, nearest_posdef, sample_truncated_joint_normal


def sample_from_function(model, samplingfn, parameters=None, force_posdef_samples=None, n=1):
    """Sample parameter vectors using a general function

    The sampling function will be given three arguments:

    - lower - lower bounds of parameters
    - upper - upper bounds of parameters
    - n - number of samples
    """
    if parameters is None:
        parameters = model.parameters.names

    parameter_summary = model.parameters.to_dataframe().loc[parameters]
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
        samples = samplingfn(lower, upper, n=remaining)
        df = pd.DataFrame(samples, columns=parameters)
        if not force_posdef:
            selected = df[df.apply(model.random_variables.validate_parameters, axis=1)]
        else:
            rvs = model.random_variables
            selected = df.transform(lambda row: rvs.nearest_valid_parameters(row), axis=1)
        kept_samples = pd.concat((kept_samples, selected))
        remaining = n - len(kept_samples)
        i += 1
        if not force_posdef and force_posdef_samples is not None and i >= force_posdef_samples:
            force_posdef = True

    return kept_samples.reset_index(drop=True)


def sample_uniformly(
    model, fraction=0.1, parameters=None, force_posdef_samples=None, n=1, seed=None
):
    """Sample parameter vectors using uniform sampling

    Each parameter value will be randomly sampled from a uniform distriution
    with lower bound estimate - estimate * fraction and upper bound
    estimate + estimate * fraction
    """

    if seed is None or isinstance(seed, int):
        seed = np.random.default_rng(seed)

    def fn(lower, upper, n):
        samples = np.empty((n, len(lower)))
        for i, (a, b) in enumerate(zip(lower, upper)):
            samples[i, :] = seed.uniform(a, b, n)
        return samples

    samples = sample_from_function(
        model, fn, parameters=parameters, force_posdef_samples=force_posdef_samples, n=n
    )
    return samples


def sample_from_covariance_matrix(
    model,
    modelfit_results=None,
    parameters=None,
    force_posdef_samples=None,
    force_posdef_covmatrix=False,
    n=1,
    seed=None,
):
    """Sample parameter vectors using the covariance matrix

    If modelfit_results is not provided the results from the model will be used

    Parameters
    ----------
    parameters
        Use to only sample a subset of the parameters. None means all
    force_posdef_samples
        Set to how many iterations to do before forcing all samples to be positive definite. None is
        default and means never and 0 means always

    Returns
    -------
    A dataframe with one sample per row
    """
    if modelfit_results is None:
        modelfit_results = model.modelfit_results

    if parameters is None:
        parameters = list(modelfit_results.parameter_estimates.index)

    if seed is None or isinstance(seed, int):
        seed = np.random.default_rng(seed)

    pe = modelfit_results.parameter_estimates[parameters]
    index = pe.index
    mu = pe.to_numpy()
    sigma = modelfit_results.covariance_matrix[parameters].loc[parameters].to_numpy()
    if not is_posdef(sigma):
        if force_posdef_covmatrix:
            old_sigma = sigma
            sigma = nearest_posdef(sigma)
            delta_frobenius = np.linalg.norm(old_sigma) - np.linalg.norm(sigma)
            delta_max = np.abs(old_sigma).max() - np.abs(sigma).max()
            warnings.warn(
                f'Covariance matrix was forced to become positive definite.\n'
                f'    Difference in the frobenius norm: {delta_frobenius:.3e}\n'
                f'    Difference in the max norm: {delta_max:.3e}\n'
            )
        else:
            raise ValueError("Uncertainty covariance matrix not positive-definite")

    fn = partial(sample_truncated_joint_normal, mu, sigma, seed=seed)
    samples = sample_from_function(
        model, fn, parameters=index, force_posdef_samples=force_posdef_samples, n=n
    )
    return samples


def sample_individual_estimates(model, parameters=None, samples_per_id=100, seed=None):
    """Sample individual estimates given their covariance.

    Parameters
    ----------
    parameters
        A list of a subset of parameters to sample. Default is None, which means all.

    Returns
    -------
    Pool of samples in a DataFrame
    """
    if seed is None or isinstance(seed, int):
        seed = np.random.default_rng(seed)
    ests = model.modelfit_results.individual_estimates
    covs = model.modelfit_results.individual_estimates_covariance
    if parameters is None:
        parameters = ests.columns
    ests = ests[parameters]
    samples = pd.DataFrame()
    for (idx, mu), sigma in zip(ests.iterrows(), covs):
        sigma = sigma[parameters].loc[parameters]
        sigma = nearest_posdef(sigma)
        id_samples = seed.multivariate_normal(mu.values, sigma.values, size=samples_per_id)
        id_df = pd.DataFrame(id_samples, columns=ests.columns)
        id_df.index = [idx] * len(id_df)  # ID as index
        samples = pd.concat((samples, id_df))
    return samples
