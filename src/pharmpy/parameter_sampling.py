import pandas as pd

from pharmpy.math import sample_truncated_joint_normal


def sample_from_covariance_matrix(model, modelfit_results=None, n=1):
    """Sample parameter vectors using the covariance matrix

       if modelfit_results is not provided the results from the model will be used

       returns a dataframe with one sample per row
    """
    if modelfit_results is None:
        modelfit_results = model.modelfit_results

    index = modelfit_results.parameter_estimates.index
    mu = modelfit_results.parameter_estimates.to_numpy()
    sigma = modelfit_results.covariance_matrix.to_numpy()
    parameter_summary = model.parameters.summary()
    parameter_summary = parameter_summary[~parameter_summary['fix']]
    a = parameter_summary.lower.astype('float64').to_numpy()
    b = parameter_summary.upper.astype('float64').to_numpy()

    # reject non-posdef
    kept_samples = pd.DataFrame()
    remaining = n
    while remaining > 0:
        samples = sample_truncated_joint_normal(mu, sigma, a, b, n=n)
        df = pd.DataFrame(samples, columns=index)
        selected = df[df.apply(model.random_variables.validate_parameters, axis=1)]
        kept_samples = pd.concat((kept_samples, selected))
        remaining = n - len(kept_samples)

    return kept_samples
