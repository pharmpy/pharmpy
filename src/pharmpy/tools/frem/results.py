import itertools
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import symengine
import sympy

import pharmpy.symbols as symbols
import pharmpy.visualization  # noqa
from pharmpy import Model
from pharmpy.math import conditional_joint_normal, is_posdef
from pharmpy.modeling import (
    calculate_individual_shrinkage,
    create_rng,
    get_baselines,
    get_covariate_baselines,
    sample_individual_estimates,
    sample_parameters_from_covariance_matrix,
)
from pharmpy.results import Results


class FREMResults(Results):
    """FREM Results class

    What follows is a description on how the FREM results are stored.
    See :py:mod:`pharmpy.tools.frem` for a description of how the results are calculated.

    .. attribute:: covariate_baselines

        DataFrame with the baseline covariate values used in the analysis for each indiviual.
        Index is ID. One column for each covariate.

        .. code-block::

                  WGT  APGR
            ID
            1.0   1.4   7.0
            2.0   1.5   9.0
            3.0   1.5   6.0
            4.0   0.9   6.0
            5.0   1.4   7.0
            6.0   1.2   5.0
            7.0   1.0   5.0
            8.0   1.2   7.0
            9.0   1.4   8.0
            ...   ...   ...


    .. attribute:: covariate_effects

        DataFrame with covariate effects. Index is parameter, covariate and condition where
        condition can be either 5th or 95th. Columns are p5, mean and p95 for the 5th
        percentile of the effect, the estimated effect and the 95th percentile of the effect
        respectively. Effect sizes are in fractions. Example:

        .. code-block::

                                                 p5      mean       p95
            parameter covariate condition
            ETA(1)    WGT       5th        0.901869  0.972920  1.051093
                                95th       0.903849  1.064605  1.233111
                      APGR      5th        1.084238  1.248766  1.413923
                                95th       0.848297  0.901919  0.962312
            ETA(2)    WGT       5th        0.942450  1.004400  1.068390
                                95th       0.874409  0.995763  1.127777
                      APGR      5th        0.832008  0.924457  1.021307
                                95th       0.990035  1.039444  1.091288

    .. attribute:: covariate_effects_plot

        Plot of the covariate effects generated by :py:meth:`plot_covariate_effects`

    .. attribute:: covariate_statistics

        DataFrame with summary statistics of the covariate baselines. Index is covariates.
        Columns are p5 with 95th percentile, mean, p95 with 95th percentile, stdev, ref with
        the used reference value, categorical and other with the non-reference value for
        categorical covariates. Example:

        .. code-block

                                    p5      mean  p95     stdev       ref  categorical  other
                        covariate
                        WGT        0.7  1.525424  3.2  0.704565  1.525424        False    NaN
                        APGR       1.0  6.423729  9.0  2.237636  6.423729        False    NaN

    .. attribute:: individual_effects

        DataFrame with individual covariate effects. Index is ID and parameter. Columns
        are observed, p5 and p95 for the observed individual effect, 5th and 95th
        percentiles of the estimated individual effects respectively. Example:

        .. code-block::

                            observed        p5       p95
            ID   parameter
            1.0  ETA(1)     0.973309  0.946282  0.982391
                 ETA(2)     1.009411  0.995276  1.024539
            2.0  ETA(1)     0.911492  0.832168  0.941760
                 ETA(2)     1.036200  0.990081  1.091438
            3.0  ETA(1)     1.013772  1.007555  1.028463
            ...                  ...       ...       ...
            57.0 ETA(2)     0.987500  0.942709  1.031111
            58.0 ETA(1)     0.939409  0.883782  0.956792
                 ETA(2)     1.023321  0.993543  1.057408
            59.0 ETA(1)     0.992578  0.952785  1.027261
                 ETA(2)     0.999220  0.968931  1.033819

    .. attribute:: individual_effects_plot

        Plot of the individual effects generated by :py:meth:`plot_individual_effects`

    .. attribute:: unexplained_variability

        DataFrame with remaining unexplained variability. Index is parameter and covariate.
        Covariate is none for no covariates and all for all covariates. Example:

        .. code-block::

                                 sd_observed    sd_5th   sd_95th
            parameter covariate
            ETA(1)    none          0.195327  0.221382  0.298465
                      WGT           0.194527  0.218385  0.292261
                      APGR          0.182497  0.202192  0.279766
                      all           0.178956  0.197282  0.268090
            ETA(2)    none          0.158316  0.152510  0.210044
                      WGT           0.158313  0.148041  0.207276
                      APGR          0.155757  0.149037  0.203477
                      all           0.155551  0.144767  0.201658

    .. attribute:: unexplained_variability_plot

        Plot of the unexplained variability generated by :py:meth:`plot_unexplained_variability`

    .. attribute:: parameter_inits_and_estimates

        Initial parameter estimates and estimates after fitting for all intermediate models and
        the final model.

    .. attribute:: base_parameter_change

        The relative change in parameter estimates from base model to the FREM model.

    .. attribute:: covariate_estimates

        Model estimates of covariate statistics

    .. attribute:: parameter_variability

        Conditioned parameter variability

    .. attribute:: coefficients

        Parameter covariate coefficients. Calculated one at a time or all together.

    """

    rst_path = Path(__file__).parent / 'report.rst'

    def __init__(
        self,
        coefficients=None,
        parameter_variability=None,
        covariate_effects=None,
        individual_effects=None,
        unexplained_variability=None,
        covariate_statistics=None,
        covariate_effects_plot=None,
        individual_effects_plot=None,
        unexplained_variability_plot=None,
        covariate_baselines=None,
        parameter_inits_and_estimates=None,
        base_parameter_change=None,
        estimated_covariates=None,
        ofv=None,
    ):
        # FIXME: Lots of boilerplate code ahead. Could be simplified with python 3.7 dataclass
        #        or namedtuple
        self.coefficients = coefficients
        self.parameter_variability = parameter_variability
        self.covariate_effects = covariate_effects
        self.individual_effects = individual_effects
        self.unexplained_variability = unexplained_variability
        self.covariate_statistics = covariate_statistics
        self.covariate_effects_plot = covariate_effects_plot
        self.individual_effects_plot = individual_effects_plot
        self.unexplained_variability_plot = unexplained_variability_plot
        self.covariate_baselines = covariate_baselines
        self.parameter_inits_and_estimates = parameter_inits_and_estimates
        self.base_parameter_change = base_parameter_change
        self.ofv = ofv
        self.estimated_covariates = estimated_covariates

    def add_plots(self):
        self.covariate_effects_plot = self.plot_covariate_effects()
        self.individual_effects_plot = self.plot_individual_effects()
        self.unexplained_variability_plot = self.plot_unexplained_variability()

    def plot_covariate_effects(self):
        """Plot covariate effects"""
        ce = (self.covariate_effects - 1) * 100
        cov_stats = pd.melt(
            self.covariate_statistics.reset_index(),
            var_name='condition',
            id_vars=['covariate'],
            value_vars=['p5', 'p95', 'other'],
        )

        cov_stats = cov_stats.replace({'p5': '5th', 'p95': '95th'}).set_index(
            ['covariate', 'condition']
        )

        ce = ce.join(cov_stats, how='inner')

        # The left join reorders the index, pandas bug #34133
        ce = ce.reorder_levels(['parameter', 'covariate', 'condition'])

        param_names = list(ce.index.get_level_values('parameter').unique())
        plots = []

        for parameter in param_names:
            df = ce.xs(parameter, level=0)
            df = df.reset_index()

            error_bars = (
                alt.Chart(df)
                .mark_errorbar(ticks=True)
                .encode(
                    x=alt.X('p5:Q', title='Effect size in percent', scale=alt.Scale(zero=False)),
                    x2=alt.X2('p95:Q'),
                    y=alt.Y('condition:N', title=None),
                )
            )

            rule = (
                alt.Chart(df)
                .mark_rule(strokeDash=[10, 4], color='gray')
                .encode(x=alt.X('xzero:Q'))
                .transform_calculate(xzero="0")
            )

            points = (
                alt.Chart(df)
                .mark_point(filled=True, color='black')
                .encode(
                    x=alt.X('mean:Q'),
                    y=alt.Y('condition:N'),
                )
            )

            text = (
                alt.Chart(df)
                .mark_text(dy=-15, color="red")
                .encode(x=alt.X("mean:Q"), y=alt.Y("condition:N"), text=alt.Text("value:Q"))
            )

            plot = (
                alt.layer(error_bars, rule, points, text, data=df, width=700, height=100)
                .facet(columns=1.0, row=alt.Facet('covariate:N', title=None), title=f'{parameter}')
                .resolve_scale(y='independent')
            )

            plots.append(plot)

        v = alt.vconcat(*plots).resolve_scale(x='shared')
        return v

    def plot_individual_effects(self):
        """Plot individual effects"""
        covs = self.covariate_baselines
        ie = self.individual_effects.join(covs)
        param_names = list(ie.index.get_level_values('parameter').unique())
        ie = (ie - 1) * 100
        ie = ie.sort_values(by=['observed'])

        plots = []

        for parameter in param_names:
            df = ie.xs(parameter, level=1)

            id_order = list(df.index)
            id_order = [str(int(x)) for x in id_order]

            if len(df) > 20:
                id_order[10] = '...'

            df = df.reset_index()
            df['ID'] = df['ID'].astype(int).astype(str)

            error_bars = (
                alt.Chart(df)
                .mark_errorbar(ticks=True)
                .encode(
                    x=alt.X('p5:Q', title='Effect size in percent', scale=alt.Scale(zero=False)),
                    x2=alt.X2('p95:Q'),
                    y=alt.Y('ID:N', title='ID', sort=id_order),
                    tooltip=['ID', 'p5', 'observed', 'p95'] + list(covs.columns),
                )
            )

            rule = (
                alt.Chart(df)
                .mark_rule(strokeDash=[10, 2], color='gray')
                .encode(x=alt.X('xzero:Q'))
                .transform_calculate(xzero="0")
            )

            points = (
                alt.Chart(df)
                .mark_point(size=40, filled=True, color='black')
                .encode(
                    x=alt.X('observed:Q'),
                    y=alt.Y('ID:N', sort=id_order),
                )
            )

            plot = alt.layer(
                points,
                error_bars,
                rule,
                data=df,
                width=700,
                title=f'Individuals for parameter {parameter}',
            )
            if len(df) > 20:
                plot = (
                    plot.transform_window(
                        sort=[alt.SortField('observed', order='ascending')],
                        rank='row_number(observed)',
                    )
                    .transform_window(
                        sort=[alt.SortField('observed', order='descending')],
                        nrank='row_number(observed)',
                    )
                    .transform_filter('datum.rank <= 10 | datum.nrank <= 11')
                    .transform_calculate(
                        ID="datum.nrank == 11 ? '...' : datum.ID",
                        p5="datum.nrank == 11 ? '...' : datum.p5",
                        p95="datum.nrank == 11 ? '...' : datum.p95",
                        observed="datum.nrank == 11 ? '...' : datum.observed",
                    )
                )
            plots.append(plot)

        v = alt.vconcat(*plots).resolve_scale(x='shared')
        return v

    def plot_unexplained_variability(self):
        """Plot unexplained variability"""
        uv = self.unexplained_variability
        param_names = list(uv.index.get_level_values('parameter').unique())
        cov_order = list(uv.index.get_level_values('covariate').unique())
        plots = []

        for parameter in param_names:
            df = uv.xs(parameter, level=0)
            df = df.reset_index()

            error_bars = (
                alt.Chart(df)
                .mark_errorbar(ticks=True)
                .encode(
                    x=alt.X(
                        'sd_5th:Q',
                        title='SD of unexplained variability',
                        scale=alt.Scale(zero=False),
                    ),
                    x2=alt.X2('sd_95th:Q'),
                    y=alt.Y('covariate:N', title='covariate', sort=cov_order),
                    tooltip=['sd_5th', 'sd_observed', 'sd_95th', 'covariate'],
                )
            )

            rule = (
                alt.Chart(df)
                .mark_rule(strokeDash=[10, 2], color='gray')
                .encode(x=alt.X('xzero:Q'))
                .transform_calculate(xzero="0")
            )

            points = (
                alt.Chart(df)
                .mark_point(size=40, filled=True, color='black')
                .encode(
                    x=alt.X('sd_observed:Q'),
                    y=alt.Y('covariate:N', sort=cov_order),
                )
            )

            plot = alt.layer(
                points,
                error_bars,
                rule,
                data=df,
                width=700,
                title=f'Unexplained variability on {parameter}',
            )

            plots.append(plot)

        v = alt.vconcat(*plots).resolve_scale(x='shared')
        return v


def calculate_results(
    frem_model, continuous, categorical, method=None, intermediate_models=None, rng=None, **kwargs
):
    """Calculate FREM results

    :param method: Either 'cov_sampling' or 'bipp'
    """
    if intermediate_models is None:
        intermediate_models = []

    if method is None or method == 'cov_sampling':
        try:
            res = calculate_results_using_cov_sampling(
                frem_model, continuous, categorical, rng=rng, **kwargs
            )
        except AttributeError:
            # Fallback to bipp
            res = calculate_results_using_bipp(frem_model, continuous, categorical, rng=rng)
    elif method == 'bipp':
        res = calculate_results_using_bipp(frem_model, continuous, categorical, rng=rng)
    else:
        raise ValueError(f'Unknown frem postprocessing method {method}')
    mod_names = []
    mod_ofvs = []
    if intermediate_models:
        for intmod in intermediate_models:
            intmod_res = intmod.modelfit_results
            if intmod_res is not None:
                mod_ofvs.append(intmod_res.ofv)
                mod_names.append(intmod.name)
    mod_ofvs.append(frem_model.modelfit_results.ofv)
    mod_names.append(frem_model.name)
    ofv = pd.DataFrame({'ofv': mod_ofvs}, index=mod_names)
    ofv.index.name = 'model_name'
    res.ofv = ofv

    add_parameter_inits_and_estimates(res, frem_model, intermediate_models)
    if intermediate_models:
        add_base_vs_frem_model(res, frem_model, intermediate_models[0])

    estimated_covbase = _calculate_covariate_baselines(frem_model, continuous + categorical)
    mean = estimated_covbase.mean()
    stdev = estimated_covbase.std()
    estcovs = pd.DataFrame({'mean': mean, 'stdev': stdev})
    res.estimated_covariates = estcovs
    return res


def add_base_vs_frem_model(res, frem_model, model_1):
    base_ests = model_1.modelfit_results.parameter_estimates
    final_ests = frem_model.modelfit_results.parameter_estimates
    ser = pd.Series(dtype=np.float64)
    for param in base_ests.keys():
        ser[param] = (final_ests[param] - base_ests[param]) / abs(base_ests[param]) * 100
    ser.name = 'relative_change'
    res.base_parameter_change = ser


def add_parameter_inits_and_estimates(res, frem_model, intermediate_models):
    model_names = []
    df = pd.DataFrame()

    for model in intermediate_models:
        try:
            df = df.append(model.parameters.nonfixed_inits, ignore_index=True)
        except AttributeError:
            df = df.append(np.NaN)
        try:
            df = df.append(model.modelfit_results.parameter_estimates, ignore_index=True)
        except AttributeError:
            df = df.append(np.NaN)
        model_names.append(model.name)

    try:
        df = df.append(frem_model.parameters.nonfixed_inits, ignore_index=True)
        df = df.reindex(columns=frem_model.parameters.nonfixed_inits.keys())
    except AttributeError:
        df = df.append(np.NaN)
    try:
        df = df.append(frem_model.modelfit_results.parameter_estimates, ignore_index=True)
    except AttributeError:
        df = df.append(np.NaN)
    model_names.append(frem_model.name)
    index = pd.MultiIndex.from_product([model_names, ['init', 'estimate']], names=['model', 'type'])
    df.index = index
    res.parameter_inits_and_estimates = df


def calculate_results_using_cov_sampling(
    frem_model,
    continuous,
    categorical,
    cov_model=None,
    force_posdef_samples=500,
    force_posdef_covmatrix=False,
    samples=1000,
    rescale=True,
    rng=None,
):
    """Calculate the FREM results using covariance matrix for uncertainty

    :param cov_model: Take the parameter uncertainty covariance matrix from this model
                      instead of the frem model.
    :param force_posdef_samples: The number of sampling tries before stopping to use
                                 rejection sampling and instead starting to shift values so
                                 that the frem matrix becomes positive definite. Set to 0 to
                                 always force positive definiteness.
    :param force_posdef_covmatrix: Set to force the covariance matrix of the frem movdel or
                                   the cov model to be positive definite. Default is to raise
                                   in this case.
    :param samples: The number of parameter vector samples to use.
    """
    if cov_model is not None:
        uncertainty_results = cov_model.modelfit_results
    else:
        uncertainty_results = frem_model.modelfit_results

    _, dist = frem_model.random_variables.iiv.distributions()[-1]
    sigma_symb = dist.sigma

    parameters = [
        s
        for s in frem_model.modelfit_results.parameter_estimates.index
        if symbols.symbol(s) in sigma_symb.free_symbols
    ]
    parvecs = sample_parameters_from_covariance_matrix(
        frem_model,
        modelfit_results=uncertainty_results,
        force_posdef_samples=force_posdef_samples,
        force_posdef_covmatrix=force_posdef_covmatrix,
        parameters=parameters,
        n=samples,
        rng=rng,
    )
    res = calculate_results_from_samples(
        frem_model, continuous, categorical, parvecs, rescale=rescale
    )
    return res


def calculate_results_from_samples(frem_model, continuous, categorical, parvecs, rescale=True):
    """Calculate the FREM results given samples of parameter estimates"""
    n = len(parvecs)
    rvs, dist = frem_model.random_variables.iiv.distributions()[-1]
    sigma_symb = dist.sigma
    parameters = [
        s
        for s in frem_model.modelfit_results.parameter_estimates.index
        if symbols.symbol(s) in sigma_symb.free_symbols
    ]
    parvecs = parvecs.append(frem_model.modelfit_results.parameter_estimates.loc[parameters])

    df = frem_model.dataset
    covariates = continuous + categorical
    frem_model.datainfo.set_column_type(covariates, 'covariate')
    covariate_baselines = get_covariate_baselines(frem_model)
    covariate_baselines = covariate_baselines[covariates]
    cov_means = covariate_baselines.mean()
    cov_modes = covariate_baselines.mode().iloc[0]  # Select first mode if more than one
    cov_others = pd.Series(index=cov_modes[categorical].index, dtype=np.float64)
    cov_stdevs = covariate_baselines.std()
    for _, row in covariate_baselines.iterrows():
        for cov in categorical:
            if row[cov] != cov_modes[cov]:
                cov_others[cov] = row[cov]
        if not cov_others.isna().values.any():
            break

    cov_refs = pd.concat((cov_means[continuous], cov_modes[categorical]))
    cov_5th = covariate_baselines.quantile(0.05, interpolation='lower')
    cov_95th = covariate_baselines.quantile(0.95, interpolation='higher')
    is_categorical = cov_refs.index.isin(categorical)

    res = FREMResults()

    res.covariate_statistics = pd.DataFrame(
        {
            'p5': cov_5th,
            'mean': cov_means,
            'p95': cov_95th,
            'stdev': cov_stdevs,
            'ref': cov_refs,
            'categorical': is_categorical,
            'other': cov_others,
        },
        index=covariates,
    )
    res.covariate_statistics.index.name = 'covariate'

    ncovs = len(covariates)
    npars = sigma_symb.rows - ncovs
    param_names = get_params(frem_model, rvs, npars)
    nids = len(covariate_baselines)
    param_indices = list(range(npars))
    if rescale:
        scaling = np.diag(np.concatenate((np.ones(npars), cov_stdevs.values)))

    mu_bars_given_5th = np.empty((n, ncovs, npars))
    mu_bars_given_95th = np.empty((n, ncovs, npars))
    mu_id_bars = np.empty((n, nids, npars))
    original_id_bar = np.empty((nids, npars))
    variability = np.empty((n, ncovs + 2, npars))  # none, cov1, cov2, ..., all
    original_variability = np.empty((ncovs + 2, npars))
    coefficients_index = pd.MultiIndex.from_product(
        [['all', 'each'], param_names], names=['condition', 'parameter']
    )
    coefficients = pd.DataFrame(index=coefficients_index, columns=covariates, dtype=np.float64)
    parameter_variability = []

    # Switch to symengine for speed
    # Could also assume order of parameters, but not much gain
    sigma_symb = symengine.sympify(sigma_symb)
    parvecs.columns = [symengine.Symbol(colname) for colname in parvecs.columns]

    estimated_covbase = _calculate_covariate_baselines(frem_model, covariates)
    covbase = estimated_covbase.to_numpy()

    for sample_no, params in parvecs.iterrows():
        sigma = sigma_symb.subs(dict(params))
        sigma = np.array(sigma).astype(np.float64)
        if rescale:
            sigma = scaling @ sigma @ scaling
        if sample_no != 'estimates':
            variability[sample_no, 0, :] = np.diag(sigma)[:npars]
        else:
            original_variability[0, :] = np.diag(sigma)[:npars]
            # Coefficients conditioned on all parameters
            # Sigma_12 * Sigma_22^-1
            coeffs_all = sigma[0:npars, npars:] @ np.linalg.inv(sigma[npars:, npars:])
            coefficients.loc['all'] = coeffs_all
        for i, cov in enumerate(covariates):
            indices = param_indices + [i + npars]
            cov_sigma = sigma[indices][:, indices]
            cov_mu = np.array([0] * npars + [cov_refs[cov]])
            if cov in categorical:
                first_reference = cov_others[cov]
            else:
                first_reference = cov_5th[cov]
            mu_bar_given_5th_cov, sigma_bar = conditional_joint_normal(
                cov_mu, cov_sigma, np.array([first_reference])
            )
            if sample_no != 'estimates':
                mu_bar_given_95th_cov, _ = conditional_joint_normal(
                    cov_mu, cov_sigma, np.array([cov_95th[cov]])
                )
                mu_bars_given_5th[sample_no, i, :] = mu_bar_given_5th_cov
                mu_bars_given_95th[sample_no, i, :] = mu_bar_given_95th_cov
                variability[sample_no, i + 1, :] = np.diag(sigma_bar)
            else:
                original_variability[i + 1, :] = np.diag(sigma_bar)
                parameter_variability.append(sigma_bar)
                # Calculate coefficients
                # Cov(Par, covariate) / Var(covariate)
                for parind, parname in enumerate(param_names):
                    coefficients[cov]['each', parname] = cov_sigma[parind][-1] / cov_sigma[-1][-1]

        for i in range(len(estimated_covbase)):
            row = covbase[i, :]
            id_mu = np.array([0] * npars + list(cov_refs))
            mu_id_bar, sigma_id_bar = conditional_joint_normal(id_mu, sigma, row)
            if sample_no != 'estimates':
                mu_id_bars[sample_no, i, :] = mu_id_bar
                variability[sample_no, -1, :] = np.diag(sigma_id_bar)
            else:
                original_id_bar[i, :] = mu_id_bar
                original_variability[ncovs + 1, :] = np.diag(sigma_id_bar)
                parameter_variability_all = sigma_id_bar

    res.coefficients = coefficients

    # Create covariate effects table
    mu_bars_given_5th = np.exp(mu_bars_given_5th)
    mu_bars_given_95th = np.exp(mu_bars_given_95th)

    means_5th = np.mean(mu_bars_given_5th, axis=0)
    means_95th = np.mean(mu_bars_given_95th, axis=0)
    q5_5th = np.quantile(mu_bars_given_5th, 0.05, axis=0)
    q5_95th = np.quantile(mu_bars_given_95th, 0.05, axis=0)
    q95_5th = np.quantile(mu_bars_given_5th, 0.95, axis=0)
    q95_95th = np.quantile(mu_bars_given_95th, 0.95, axis=0)

    df = pd.DataFrame(columns=['parameter', 'covariate', 'condition', 'p5', 'mean', 'p95'])

    for param, cov in itertools.product(range(npars), range(ncovs)):
        if covariates[cov] in categorical:
            df = df.append(
                {
                    'parameter': param_names[param],
                    'covariate': covariates[cov],
                    'condition': 'other',
                    'p5': q5_5th[cov, param],
                    'mean': means_5th[cov, param],
                    'p95': q95_5th[cov, param],
                },
                ignore_index=True,
            )
        else:
            df = df.append(
                {
                    'parameter': param_names[param],
                    'covariate': covariates[cov],
                    'condition': '5th',
                    'p5': q5_5th[cov, param],
                    'mean': means_5th[cov, param],
                    'p95': q95_5th[cov, param],
                },
                ignore_index=True,
            )
            df = df.append(
                {
                    'parameter': param_names[param],
                    'covariate': covariates[cov],
                    'condition': '95th',
                    'p5': q5_95th[cov, param],
                    'mean': means_95th[cov, param],
                    'p95': q95_95th[cov, param],
                },
                ignore_index=True,
            )
    df.set_index(['parameter', 'covariate', 'condition'], inplace=True)
    res.covariate_effects = df

    # Create id table
    mu_id_bars = np.exp(mu_id_bars)
    original_id_bar = np.exp(original_id_bar)

    with np.testing.suppress_warnings() as sup:  # Would warn in case of missing covariates
        sup.filter(RuntimeWarning, "All-NaN slice encountered")
        id_5th = np.nanquantile(mu_id_bars, 0.05, axis=0)
        id_95th = np.nanquantile(mu_id_bars, 0.95, axis=0)

    df = pd.DataFrame(columns=['parameter', 'observed', 'p5', 'p95'])
    for curid, param in itertools.product(range(nids), range(npars)):
        df = df.append(
            pd.Series(
                {
                    'parameter': param_names[param],
                    'observed': original_id_bar[curid, param],
                    'p5': id_5th[curid, param],
                    'p95': id_95th[curid, param],
                },
                name=covariate_baselines.index[curid],
            )
        )
    df.index.name = 'ID'
    df = df.set_index('parameter', append=True)
    res.individual_effects = df

    # Create unexplained variability table
    sd_5th = np.sqrt(np.nanquantile(variability, 0.05, axis=0))
    sd_95th = np.sqrt(np.nanquantile(variability, 0.95, axis=0))
    original_sd = np.sqrt(original_variability)

    df = pd.DataFrame(columns=['parameter', 'covariate', 'sd_observed', 'sd_5th', 'sd_95th'])
    for par, cond in itertools.product(range(npars), range(ncovs + 2)):
        if cond == 0:
            condition = 'none'
        elif cond == ncovs + 1:
            condition = 'all'
        else:
            condition = covariates[cond - 1]
        df = df.append(
            {
                'parameter': param_names[par],
                'covariate': condition,
                'sd_observed': original_sd[cond, par],
                'sd_5th': sd_5th[cond, par],
                'sd_95th': sd_95th[cond, par],
            },
            ignore_index=True,
        )
    df = df.set_index(['parameter', 'covariate'])
    res.unexplained_variability = df

    res.covariate_baselines = covariate_baselines

    # Create frem_parameter_variability
    index = pd.MultiIndex.from_product(
        [['all'] + covariates, param_names], names=['condition', 'parameter']
    )
    df = pd.DataFrame(index=index)
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            df.loc[('all', param_names[i]), param_names[j]] = parameter_variability_all[i][j]
    for k, name in enumerate(covariates):
        for i in range(len(param_names)):
            for j in range(len(param_names)):
                df.loc[(name, param_names[i]), param_names[j]] = parameter_variability[k][i][j]
    res.parameter_variability = df
    return res


def get_params(frem_model, rvs, npars):
    param_names = [rv.name for rv in rvs][:npars]
    sset = frem_model.statements.before_odes
    symbs = []

    for p in param_names:
        statement = [s for s in reversed(sset) if sympy.Symbol(p) in s.rhs_symbols][0]
        if str(statement.expression) == p:
            statement = [s for s in reversed(sset) if statement.symbol in s.rhs_symbols][0]
        symbs.append(statement.symbol.name)

    duplicates = set([e for e in symbs if symbs.count(e) > 1])

    for i, s in enumerate(symbs):
        if s in duplicates:
            symbs[i] = rename_duplicate(symbs, s)

    return symbs


def rename_duplicate(params, stem):
    i = 1
    while True:
        candidate = f'{stem}({i})'
        if candidate not in params:
            return candidate
        i += 1


def _calculate_covariate_baselines(model, covariates):
    exprs = [
        ass.expression.args[0][0]
        for ass in model.statements
        if symbols.symbol('FREMTYPE') in ass.free_symbols and ass.symbol.name == 'IPRED'
    ]
    exprs = [
        expr.subs(dict(model.modelfit_results.parameter_estimates)).subs(model.parameters.inits)
        for expr in exprs
    ]
    new = []
    for expr in exprs:
        for symb in expr.free_symbols:
            stat = model.statements.find_assignment(symb.name)
            if stat is not None:
                expr = expr.subs(symb, stat.expression)
        new.append(expr)
    exprs = new

    def fn(row):
        return [np.float64(expr.subs(dict(row))) for expr in exprs]

    df = model.modelfit_results.individual_estimates.apply(fn, axis=1, result_type='expand')
    df.columns = covariates
    return df


def calculate_results_using_bipp(
    frem_model, continuous, categorical, rescale=True, samples=2000, rng=None
):
    """Estimate a covariance matrix for the frem model using the BIPP method

    Bootstrap on the individual parameter posteriors
    Only the individual estimates, individual unvertainties and the parameter estimates
    are needed.

    """
    rng = create_rng(rng)
    rvs, dist = frem_model.random_variables.iiv.distributions()[-1]
    etas = [rv.name for rv in rvs]
    pool = sample_individual_estimates(frem_model, parameters=etas, rng=rng).droplevel('sample')
    ninds = len(pool.index.unique())
    ishr = calculate_individual_shrinkage(frem_model)
    ishr = ishr[pool.columns]
    lower_indices = np.tril_indices(len(etas))
    pop_params = np.array(dist.sigma).astype(str)[lower_indices]
    parameter_samples = np.empty((samples, len(pop_params)))
    remaining_samples = samples
    k = 0
    while k < remaining_samples:
        bootstrap = pool.sample(n=ninds, replace=True, random_state=rng.bit_generator)
        ishk = ishr.loc[bootstrap.index]
        cf = (1 / (1 - ishk.mean())) ** (1 / 2)
        corrected_bootstrap = bootstrap * cf
        bootstrap_cov = corrected_bootstrap.cov()
        if not is_posdef(bootstrap_cov.to_numpy()):
            continue
        parameter_samples[k, :] = bootstrap_cov.values[lower_indices]
        k += 1
    frame = pd.DataFrame(parameter_samples, columns=pop_params)
    # Shift to the mean of the parameter estimate
    shift = frem_model.modelfit_results.parameter_estimates[pop_params] - frame.mean()
    frame = frame + shift
    res = calculate_results_from_samples(
        frem_model, continuous, categorical, frame, rescale=rescale
    )
    return res


def psn_reorder_base_model_inits(model, path):
    """Reorder omega inits from base model in PsN

    If base model was reordered PsN writes the omega inits dict to m1/model_1.inits
    """
    order_path = path / 'm1' / 'model_1.inits'
    if order_path.is_file():
        with open(order_path, 'r') as fh:
            lines = fh.readlines()
        lines = lines[1:-1]
        replacements = dict()
        for line in lines:
            stripped = line.strip().replace(' ', '').replace("'", '').rstrip(',').replace('>', '')
            a = stripped.split('=')
            coords = a[0][6:-1]
            t = tuple(coords.split(','))
            t = (int(t[0]), int(t[1]))
            try:
                replacements[t] = float(a[1])
            except ValueError:
                pass

        def sortfunc(x):
            return float(x[0]) + float(x[1]) / 1000

        order = sorted(replacements, key=sortfunc)
        values = [replacements[i] for i in order]
        i = 0
        for p in model.parameters:
            if i == len(values):
                break
            if p.name in model.random_variables.parameter_names:
                p.init = values[i]
                i += 1


def psn_frem_results(path, force_posdef_covmatrix=False, force_posdef_samples=500, method=None):
    """Create frem results from a PsN FREM run

    :param path: Path to PsN frem run directory
    :return: A :class:`FREMResults` object

    """
    path = Path(path)

    model_4_path = path / 'final_models' / 'model_4.mod'
    if not model_4_path.is_file():
        raise IOError(f'Could not find FREM model 4: {str(model_4_path)}')
    model_4 = Model(model_4_path)
    if model_4.modelfit_results is None:
        raise ValueError('Model 4 has no results')
    cov_model = None
    if method == 'cov_sampling':
        try:
            model_4.modelfit_results.covariance_matrix
        except Exception:
            model_4b_path = path / 'final_models' / 'model_4b.mod'
            try:
                model_4b = Model(model_4b_path)
            except FileNotFoundError:
                pass
            else:
                cov_model = model_4b

    with open(path / 'covariates_summary.csv') as covsum:
        covsum.readline()
        raw_cov_list = covsum.readline()
    all_covariates = raw_cov_list[1:].rstrip().split(',')

    # FIXME: Not introducing yaml parser in pharmpy just yet. Options should be collected
    # differently. Perhaps using json
    logtransformed_covariates = []
    with open(path / 'meta.yaml') as meta:
        for row in meta:
            row = row.strip()
            if row.startswith('rescale: 1'):
                rescale = True
            elif row.startswith('rescale: 0'):
                rescale = False
            if row.startswith("log: ''"):
                logtransformed_covariates = []
            elif row.startswith('log: '):
                logtransformed_covariates = row[5:].split(',')

    # add log transformed columns for the -log option. Should be done when creating dataset
    df = model_4.dataset
    if logtransformed_covariates:
        for lncov in logtransformed_covariates:
            df[f'LN{lncov}'] = np.log(df[lncov])
        model_4.dataset = df

    nunique = get_baselines(model_4)[all_covariates].nunique()
    continuous = list(nunique.index[nunique != 2])
    categorical = list(nunique.index[nunique == 2])

    intmod_names = ['model_1.mod', 'model_2.mod', 'model_3.mod', 'model_3b.mod']
    intmods = []
    for m in intmod_names:
        intmod_path = path / 'm1' / m
        if intmod_path.is_file():
            intmod = Model(intmod_path)
            intmods.append(intmod)

    model1b = Model(path / 'm1' / 'model_1b.mod')
    model1 = intmods[0]
    model1b.modelfit_results = model1.modelfit_results
    model1b.modelfit_results.parameter_estimates = model1b.parameters.nonfixed_inits
    psn_reorder_base_model_inits(model1b, path)

    res = calculate_results(
        model_4,
        continuous,
        categorical,
        method=method,
        force_posdef_covmatrix=force_posdef_covmatrix,
        force_posdef_samples=force_posdef_samples,
        cov_model=cov_model,
        rescale=rescale,
        intermediate_models=intmods,
        rng=np.random.default_rng(9843),
    )
    return res
