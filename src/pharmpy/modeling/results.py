import math
import warnings

import numpy as np
import pandas as pd
import sympy

from pharmpy.math import round_to_n_sigdig
from pharmpy.model import Model
from pharmpy.modeling import create_rng, get_observations, sample_parameters_from_covariance_matrix
from pharmpy.random_variables import RandomVariables
from pharmpy.statements import sympify

from .data import get_ids


def calculate_eta_shrinkage(model, sd=False):
    """Calculate eta shrinkage for each eta

    Parameters
    ----------
    model : Model
        Pharmpy model
    sd : bool
        Calculate shrinkage on the standard deviation scale (default is to calculate on the
        variance scale)

    Return
    ------
    Series
        Shrinkage for each eta

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> calculate_eta_shrinkage(model)
    ETA(1)    0.720481
    ETA(2)    0.240295
    dtype: float64
    >>> calculate_eta_shrinkage(model, sd=True)
    ETA(1)    0.471305
    ETA(2)    0.128389
    dtype: float64

    See also
    --------
    calculate_individual_shrinkage

    """
    res = model.modelfit_results
    pe = res.parameter_estimates
    # Want parameter estimates combined with fixed parameter values
    param_inits = model.parameters.to_dataframe()['value']
    pe = pe.combine_first(param_inits)

    ie = res.individual_estimates
    param_names = model.random_variables.iiv.variance_parameters
    diag_ests = pe[param_names]
    diag_ests.index = ie.columns
    if not sd:
        shrinkage = 1 - (ie.var() / diag_ests)
    else:
        shrinkage = 1 - (ie.std() / (diag_ests**0.5))
    return shrinkage


def calculate_individual_shrinkage(model):
    """Calculate the individual eta-shrinkage

    Definition: ieta_shr = (var(eta) / omega)

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    DataFrame
        Shrinkage for each eta and individual

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> calculate_individual_shrinkage(model)
          ETA(1)    ETA(2)
    ID
    1   0.847789  0.256473
    2   0.796643  0.210669
    3   0.755025  0.226957
    4   0.764541  0.216405
    5   0.816192  0.203974
    6   0.778108  0.210992
    7   0.659420  0.236875
    8   0.668551  0.240097
    9   0.260056  0.200374
    10  0.725190  0.226563
    11  0.972110  0.421852
    12  0.249640  0.254119
    13  0.730294  0.364932
    14  0.165785  0.194464
    15  0.813399  0.313554
    16  0.797328  0.213211
    17  0.769059  0.278079
    18  0.098506  0.176778
    19  0.749022  0.235386
    20  0.742181  0.222932
    21  0.317956  0.264473
    22  0.943950  0.232732
    23  0.707183  0.259077
    24  0.553787  0.247717
    25  0.826349  0.114302
    26  0.854777  0.341384
    27  0.820829  0.263235
    28  0.999942  0.319986
    29  0.967084  0.432760
    30  0.404773  0.325215
    31  0.999980  0.318421
    32  0.925283  0.167667
    33  0.913706  0.242106
    34  0.875554  0.249197
    35  0.849135  0.294294
    36  0.172206  0.246422
    37  0.747380  0.278340
    38  0.187440  0.231249
    39  0.237805  0.254485
    40  0.999925  0.189793
    41  0.941906  0.170998
    42  0.923801  0.244046
    43  0.999928  0.320403
    44  0.237637  0.260453
    45  0.869540  0.194503
    46  0.999949  0.319750
    47  0.983782  0.393234
    48  0.698267  0.169337
    49  0.776674  0.214962
    50  0.688847  0.192608
    51  0.822213  0.202534
    52  0.511489  0.273601
    53  0.964757  0.223448
    54  0.762153  0.181648
    55  0.965657  0.435741
    56  0.995278  0.354798
    57  0.813382  0.263372
    58  0.727295  0.232867
    59  0.738777  0.224742

    See also
    --------
    calculate_eta_shrinkage

    """
    res = model.modelfit_results
    cov = res.individual_estimates_covariance
    pe = res.parameter_estimates
    # Want parameter estimates combined with fixed parameter values
    param_inits = model.parameters.to_dataframe()['value']
    pe = pe.combine_first(param_inits)

    # Get all iiv and iov variance parameters
    diag = model.random_variables.etas.covariance_matrix.diagonal()
    param_names = [s.name for s in diag]
    # param_names = model.random_variables.etas.variance_parameters
    # param_names = list(OrderedSet(param_names))  # Only unique in order

    diag_ests = pe[param_names]

    def fn(row, ests):
        names = row[0].index
        ser = pd.Series(np.diag(row[0].values) / ests, index=names)
        return ser

    ish = pd.DataFrame(cov).apply(fn, axis=1, ests=diag_ests.values)
    return ish


def calculate_individual_parameter_statistics(model, exprs, rng=None):
    """Calculate statistics for individual parameters

    Calculate the mean (expected value of the distribution), variance
    (variance of the distribution) and standard error for individual
    parameters described by arbitrary expressions. Any dataset column or
    variable used in the model can be used in the expression. The exception
    being that variables that depends on the solution of the ODE system
    cannot be used. If covariates are used in the expression the statistics
    of the parameter is calculated at the median value of each covariate as well
    as at the 5:th and 95:th percentiles. If no parameter uncertainty is available
    for the model the standard error will not be calculated.

    Parameters
    ----------
    model : Model
        A previously estimated model
    exprs : str, sympy expression or iterable of str or sympy expressions
        Expressions or equations for parameters of interest. If equations are used
        the names of the left hand sides will be used as the names of the parameters.
    rng : Generator or int
        Random number generator or int seed

    Returns
    -------
    pd.DataFrame
        A DataFrame of statistics indexed on parameter and covariate value.

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, create_rng
    >>> from pharmpy.modeling import calculate_individual_parameter_statistics
    >>> model = load_example_model("pheno")
    >>> rng = create_rng(23)
    >>> calculate_individual_parameter_statistics(model, "K=CL/V", rng=rng)
                              mean  variance    stderr
    parameter covariates
    K         p5          0.004234  0.000001  0.001138
              median      0.004909  0.000001  0.001212
              p95         0.004910  0.000001  0.001263
    """
    rng = create_rng(rng)

    if isinstance(exprs, str) or isinstance(exprs, sympy.Basic):
        exprs = [_split_equation(exprs)]
    else:
        exprs = [_split_equation(expr) for expr in exprs]
    cols = set(model.datainfo.names)
    i = 0
    table = pd.DataFrame(columns=['parameter', 'covariates', 'mean', 'variance', 'stderr'])
    for name, expr in exprs:
        full_expr = model.statements.before_odes.full_expression(expr)
        covariates = {symb.name for symb in full_expr.free_symbols if symb.name in cols}
        if not covariates:
            cases = {'median': dict()}
        else:
            dataset = model.dataset
            q5 = dataset[['ID'] + list(covariates)].groupby('ID').median().quantile(0.05)
            q95 = dataset[['ID'] + list(covariates)].groupby('ID').median().quantile(0.95)
            median = dataset[['ID'] + list(covariates)].groupby('ID').median().median()
            cases = {'p5': dict(q5), 'median': dict(median), 'p95': dict(q95)}

        df = pd.DataFrame(index=list(cases.keys()), columns=['mean', 'variance', 'stderr'])
        for case, cov_values in cases.items():
            pe = dict(model.modelfit_results.parameter_estimates)
            cov_expr = full_expr.subs(cov_values)
            expr = cov_expr.subs(pe)
            samples = model.random_variables.sample(expr, parameters=pe, samples=1000000, rng=rng)

            mean = np.mean(samples)
            variance = np.var(samples)

            if model.modelfit_results.covariance_matrix is not None:
                parameters = sample_parameters_from_covariance_matrix(
                    model,
                    n=100,
                    force_posdef_covmatrix=True,
                    rng=rng,
                )
                samples = []
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    for _, row in parameters.iterrows():
                        batch = model.random_variables.sample(
                            cov_expr.subs(dict(row)),
                            parameters=dict(row),
                            samples=10,
                            rng=rng,
                        )
                        samples.extend(list(batch))
                stderr = pd.Series(samples).std()
            else:
                stderr = np.nan
            df.loc[case] = [mean, variance, stderr]
            df.index.name = 'covariates'

        df.reset_index(inplace=True)
        if not name:
            name = f'unknown{i}'
            i += 1
        df['parameter'] = name
        table = pd.concat([table, df])
    table.set_index(['parameter', 'covariates'], inplace=True)
    table = table.astype('float64')
    return table


def calculate_pk_parameters_statistics(model, rng=None):
    """Calculate statistics for common pharmacokinetic parameters

    Calculate the mean (expected value of the distribution), variance
    (variance of the distribution) and standard error for some individual
    pre-defined pharmacokinetic parameters.

    Parameters
    ----------
    model : Model
        A previously estimated model
    rng : Generator or int
        Random number generator or seed

    Returns
    -------
    pd.DataFrame
        A DataFrame of statistics indexed on parameter and covariate value.

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, create_rng
    >>> from pharmpy.modeling import calculate_pk_parameters_statistics
    >>> model = load_example_model("pheno")
    >>> rng = create_rng(23)
    >>> calculate_pk_parameters_statistics(model, rng=rng)  # doctest: +NORMALIZE_WHITESPACE
                              mean  variance    stderr
    parameter covariates
    k_e       p5          0.004234  0.000001  0.001138
              median      0.004909  0.000001  0.001212
              p95         0.004910  0.000001  0.001263

    See Also
    --------
    calculate_individual_parameter_statistics : Calculation of statistics for arbitrary parameters

    """

    statements = model.statements
    odes = statements.ode_system
    central = odes.central_compartment
    output = odes.output_compartment
    depot = odes.find_depot(statements)
    peripherals = odes.peripheral_compartments
    elimination_rate = odes.get_flow(central, output)

    expressions = []  # Eq(name, expr)
    # FO abs + 1comp + FO elimination
    if len(odes) == 3 and depot and odes.t not in elimination_rate.free_symbols:
        exodes = odes.to_explicit_system(skip_output=True)
        ode_list, ics = exodes.odes, exodes.ics
        sols = sympy.dsolve(ode_list, ics=ics)
        expr = sols[1].rhs
        d = sympy.diff(expr, odes.t)
        tmax_closed_form = sympy.solve(d, odes.t)[0]
        expressions.append(sympy.Eq(sympy.Symbol('t_max'), tmax_closed_form))
        e2 = sympy.simplify(expr / depot.dose.amount / sympy.denom(elimination_rate))
        cmax_dose_closed_form = sympy.simplify(e2.subs({odes.t: tmax_closed_form}))
        expressions.append(sympy.Eq(sympy.Symbol('C_max_dose'), cmax_dose_closed_form))

    # Any abs + 1comp + FO elimination
    if not peripherals and odes.t not in elimination_rate.free_symbols:
        elimination_system = statements.copy().ode_system
        # keep central and output
        for name in elimination_system.names:
            if name not in [central.name, output.name]:
                elimination_system.remove_compartment(elimination_system.find_compartment(name))
                exodes = elimination_system.to_explicit_system(skip_output=True)
                ode_list, ics = exodes.odes, exodes.ics
                A0 = sympy.Symbol('A0')
                ic = ics.popitem()
                ics = {ic[0]: A0}
                sols = sympy.dsolve(ode_list[0], ics=ics)
                eq = sympy.Eq(sympy.Rational(1, 2) * A0, sols.rhs)
                thalf_elim = sympy.solve(eq, odes.t)[0]
                expressions.append(sympy.Eq(sympy.Symbol('t_half_elim'), thalf_elim))

    # Bolus dose + 2comp + FO elimination
    if len(peripherals) == 1 and len(odes) == 3 and odes.t not in elimination_rate.free_symbols:
        exodes = odes.to_explicit_system(skip_output=True)
        ode_list, ics = exodes.odes, exodes.ics
        sols = sympy.dsolve(ode_list, ics=ics)
        A = sympy.Wild('A')
        B = sympy.Wild('B')
        alpha = sympy.Wild('alpha')
        beta = sympy.Wild('beta')

        m = sols[0].rhs.match(A * sympy.exp(-alpha) + B * sympy.exp(-beta))

        beta = m[beta] / odes.t
        alpha = m[alpha] / odes.t
        A = m[A] / central.dose.amount
        B = m[B] / central.dose.amount

        if (alpha - alpha).extract_multiplicatively(-1) is not None:
            # alpha > beta  (sympy couldn't simplify this directly)
            alpha, beta = beta, alpha
            A, B = B, A
        expressions.append(sympy.Eq(sympy.Symbol('A'), A))
        expressions.append(sympy.Eq(sympy.Symbol('B'), B))
        expressions.append(sympy.Eq(sympy.Symbol('alpha'), alpha))
        expressions.append(sympy.Eq(sympy.Symbol('beta'), beta))

    # Any abs + any comp + FO elimination
    if odes.t not in elimination_rate.free_symbols:
        expressions.append(sympy.Eq(sympy.Symbol('k_e'), elimination_rate))

    df = calculate_individual_parameter_statistics(model, expressions, rng=rng)
    return df


def _split_equation(s):
    if isinstance(s, str):
        a = s.split('=')
        if len(a) == 1:
            name = None
            expr = sympify(s)
        else:
            name = a[0].strip()
            expr = sympify(a[1])
    elif isinstance(s, sympy.Eq):
        name = s.lhs.name
        expr = s.rhs
    else:  # sympy expr
        name = None
        expr = s
    if name is None and isinstance(expr, sympy.Symbol):
        name = expr.name
    return name, expr


def _result_summary(model, include_all_estimation_steps=False):
    res = model.modelfit_results
    if not include_all_estimation_steps:
        summary_dict = _summarize_step(model, -1)
        summary_df = pd.DataFrame(summary_dict, index=[res.model_name])
        return summary_df
    else:
        summary_dicts = []
        tuples = []
        for i in range(len(res)):
            summary_dict = _summarize_step(model, i)
            is_evaluation = res.model.estimation_steps[i].evaluation
            if is_evaluation:
                run_type = 'evaluation'
            else:
                run_type = 'estimation'
            summary_dict = {**{'run_type': run_type}, **summary_dict}
            summary_dicts.append(summary_dict)
            tuples.append((res.model_name, i + 1))
        index = pd.MultiIndex.from_tuples(tuples, names=['model_name', 'step'])
        summary_df = pd.DataFrame(summary_dicts, index=index)
        return summary_df


def _summarize_step(model, i):
    res = model.modelfit_results
    summary_dict = dict()

    if i >= 0:
        step = res[i]
    else:
        step = res

    if step.minimization_successful is not None:
        summary_dict['minimization_successful'] = step.minimization_successful
    else:
        summary_dict['minimization_successful'] = False

    summary_dict['ofv'] = step.ofv
    summary_dict['aic'] = calculate_aic(model, modelfit_results=res)
    summary_dict['bic'] = calculate_bic(model, modelfit_results=res)
    summary_dict['runtime_total'] = step.runtime_total
    summary_dict['estimation_runtime'] = step.estimation_runtime

    pe = step.parameter_estimates
    ses = step.standard_errors
    rses = step.relative_standard_errors

    for param in pe.index:
        summary_dict[f'{param}_estimate'] = pe[param]
        if ses is not None:
            summary_dict[f'{param}_SE'] = ses[param]
        if rses is not None:
            summary_dict[f'{param}_RSE'] = rses[param]

    return summary_dict


def summarize_modelfit_results(models, include_all_estimation_steps=False):
    """Summarize results of model runs

    Summarize different results after fitting a model, includes runtime, ofv,
    and parameter estimates (with errors). If include_all_estimation_steps is False,
    only the last estimation step will be included (note that in that case, the
    minimization_successful value will be referring to the last estimation step, if
    last step is evaluation it will go backwards until it finds an estimation step
    that wasn't an evaluation).

    Parameters
    ----------
    models : list, Model
        List of models or single model
    include_all_estimation_steps : bool
        Whether to include all estimation steps, default is False

    Return
    ------
    pd.DataFrame
        A DataFrame of modelfit results with model name and estmation step as index.

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, summarize_modelfit_results
    >>> model = load_example_model("pheno")
    >>> summarize_modelfit_results([model]) # doctest: +ELLIPSIS
          minimization_successful         ofv  ... runtime_total  ... SIGMA(1,1)_RSE
    pheno                    True  586.276056  ...           4.0  ...       0.172147
    """
    if isinstance(models, Model):
        models = [models]

    summaries = []

    for model in models:
        if model.modelfit_results:
            summaries.append(_result_summary(model, include_all_estimation_steps))
        else:
            # FIXME: in read_modelfit_results, maybe some parts can be extracted (i.e.
            #   create modelfit_results object)
            if include_all_estimation_steps:
                for i, est in enumerate(model.estimation_steps):
                    index = pd.MultiIndex.from_tuples(
                        [(model.name, i + 1)], names=['model_name', 'step']
                    )
                    if est.evaluation:
                        run_type = 'evaluation'
                    else:
                        run_type = 'estimation'
                    empty_df = pd.DataFrame(
                        {'run_type': run_type, 'minimization_successful': False}, index=index
                    )
                    summaries.append(empty_df)
            else:
                empty_df = pd.DataFrame({'minimization_successful': False}, index=[model.name])
                summaries.append(empty_df)

    summary = pd.concat(summaries).sort_index()

    return summary


def calculate_aic(model, modelfit_results=None):
    """Calculate final AIC for model assuming the OFV to be -2LL

    AIC = OFV + 2*n_estimated_parameters

    Parameters
    ----------
    model : Model
        Pharmpy model object
    modelfit_results : ModelfitResults
        Alternative results object. Default is to use the one in model

    Returns
    -------
    float
        AIC of model fit
    """
    if modelfit_results is None:
        modelfit_results = model.modelfit_results

    parameters = model.parameters.copy()
    parameters.remove_fixed()
    return modelfit_results.ofv + 2 * len(parameters)


def _random_etas(model):
    var = model.random_variables.etas.variance_parameters
    zerofix = [model.parameters[e].fix and model.parameters[e].init == 0 for e in var]
    keep = []
    for eta, zf in zip(model.random_variables.etas, zerofix):
        if not zf:
            keep.append(eta)
    return RandomVariables(keep)


def calculate_bic(model, type=None, modelfit_results=None):
    """Calculate final BIC value assuming the OFV to be -2LL

    Different variations of the BIC can be calculated:

    * | mixed (default)
      | BIC = OFV + n_random_parameters * log(n_individuals) +
      |       n_fixed_parameters * log(n_observations)
    * | fixed
      | BIC = OFV + n_estimated_parameters * log(n_observations)
    * | random
      | BIC = OFV + n_estimated_parameters * log(n_individals)
    * | iiv
      | BIC = OFV + n_estimated_iiv_omega_parameters * log(n_individals)

    Parameters
    ----------
    model : Model
        Pharmpy model object
    type : str
        Type of BIC to calculate. Default is the mixed effects.
    modelfit_results : ModelfitResults
        Alternative results object. Default is to use the one in model

    Returns
    -------
    float
        BIC of model fit

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> calculate_bic(model)
    611.7071686183284
    >>> calculate_bic(model, type='fixed')
    616.536606983396
    >>> calculate_bic(model, type='random')
    610.7412809453149
    >>> calculate_bic(model, type='iiv')
    594.431131169692
    """
    if modelfit_results is None:
        modelfit_results = model.modelfit_results

    parameters = model.parameters.copy()
    parameters.remove_fixed()
    if type == 'fixed':
        penalty = len(parameters) * math.log(len(get_observations(model)))
    elif type == 'random':
        penalty = len(parameters) * math.log(len(get_ids(model)))
    elif type == 'iiv':
        nomegas_iiv = len(
            [name for name in model.random_variables.iiv.parameter_names if name in parameters]
        )
        penalty = nomegas_iiv * math.log(len(get_ids(model)))
    else:
        random_thetas = set()
        for param in model.statements.ode_system.free_symbols:
            assignment = model.statements.find_assignment(param)
            if assignment:
                expr = model.statements.before_odes.full_expression(assignment.symbol)
                for eta in _random_etas(model):
                    if eta.symbol in expr.free_symbols:
                        symbols = {p.symbol for p in parameters if p.symbol in expr.free_symbols}
                        random_thetas.update(symbols)
                        break
        yexpr = model.statements.after_odes.full_expression(model.dependent_variable)
        for eta in _random_etas(model):
            if eta.symbol in yexpr.free_symbols:
                symbols = {p.symbol for p in parameters if p.symbol in yexpr.free_symbols}
                random_thetas.update(symbols)
                for eps in model.random_variables.epsilons:
                    if eps.symbol in yexpr.free_symbols:
                        params = {p.symbol for p in parameters if p.name in eps.parameter_names}
                        random_thetas.update(params)
                break
        nomegas = len(
            [name for name in model.random_variables.etas.parameter_names if name in parameters]
        )
        dim_theta_r = nomegas + len(random_thetas)
        dim_theta_f = len(parameters) - dim_theta_r
        nsubs = len(get_ids(model))
        nobs = len(get_observations(model))
        penalty = dim_theta_r * math.log(nsubs) + dim_theta_f * math.log(nobs)
    ofv = modelfit_results.ofv
    return ofv + penalty


def check_high_correlations(model, limit=0.9):
    """Check for highly correlated parameter estimates

    Parameters
    ----------
    model : Model
        Pharmpy model object
    limit : float
        Lower limit for a high correlation

    Returns
    -------
    pd.Series
        Correlation values indexed on pairs of parameters for (absolute) correlations above limit

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> check_high_correlations(model, limit=0.3)
    THETA(1)  OMEGA(1,1)   -0.388059
    THETA(2)  THETA(3)     -0.356899
              OMEGA(2,2)    0.356662
    dtype: float64
    """
    df = model.modelfit_results.correlation_matrix
    if df is not None:
        high_and_below_diagonal = df.abs().ge(limit) & np.triu(np.ones(df.shape), k=1).astype(bool)
    return df.where(high_and_below_diagonal).stack()


def check_parameters_near_bounds(model, values=None, zero_limit=0.001, significant_digits=2):
    """Check if any estimated parameter value is close to its bounds

    Parameters
    ----------
    model : Model
        Pharmpy model object
    values : pd.Series
        Series of values with index a subset of parameter names.
        Default is to use all parameter estimates
    zero_limit : number
        maximum distance to 0 bounds
    significant_digits : int
        maximum distance to non-zero bounds in number of significant digits

    Returns
    -------
    pd.Series
        Logical Series with same index as values

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> check_parameters_near_bounds(model)
    THETA(1)      False
    THETA(2)      False
    THETA(3)      False
    OMEGA(1,1)    False
    OMEGA(2,2)    False
    SIGMA(1,1)    False
    dtype: bool

    """
    if values is None:
        values = model.modelfit_results.parameter_estimates
    ser = pd.Series(
        [
            _is_close_to_bound(model.parameters[p], values.loc[p], zero_limit, significant_digits)
            for p in values.index
        ],
        index=values.index,
        dtype=bool,
    )
    return ser


def _is_close_to_bound(param, value=None, zero_limit=0.01, significant_digits=2):
    if value is None:
        value = param.init
    return (
        param.lower > -sympy.oo
        and _is_near_target(value, param.lower, zero_limit, significant_digits)
    ) or (
        param.upper < sympy.oo
        and _is_near_target(value, param.upper, zero_limit, significant_digits)
    )


def _is_near_target(x, target, zero_limit, significant_digits):
    if target == 0:
        return abs(x) < abs(zero_limit)
    else:
        return round_to_n_sigdig(x, n=significant_digits) == round_to_n_sigdig(
            target, n=significant_digits
        )


def write_results(results, path, lzma=False, csv=False):
    """Write results object to json (or csv) file

    Note that the csv-file cannot be read into a results object again.

    Parameters
    ----------
    results : Results
        Pharmpy results object
    path : Path
        Path to results file
    lzma : bool
        True for lzma compression. Not applicable to csv file
    csv : bool
        Save as csv file
    """
    if csv:
        results.to_csv(path)
    else:
        results.to_json(path, lzma=lzma)


def print_fit_summary(model):
    """Print a summary of the model fit

    Parameters
    ----------
    model : Model
        Pharmpy model object
    """

    def bool_ok_error(x):
        return "OK" if x else "ERROR"

    def bool_yes_no(x):
        return "YES" if x else "NO"

    def print_header(text, first=False):
        if not first:
            print()
        print(text)
        print("-" * len(text))

    def print_fmt(text, result):
        print(f"{text:33} {result}")

    res = model.modelfit_results

    print_header("Parameter estimation status", first=True)
    print_fmt("Minimization successful", bool_ok_error(res.minimization_successful))
    print_fmt("No rounding errors", bool_ok_error(res.termination_cause != 'rounding_errors'))
    print_fmt("Objective function value", round(res.ofv, 1))

    print_header("Parameter uncertainty status")
    cov_run = model.estimation_steps[-1].cov
    print_fmt("Covariance step run", bool_yes_no(cov_run))

    if cov_run:
        condno = round(np.linalg.cond(res.correlation_matrix), 1)
        print_fmt("Condition number", condno)
        print_fmt("Condition number < 1000", bool_ok_error(condno < 1000))
        hicorr = check_high_correlations(model)
        print_fmt("No correlations arger than 0.9", bool_ok_error(hicorr.empty))

    print_header("Parameter estimates")
    pe = res.parameter_estimates
    if cov_run:
        se = res.standard_errors
        rse = se / pe
        rse.name = 'RSE'
        df = pd.concat([pe, se, rse], axis=1)
    else:
        df = pd.concat([pe], axis=1)
    print(df)
