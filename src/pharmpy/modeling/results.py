import warnings

import numpy as np
import pandas as pd
import sympy

from pharmpy.data_structures import OrderedSet
from pharmpy.modeling import create_rng, sample_parameters_from_covariance_matrix


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
        shrinkage = 1 - (ie.std() / (diag_ests ** 0.5))
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

    # Get all iiv variance parameters
    param_names = model.random_variables.etas.variance_parameters
    param_names = list(OrderedSet(param_names))  # Only unique in order

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
    dataset = model.dataset
    cols = set(dataset.columns)
    i = 0
    table = pd.DataFrame(columns=['parameter', 'covariates', 'mean', 'variance', 'stderr'])
    for name, expr in exprs:
        full_expr = model.statements.full_expression_from_odes(expr)
        covariates = {symb.name for symb in full_expr.free_symbols if symb.name in cols}
        if not covariates:
            cases = {'median': dict()}
        else:
            q5 = dataset[{'ID'} | covariates].groupby('ID').median().quantile(0.05)
            q95 = dataset[{'ID'} | covariates].groupby('ID').median().quantile(0.95)
            median = dataset[{'ID'} | covariates].groupby('ID').median().median()
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
    central = odes.find_central()
    output = odes.find_output()
    depot = odes.find_depot(statements)
    peripherals = odes.find_peripherals()
    elimination_rate = odes.get_flow(central, output)

    expressions = []  # Eq(name, expr)
    # FO abs + 1comp + FO elimination
    if len(odes) == 3 and depot and odes.t not in elimination_rate.free_symbols:
        ode_list, ics = odes.to_explicit_odes(skip_output=True)
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
                ode_list, ics = elimination_system.to_explicit_odes(skip_output=True)
                A0 = sympy.Symbol('A0')
                ic = ics.popitem()
                ics = {ic[0]: A0}
                sols = sympy.dsolve(ode_list[0], ics=ics)
                eq = sympy.Eq(sympy.Rational(1, 2) * A0, sols.rhs)
                thalf_elim = sympy.solve(eq, odes.t)[0]
                expressions.append(sympy.Eq(sympy.Symbol('t_half_elim'), thalf_elim))

    # Bolus dose + 2comp + FO elimination
    if len(peripherals) == 1 and len(odes) == 3 and odes.t not in elimination_rate.free_symbols:
        ode_list, ics = odes.to_explicit_odes(skip_output=True)
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
            expr = sympy.sympify(s)
        else:
            name = a[0].strip()
            expr = sympy.sympify(a[1])
    elif isinstance(s, sympy.Eq):
        name = s.lhs.name
        expr = s.rhs
    else:  # sympy expr
        name = None
        expr = s
    if name is None and isinstance(expr, sympy.Symbol):
        name = expr.name
    return name, expr


# TODO: consider moving part of function to results-object
def summarize_modelfit_results(models):
    """Summarize results of multiple model runs

    Summarize different results after fitting a model, includes runtime, ofv,
    and parameter estimates (with errors).

    Parameters
    ----------
    models : list
        List of models

    Return
    ------
    pd.DataFrame
        A DataFrame of modelfit results, one row per model.

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, summarize_modelfit_results
    >>> model = load_example_model("pheno")
    >>> summarize_modelfit_results([model]) # doctest: +ELLIPSIS
          minimization_successful         ofv runtime_total  THETA(1)_estimate  ... SIGMA(1,1)_RSE
    pheno                    True  586.276056           4.0           0.004696  ...       0.172147
    """
    run_summaries = []
    parameter_summaries = []

    for model in models:
        res = model.modelfit_results
        parameter_summary = res.parameter_summary().stack().rename(model.name)
        run_summary = pd.Series(
            {
                'minimization_successful': res.minimization_successful,
                'ofv': res.ofv,
                'runtime_total': res.runtime_total,
            },
            name=model.name,
        )
        parameter_summaries.append(parameter_summary)
        run_summaries.append(run_summary)

    parameter_summaries = pd.concat(parameter_summaries, axis=1).T
    run_summaries = pd.concat(run_summaries, axis=1).T
    summary = pd.concat([run_summaries, parameter_summaries], axis=1)

    rename_map = {t: (f'{t[0]}_{t[1]}' if isinstance(t, tuple) else t) for t in summary.columns}
    return summary.rename(columns=rename_map)
