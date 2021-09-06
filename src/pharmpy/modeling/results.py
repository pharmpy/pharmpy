import warnings

import numpy as np
import pandas as pd
import sympy

from pharmpy.data_structures import OrderedSet
from pharmpy.parameter_sampling import sample_from_covariance_matrix


def calculate_eta_shrinkage(model, sd=False):
    """Calculate eta shrinkage for each eta

    Variance = False to get sd scale
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


def calculate_individual_parameter_statistics(model, exprs, seed=None):
    """Calculate statistics for individual parameters

    Calculate the mean (expected value of the distribution), variance
    (variance of the distribution) and standard error for individual
    parameters described by arbitrary expressions. Any dataset column or
    variable used in the model can be used in the expression. The exception
    being that variables that depends on the solution of the ODE system
    cannot be used. If covariates are used in the expression the statistics
    of the parameter is calculated at the median value of each covariate as well
    as at the 5:th and 95:th percentiles.

    Parameters
    ----------
    model : Model
        A previously estimated model
    exprs : str, sympy expression or iterable of str or sympy expressions
        Expressions or equations for parameters of interest. If equations are used
        the names of the left hand sides will be used as the names of the parameters.
    seed : int or numpy rng

    Returns
    -------
    pd.DataFrame
        A DataFrame of statistics indexed on parameter and covariate value.

    """
    if seed is None or isinstance(seed, int):
        seed = np.random.default_rng(seed)

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
            samples = model.random_variables.sample(expr, parameters=pe, samples=1000000, seed=seed)

            mean = np.mean(samples)
            variance = np.var(samples)

            parameters = sample_from_covariance_matrix(
                model,
                n=100,
                force_posdef_covmatrix=True,
                seed=seed,
            )
            samples = []
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                for _, row in parameters.iterrows():
                    batch = model.random_variables.sample(
                        cov_expr.subs(dict(row)),
                        parameters=dict(row),
                        samples=10,
                        seed=seed,
                    )
                    samples.extend(list(batch))
            stderr = pd.Series(samples).std()
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


def calculate_pk_parameters_statistics(model, seed=None):
    """Calculate statistics for common pharmacokinetic parameters

    Calculate the mean (expected value of the distribution), variance
    (variance of the distribution) and standard error for some individual
    pre-defined pharmacokinetic parameters.

    Parameters
    ----------
    model : Model
        A previously estimated model
    seed : int or numpy rng

    Returns
    -------
    pd.DataFrame
        A DataFrame of statistics indexed on parameter and covariate value.

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

    df = calculate_individual_parameter_statistics(model, expressions, seed=seed)
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
    """Summarize results of multiple model runs, includes runtime, ofv and parameter estimates
    (with errors).

    Parameters
    ----------
    models : list
        List of models
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
