import warnings

import numpy as np
import pandas as pd
import sympy

from pharmpy.parameter_sampling import sample_from_covariance_matrix


def calculate_individual_parameter_statistics(model, exprs, seed=None):
    """Calculate statistics for individual parameters

    exprs - is one string or an iterable of strings

    The parameter does not have to be in the model, but can be an
    expression of other parameters from the model.
    Does not support parameters that relies on the solution of the ODE-system
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
    return table


def calculate_pk_parameters_statistics(model, seed=None):
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
