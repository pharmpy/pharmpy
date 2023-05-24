from __future__ import annotations

import math
from itertools import chain
from typing import Iterable, Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.expr.parse import parse as parse_expr
from pharmpy.internals.expr.subs import subs, xreplace_dict
from pharmpy.internals.math import round_to_n_sigdig
from pharmpy.model import CompartmentalSystem, CompartmentalSystemBuilder, Model, output
from pharmpy.model.distributions.numeric import ConstantDistribution
from pharmpy.model.random_variables import (
    eval_expr,
    filter_distributions,
    sample_rvs,
    subs_distributions,
)

from .data import get_ids, get_observations
from .odes import get_initial_conditions
from .parameter_sampling import create_rng, sample_parameters_from_covariance_matrix

RANK_TYPES = frozenset(('ofv', 'lrt', 'aic', 'bic'))


def calculate_eta_shrinkage(
    model: Model,
    parameter_estimates: pd.Series,
    individual_estimates: pd.DataFrame,
    sd: bool = False,
):
    """Calculate eta shrinkage for each eta

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_estimates : pd.Series
        Parameter estimates
    individual_estimates : pd.DataFrame
        Table of individual (eta) estimates
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> pe = results.parameter_estimates
    >>> ie = results.individual_estimates
    >>> calculate_eta_shrinkage(model, pe, ie)
    ETA_1    0.720481
    ETA_2    0.240295
    dtype: float64
    >>> calculate_eta_shrinkage(model, pe, ie, sd=True)
    ETA_1    0.471305
    ETA_2    0.128389
    dtype: float64

    See also
    --------
    calculate_individual_shrinkage

    """
    # Want parameter estimates combined with fixed parameter values
    param_inits = model.parameters.to_dataframe()['value']
    pe = parameter_estimates.combine_first(param_inits)

    param_names = model.random_variables.iiv.variance_parameters
    diag_ests = pe[param_names]
    diag_ests.index = individual_estimates.columns
    if not sd:
        shrinkage = 1 - (individual_estimates.var() / diag_ests)
    else:
        shrinkage = 1 - (individual_estimates.std() / (diag_ests**0.5))
    return shrinkage


def calculate_individual_shrinkage(
    model: Model, parameter_estimates: pd.Series, individual_estimates_covariance: pd.DataFrame
):
    """Calculate the individual eta-shrinkage

    Definition: ieta_shr = (var(eta) / omega)

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameter_estimates : pd.Series
        Parameter estimates of model
    individual_estimates_covariance : pd.DataFrame
        Uncertainty covariance matrices of individual estimates

    Return
    ------
    DataFrame
        Shrinkage for each eta and individual

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> pe = results.parameter_estimates
    >>> covs = results.individual_estimates_covariance
    >>> calculate_individual_shrinkage(model, pe, covs)
           ETA_1     ETA_2
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
    cov = individual_estimates_covariance
    pe = parameter_estimates
    # Want parameter estimates combined with fixed parameter values
    param_inits = model.parameters.to_dataframe()['value']
    pe = pe.combine_first(param_inits)

    # Get all iiv and iov variance parameters
    diag = model.random_variables.etas.covariance_matrix.diagonal()
    param_names = [s.name for s in diag]

    diag_ests = pe[param_names]

    def fn(row, ests):
        names = row[0].index
        ser = pd.Series(np.diag(row[0].values) / ests, index=names)
        return ser

    ish = pd.DataFrame(cov).apply(fn, axis=1, ests=diag_ests.values)
    return ish


def calculate_individual_parameter_statistics(
    model: Model,
    expr_or_exprs: Union[
        Iterable[sympy.Eq], Iterable[sympy.Expr], Iterable[str], sympy.Eq, sympy.Expr, str
    ],
    parameter_estimates: pd.Series,
    covariance_matrix: Optional[pd.DataFrame] = None,
    rng: Optional[Union[np.random.Generator, int]] = None,
):
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
    parameter_estimates : pd.Series
        Parameter estimates
    covariance_matrix : pd.DataFrame
        Parameter uncertainty covariance matrix
    expr_or_exprs : str
        sympy expression or iterable of str or sympy expressions
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> rng = create_rng(23)
    >>> pe = results.parameter_estimates
    >>> cov = results.covariance_matrix
    >>> calculate_individual_parameter_statistics(model, "K=CL/V", pe, cov, rng=rng)
                              mean  variance    stderr
    parameter covariates
    K         p5          0.004234  0.000001  0.001138
              median      0.004907  0.000001  0.001247
              p95         0.004907  0.000001  0.001247
    """
    rng = create_rng(rng)

    split_exprs = map(
        _split_equation,
        [expr_or_exprs]
        if isinstance(expr_or_exprs, str) or isinstance(expr_or_exprs, sympy.Basic)
        else expr_or_exprs,
    )

    full_exprs = list(
        map(
            lambda e: (e[0], model.statements.before_odes.full_expression(e[1])),
            split_exprs,
        )
    )

    input_parameter_estimates = parameter_estimates
    parameter_estimates = xreplace_dict(parameter_estimates)

    all_free_symbols = set().union(*map(lambda e: e[1].free_symbols, full_exprs))

    all_covariate_free_symbols = all_free_symbols.intersection(
        map(sympy.Symbol, model.datainfo.names)
    )
    all_parameter_free_symbols = set(parameter_estimates.keys())
    all_random_free_symbols = all_free_symbols.difference(
        all_parameter_free_symbols, all_covariate_free_symbols
    )

    distributions = list(
        filter_distributions(
            model.random_variables,
            all_random_free_symbols,
        )
    )

    sampling_rvs = list(
        subs_distributions(
            distributions,
            parameter_estimates,
        )
    )

    batches = []

    if not all_covariate_free_symbols:
        cases = {'median': {}}
    else:
        dataset = model.dataset
        column_filter = ['ID'] + list(symbol.name for symbol in all_covariate_free_symbols)
        q5 = dataset[column_filter].groupby('ID').median().quantile(0.05)
        q95 = dataset[column_filter].groupby('ID').median().quantile(0.95)
        median = dataset[column_filter].groupby('ID').median().median()
        cases = {
            'p5': xreplace_dict(q5),
            'median': xreplace_dict(median),
            'p95': xreplace_dict(q95),
        }

    filtered_sampling_rvs = list(
        filter(
            lambda r: any(map(all_random_free_symbols.__contains__, r[0])),
            sampling_rvs,
        )
    )

    nsamples = 1000000
    nbatches = 100
    batchsize = 10

    samples = sample_rvs(filtered_sampling_rvs, nsamples, rng)

    if covariance_matrix is not None:
        parameters_samples = sample_parameters_from_covariance_matrix(
            model,
            input_parameter_estimates,
            covariance_matrix,
            n=nbatches,
            force_posdef_covmatrix=True,
            rng=rng,
        )

        for _, row in parameters_samples.iterrows():
            parameters = xreplace_dict(row)
            local_sampling_rvs = list(subs_distributions(distributions, parameters)) + [
                ((key,), ConstantDistribution(value))
                for key, value in parameters.items()
                if key in all_parameter_free_symbols
            ]
            batch = sample_rvs(local_sampling_rvs, batchsize, rng)
            batches.append(batch)

    table = pd.DataFrame(columns=['parameter', 'covariates', 'mean', 'variance', 'stderr'])
    i = 0

    for name, full_expr in full_exprs:
        df = pd.DataFrame(index=list(cases.keys()), columns=['mean', 'variance', 'stderr'])
        parameter_estimates_expr = subs(full_expr, parameter_estimates, simultaneous=True)

        for case, cov_values in cases.items():
            expr = subs(parameter_estimates_expr, cov_values, simultaneous=True)
            values = eval_expr(expr, nsamples, samples)

            mean = np.mean(values)
            variance = np.var(values)

            # NOTE This is NaN for empty inputs, dtype is required for those.
            cov_expr = subs(full_expr, cov_values, simultaneous=True)
            stderr = pd.Series(
                chain.from_iterable(eval_expr(cov_expr, batchsize, batch) for batch in batches),
                dtype='float64',
            ).std()

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


def calculate_pk_parameters_statistics(
    model: Model,
    parameter_estimates: pd.Series,
    covariance_matrix: Optional[pd.DataFrame] = None,
    rng: Optional[Union[np.random.Generator, int]] = None,
):
    """Calculate statistics for common pharmacokinetic parameters

    Calculate the mean (expected value of the distribution), variance
    (variance of the distribution) and standard error for some individual
    pre-defined pharmacokinetic parameters.

    Parameters
    ----------
    model : Model
        A previously estimated model
    parameter_estimates : pd.Series
        Parameter estimates
    covariance_matrix : pd.DataFrame
        Parameter uncertainty covariance matrix
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> rng = create_rng(23)
    >>> pe = results.parameter_estimates
    >>> cov = results.covariance_matrix
    >>> calculate_pk_parameters_statistics(model, pe, cov, rng=rng)
                                  mean     variance     stderr
    parameter   covariates
    t_half_elim p5          173.337164  1769.493756  42.843398
                median      149.567842  1317.474199  36.233070
                p95         149.567842  1317.474199  36.233070
    k_e         p5            0.004234     0.000001   0.001138
                median        0.004907     0.000001   0.001247
                p95           0.004907     0.000001   0.001247


    See Also
    --------
    calculate_individual_parameter_statistics : Calculation of statistics for arbitrary parameters

    """

    statements = model.statements
    odes = statements.ode_system
    central = odes.central_compartment
    depot = odes.find_depot(statements)
    peripherals = odes.peripheral_compartments
    elimination_rate = odes.get_flow(central, output)

    expressions = []  # Eq(name, expr)
    # FO abs + 1comp + FO elimination
    if len(odes) == 2 and depot and odes.t not in elimination_rate.free_symbols:
        ode_list, ics = odes.eqs, get_initial_conditions(model, dosing=True)
        sols = sympy.dsolve(ode_list, ics=ics)
        expr = sols[1].rhs
        d = sympy.diff(expr, odes.t)
        tmax_closed_form = sympy.solve(d, odes.t)[0]
        expressions.append(sympy.Eq(sympy.Symbol('t_max'), tmax_closed_form))
        e2 = sympy.simplify(expr / depot.dose.amount / sympy.denom(elimination_rate))
        cmax_dose_closed_form = sympy.simplify(
            subs(e2, {odes.t: tmax_closed_form}, simultaneous=True)
        )
        expressions.append(sympy.Eq(sympy.Symbol('C_max_dose'), cmax_dose_closed_form))

    # Any abs + 1comp + FO elimination
    if not peripherals and odes.t not in elimination_rate.free_symbols:
        elimination_system = statements.ode_system
        for name in elimination_system.compartment_names:
            if name != central.name:  # NOTE keep central
                cb = CompartmentalSystemBuilder(elimination_system)
                cb.remove_compartment(elimination_system.find_compartment(name))
                elimination_system = CompartmentalSystem(cb)
        eq = elimination_system.eqs[0]
        ic = sympy.Function(elimination_system.amounts[0].name)(0)
        A0 = sympy.Symbol('A0')
        sols = sympy.dsolve(eq, ics={ic: A0})
        eq_half = sympy.Eq(sympy.Rational(1, 2) * A0, sols.rhs)
        thalf_elim = sympy.solve(eq_half, odes.t)[0]
        expressions.append(sympy.Eq(sympy.Symbol('t_half_elim'), thalf_elim))

    # Bolus dose + 2comp + FO elimination
    if len(peripherals) == 1 and len(odes) == 2 and odes.t not in elimination_rate.free_symbols:
        ode_list, ics = odes.eqs, get_initial_conditions(model, dosing=True)
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

        if (beta - alpha).extract_multiplicatively(-1) is not None:
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

    df = calculate_individual_parameter_statistics(
        model, expressions, parameter_estimates, covariance_matrix, rng=rng
    )
    return df


def _split_equation(s):
    if isinstance(s, str):
        a = s.split('=')
        if len(a) == 1:
            name = None
            expr = parse_expr(s)
        else:
            name = a[0].strip()
            expr = parse_expr(a[1])
    elif isinstance(s, sympy.Eq):
        name = s.lhs.name
        expr = s.rhs
    else:  # sympy expr
        name = None
        expr = s
    if name is None and isinstance(expr, sympy.Symbol):
        name = expr.name
    return name, expr


def calculate_aic(model: Model, likelihood: float):
    """Calculate AIC

    AIC = -2LL + 2*n_estimated_parameters

    Parameters
    ----------
    model : Model
        Pharmpy model object
    likelihood : float
        -2LL

    Returns
    -------
    float
        AIC of model fit
    """
    parameters = model.parameters.nonfixed
    return likelihood + 2 * len(parameters)


def _random_etas(model):
    var = model.random_variables.etas.variance_parameters
    zerofix = [model.parameters[e].fix and model.parameters[e].init == 0 for e in var]
    keep = []
    for eta, zf in zip(model.random_variables.etas.names, zerofix):
        if not zf:
            keep.append(eta)
    return model.random_variables.etas[keep]


def calculate_bic(model: Model, likelihood: float, type: Optional[str] = None):
    """Calculate BIC

    Different variations of the BIC can be calculated:

    * | mixed (default)
      | BIC = -2LL + n_random_parameters * log(n_individuals) +
      |       n_fixed_parameters * log(n_observations)
    * | fixed
      | BIC = -2LL + n_estimated_parameters * log(n_observations)
    * | random
      | BIC = -2LL + n_estimated_parameters * log(n_individals)
    * | iiv
      | BIC = -2LL + n_estimated_iiv_omega_parameters * log(n_individals)

    Parameters
    ----------
    model : Model
        Pharmpy model object
    likelihood : float
        -2LL to use
    type : str
        Type of BIC to calculate. Default is the mixed effects.

    Returns
    -------
    float
        BIC of model fit

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> ofv = results.ofv
    >>> calculate_bic(model, ofv)
    611.7071686183284
    >>> calculate_bic(model, ofv, type='fixed')
    616.536606983396
    >>> calculate_bic(model, ofv, type='random')
    610.7412809453149
    >>> calculate_bic(model, ofv, type='iiv')
    594.431131169692
    """
    parameters = model.parameters.nonfixed
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
                for eta in _random_etas(model).names:
                    if sympy.Symbol(eta) in expr.free_symbols:
                        symbols = {p.symbol for p in parameters if p.symbol in expr.free_symbols}
                        random_thetas.update(symbols)
                        break
        # FIXME: handle other DVs?
        dv = list(model.dependent_variables.keys())[0]
        yexpr = model.statements.after_odes.full_expression(dv)
        for eta in _random_etas(model).names:
            if sympy.Symbol(eta) in yexpr.free_symbols:
                symbols = {p.symbol for p in parameters if p.symbol in yexpr.free_symbols}
                random_thetas.update(symbols)
                for eps in model.random_variables.epsilons.names:
                    if sympy.Symbol(eps) in yexpr.free_symbols:
                        params = {
                            p.symbol
                            for p in parameters
                            if p.name in model.random_variables[eps].parameter_names
                        }
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
    return likelihood + penalty


def check_high_correlations(model: Model, cor: pd.DataFrame, limit: float = 0.9):
    """Check for highly correlated parameter estimates

    Parameters
    ----------
    model : Model
        Pharmpy model object
    cor : pd.DataFrame
        Estimated correlation matrix
    limit : float
        Lower limit for a high correlation

    Returns
    -------
    pd.Series
        Correlation values indexed on pairs of parameters for (absolute) correlations above limit

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> cor = results.correlation_matrix
    >>> check_high_correlations(model, cor, limit=0.3)
    PTVCL  IVCL      -0.388059
    PTVV   THETA_3   -0.356899
           IVV        0.356662
    dtype: float64
    """
    high_and_below_diagonal = cor.abs().ge(limit) & np.triu(np.ones(cor.shape), k=1).astype(bool)
    return cor.where(high_and_below_diagonal).stack()


def check_parameters_near_bounds(
    model: Model, values: pd.Series, zero_limit: float = 0.001, significant_digits: int = 2
):
    """Check if any estimated parameter value is close to its bounds

    Parameters
    ----------
    model : Model
        Pharmpy model object
    values : pd.Series
        Series of values with index a subset of parameter names.
    zero_limit : float
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> check_parameters_near_bounds(model, results.parameter_estimates)
    PTVCL        False
    PTVV         False
    THETA_3      False
    IVCL         False
    IVV          False
    SIGMA_1_1    False
    dtype: bool

    """
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
