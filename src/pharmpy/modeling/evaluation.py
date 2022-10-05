from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.deps.scipy import linalg
from pharmpy.expressions import subs, sympify

from .expressions import (
    calculate_epsilon_gradient_expression,
    calculate_eta_gradient_expression,
    get_individual_prediction_expression,
    get_population_prediction_expression,
)


def evaluate_expression(model, expression, parameter_estimates=None):
    """Evaluate expression using model

    Calculate the value of expression for each data record.
    The expression can contain dataset columns, variables in model and
    population parameters. If the model has parameter estimates these
    will be used. Initial estimates will be used for non-estimated parameters.

    Parameters
    ----------
    model : Model
        Pharmpy model
    expression : str or sympy expression
        Expression to evaluate
    parameter_estimates : pd.Series
        Parameter estimates to use instead of initial estimates

    Returns
    -------
    pd.Series
        A series of one evaluated value for each data record

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, evaluate_expression
    >>> model = load_example_model("pheno")
    >>> pe = model.modelfit_results.parameter_estimates
    >>> evaluate_expression(model, "TVCL*1000", parameter_estimates=pe)
    0      6.573770
    1      6.573770
    2      6.573770
    3      6.573770
    4      6.573770
             ...
    739    5.165105
    740    5.165105
    741    5.165105
    742    5.165105
    743    5.165105
    Length: 744, dtype: float64

    """
    expression = sympify(expression)
    full_expr = model.statements.before_odes.full_expression(expression)
    inits = model.parameters.inits
    expr = subs(subs(full_expr, dict(parameter_estimates)), inits)
    data = model.dataset

    def func(row):
        value = subs(expr, dict(row))
        return np.float64(value.evalf())

    df = data.apply(func, axis=1)
    return df


def evaluate_population_prediction(model, parameters=None, dataset=None):
    """Evaluate the numeric population prediction

    The prediction is evaluated at the current model parameter values
    or optionally at the given parameter values.
    The evaluation is done for each data record in the model dataset
    or optionally using the dataset argument.

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameters : dict
        Optional dictionary of parameters and values
    dataset : pd.DataFrame
        Optional dataset

    Returns
    -------
    pd.Series
        Population predictions

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, evaluate_population_prediction
    >>> model = load_example_model("pheno_linear")
    >>> pe = model.modelfit_results.parameter_estimates
    >>> evaluate_population_prediction(model, parameters=dict(pe))
    0      17.529739
    1      28.179910
    2       9.688648
    3      17.798916
    4      25.023225
             ...
    150    22.459036
    151    29.223295
    152    20.217288
    153    28.472888
    154    34.226455
    Name: PRED, Length: 155, dtype: float64

    See also
    --------
    evaluate_individual_prediction : Evaluate the individual prediction
    """
    y = get_population_prediction_expression(model)

    if parameters is not None:
        y = subs(y, parameters)
    else:
        y = subs(y, model.parameters.inits)

    if dataset is not None:
        df = dataset
    else:
        df = model.dataset

    pred = df.apply(lambda row: np.float64(subs(y, row.to_dict())), axis=1)
    pred.name = 'PRED'
    return pred


def evaluate_individual_prediction(model, etas=None, parameters=None, dataset=None):
    """Evaluate the numeric individual prediction

    The prediction is evaluated at the current model parameter values
    or optionally at the given parameter values.
    The evaluation is done for each data record in the model dataset
    or optionally using the dataset argument.
    The evaluation is done at the current eta values
    or optionally at the given eta values.

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model
    etas : dict
        Optional dictionary of eta values
    parameters : dict
        Optional dictionary of parameters and values
    dataset : pd.DataFrame
        Optional dataset

    Returns
    -------
    pd.Series
        Individual predictions

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, evaluate_individual_prediction
    >>> model = load_example_model("pheno_linear")
    >>> etas = model.modelfit_results.individual_estimates
    >>> evaluate_individual_prediction(model, etas=etas)
    0      17.771084
    1      28.881859
    2      11.441728
    3      21.113050
    4      29.783055
             ...
    150    25.375041
    151    31.833395
    152    22.876707
    153    31.905095
    154    38.099690
    Name: IPRED, Length: 155, dtype: float64

    See also
    --------
    evaluate_population_prediction : Evaluate the population prediction
    """

    y = get_individual_prediction_expression(model)
    if parameters is not None:
        y = subs(y, parameters)
    else:
        y = subs(y, model.parameters.inits)

    if dataset is not None:
        df = dataset
    else:
        df = model.dataset

    idcol = model.datainfo.id_column.name

    if etas is None:
        etas = pd.DataFrame(
            0,
            index=df[idcol].unique(),
            columns=model.random_variables.etas.names,
        )

    def fn(row):
        row = row.to_dict()
        curetas = etas.loc[row[idcol]].to_dict()
        a = np.float64(subs(subs(y, row), curetas))
        return a

    ipred = df.apply(fn, axis=1)
    ipred.name = 'IPRED'
    return ipred


def _replace_parameters(model, y, parameters):
    if parameters is not None:
        y = [subs(x, parameters) for x in y]
    else:
        y = [subs(x, model.parameters.inits) for x in y]
    return y


def evaluate_eta_gradient(model, etas=None, parameters=None, dataset=None):
    """Evaluate the numeric eta gradient

    The gradient is evaluated at the current model parameter values
    or optionally at the given parameter values.
    The gradient is done for each data record in the model dataset
    or optionally using the dataset argument.
    The gradient is done at the current eta values
    or optionally at the given eta values.

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model
    etas : dict
        Optional dictionary of eta values
    parameters : dict
        Optional dictionary of parameters and values
    dataset : pd.DataFrame
        Optional dataset

    Returns
    -------
    pd.DataFrame
        Gradient

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, evaluate_eta_gradient
    >>> model = load_example_model("pheno_linear")
    >>> etas = model.modelfit_results.individual_estimates
    >>> evaluate_eta_gradient(model, etas=etas)
         dF/dETA(1)  dF/dETA(2)
    0     -0.159537  -17.609116
    1     -9.325893  -19.562289
    2     -0.104417  -11.346161
    3     -4.452951  -16.682310
    4    -10.838840  -18.981836
    ..          ...         ...
    150   -5.424423  -19.973013
    151  -14.497185  -17.344797
    152   -0.198714  -22.697161
    153   -7.987731  -23.941806
    154  -15.817067  -22.309945
    <BLANKLINE>
    [155 rows x 2 columns]

    See also
    --------
    evaluate_epsilon_gradient : Evaluate the epsilon gradient
    """

    y = calculate_eta_gradient_expression(model)
    y = _replace_parameters(model, y, parameters)

    if dataset is not None:
        df = dataset
    else:
        df = model.dataset
    idcol = model.datainfo.id_column.name

    if etas is None:
        if model.initial_individual_estimates is not None:
            etas = model.initial_individual_estimates
        else:
            etas = pd.DataFrame(
                0,
                index=df[idcol].unique(),
                columns=model.random_variables.etas.names,
            )

    def fn(row):
        row = row.to_dict()
        curetas = etas.loc[row[idcol]].to_dict()
        a = [np.float64(subs(subs(x, row), curetas)) for x in y]
        return a

    derivative_names = [f'dF/d{eta}' for eta in model.random_variables.etas.names]
    grad = df.apply(fn, axis=1, result_type='expand')
    grad = pd.DataFrame(grad)
    grad.columns = derivative_names
    return grad


def evaluate_epsilon_gradient(model, etas=None, parameters=None, dataset=None):
    """Evaluate the numeric epsilon gradient

    The gradient is evaluated at the current model parameter values
    or optionally at the given parameter values.
    The gradient is done for each data record in the model dataset
    or optionally using the dataset argument.
    The gradient is done at the current eta values
    or optionally at the given eta values.

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model
    etas : dict
        Optional dictionary of eta values
    parameters : dict
        Optional dictionary of parameters and values
    dataset : pd.DataFrame
        Optional dataset

    Returns
    -------
    pd.DataFrame
        Gradient

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, evaluate_epsilon_gradient
    >>> model = load_example_model("pheno_linear")
    >>> etas = model.modelfit_results.individual_estimates
    >>> evaluate_epsilon_gradient(model, etas=etas)
         dY/dEPS(1)
    0     17.771084
    1     28.881859
    2     11.441728
    3     21.113050
    4     29.783055
    ..          ...
    150   25.375041
    151   31.833395
    152   22.876707
    153   31.905095
    154   38.099690
    <BLANKLINE>
    [155 rows x 1 columns]

    See also
    --------
    evaluate_eta_gradient : Evaluate the eta gradient
    """

    y = calculate_epsilon_gradient_expression(model)
    y = _replace_parameters(model, y, parameters)
    eps_names = model.random_variables.epsilons.names
    repl = {sympy.Symbol(eps): 0 for eps in eps_names}
    y = [subs(x, repl) for x in y]

    if dataset is not None:
        df = dataset
    else:
        df = model.dataset

    idcol = model.datainfo.id_column.name

    if etas is None:
        if model.initial_individual_estimates is not None:
            etas = model.initial_individual_estimates
        else:
            etas = pd.DataFrame(
                0,
                index=df[idcol].unique(),
                columns=model.random_variables.etas.names,
            )

    def fn(row):
        row = row.to_dict()
        curetas = etas.loc[row[idcol]].to_dict()
        a = [np.float64(subs(subs(x, row), curetas)) for x in y]
        return a

    grad = df.apply(fn, axis=1, result_type='expand')
    derivative_names = [f'dY/d{eps}' for eps in eps_names]
    grad = pd.DataFrame(grad)
    grad.columns = derivative_names
    return grad


def evaluate_weighted_residuals(model, parameters=None, dataset=None):
    """Evaluate the weighted residuals

    The residuals is evaluated at the current model parameter values
    or optionally at the given parameter values.
    The residuals is done for each data record in the model dataset
    or optionally using the dataset argument.

    This function currently only support models without ODE systems

    Parameters
    ----------
    model : Model
        Pharmpy model
    parameters : dict
        Optional dictionary of parameters and values
    dataset : pd.DataFrame
        Optional dataset

    Returns
    -------
    pd.Series
        WRES

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, evaluate_weighted_residuals
    >>> model = load_example_model("pheno_linear")
    >>> parameters = model.modelfit_results.parameter_estimates
    >>> evaluate_weighted_residuals(model, parameters=dict(parameters))
    0     -0.313859
    1      0.675721
    2     -1.544240
    3      1.921720
    4      1.517677
            ...
    150    1.223935
    151   -0.053334
    152   -0.007023
    153    0.931252
    154    0.778389
    Name: WRES, Length: 155, dtype: float64
    """

    omega = model.random_variables.etas.covariance_matrix
    sigma = model.random_variables.epsilons.covariance_matrix
    if parameters is None:
        parameters = model.parameters.inits
    omega = omega.subs(parameters)
    sigma = sigma.subs(parameters)
    omega = np.float64(omega)
    sigma = np.float64(sigma)
    if dataset is not None:
        df = dataset
    else:
        df = model.dataset
    # FIXME: Could have option to gradients to set all etas 0
    etas = pd.DataFrame(
        0,
        index=df[model.datainfo.id_column.name].unique(),
        columns=model.random_variables.etas.names,
    )
    G = evaluate_eta_gradient(model, etas=etas, parameters=parameters, dataset=dataset)
    H = evaluate_epsilon_gradient(model, etas=etas, parameters=parameters, dataset=dataset)
    F = evaluate_population_prediction(model)
    index = df[model.datainfo.id_column.name]
    G.index = index
    H.index = index
    F.index = index
    WRES = np.float64([])
    for i in df[model.datainfo.id_column.name].unique():
        Gi = np.float64(G.loc[[i]])
        Hi = np.float64(H.loc[[i]])
        Fi = F.loc[i:i]
        DVi = np.float64(df['DV'][df[model.datainfo.id_column.name] == i])
        Ci = Gi @ omega @ Gi.T + np.diag(np.diag(Hi @ sigma @ Hi.T))
        WRESi = linalg.sqrtm(linalg.inv(Ci)) @ (DVi - Fi)
        WRES = np.concatenate((WRES, WRESi))
    ser = pd.Series(WRES)
    ser.name = 'WRES'
    return ser
