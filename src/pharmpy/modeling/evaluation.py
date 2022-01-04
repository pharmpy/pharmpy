import numpy as np
import pandas as pd
import symengine
import sympy

from .expressions import get_individual_prediction_expression, get_population_prediction_expression


def evaluate_expression(model, expression):
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

    Returns
    -------
    pd.Series
        A series of one evaluated value for each data record

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, evaluate_expression
    >>> model = load_example_model("pheno")
    >>> evaluate_expression(model, "TVCL*1000")
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
    expression = sympy.sympify(expression)
    full_expr = model.statements.before_odes.full_expression(expression)
    pe = model.modelfit_results.parameter_estimates
    inits = model.parameters.inits
    expr = full_expr.subs(dict(pe)).subs(inits)
    data = model.dataset
    expr = symengine.sympify(expr)

    def func(row):
        subs = expr.subs(dict(row))
        return np.float64(subs.evalf())

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
    >>> evaluate_population_prediction(model)
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
        y = y.subs(parameters)
    else:
        if model.modelfit_results is not None:
            y = y.subs(model.modelfit_results.parameter_estimates.to_dict())
        else:
            y = y.subs(model.parameters.inits)

    if dataset is not None:
        df = dataset
    else:
        df = model.dataset

    pred = df.apply(lambda row: np.float64(y.subs(row.to_dict())), axis=1)
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
    >>> evaluate_individual_prediction(model)
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
        y = y.subs(parameters)
    else:
        y = y.subs(model.parameters.inits)

    if dataset is not None:
        df = dataset
    else:
        df = model.dataset

    idcol = model.datainfo.id_label

    if etas is None:
        if (
            model.modelfit_results is not None
            and model.modelfit_results.individual_estimates is not None
        ):
            etas = model.modelfit_results.individual_estimates
        elif model.initial_individual_estimates is not None:
            etas = model.initial_individual_estimates
        else:
            etas = pd.DataFrame(
                0,
                index=df[idcol].unique(),
                columns=[eta.name for eta in model.random_variables.etas],
            )

    def fn(row):
        row = row.to_dict()
        curetas = etas.loc[row[idcol]].to_dict()
        a = np.float64(y.subs(row).subs(curetas))
        return a

    ipred = df.apply(fn, axis=1)
    ipred.name = 'IPRED'
    return ipred
