from __future__ import annotations

from collections.abc import Mapping
from typing import Optional, Union

from pharmpy.basic import Expr, TExpr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.deps.scipy import linalg
from pharmpy.internals.expr.eval import eval_expr
from pharmpy.model import Model, get_and_check_dataset

from .expressions import (
    calculate_epsilon_gradient_expression,
    calculate_eta_gradient_expression,
    get_individual_prediction_expression,
    get_population_prediction_expression,
)

ParameterMap = Mapping[Union[str], Union[float, 'sympy.Float']]


class DataFrameMapping(Mapping['sympy.Expr', 'np.ndarray']):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __getitem__(self, symbol: sympy.Expr):
        assert isinstance(symbol, sympy.Symbol)
        col = self._df[symbol.name]
        assert col is not None
        return col.to_numpy()

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        return map(sympy.Symbol, self._df.columns)


def evaluate_expression(
    model: Model,
    expression: Union[str, TExpr],
    parameter_estimates: Optional[ParameterMap] = None,
):
    """Evaluate expression using model

    Calculate the value of expression for each data record.
    The expression can contain dataset columns, variables in model and
    population parameters. If the model has parameter estimates these
    will be used. Initial estimates will be used for non-estimated parameters.

    Parameters
    ----------
    model : Model
        Pharmpy model
    expression : str or TExpr
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno")
    >>> results = load_example_modelfit_results("pheno")
    >>> pe = results.parameter_estimates
    >>> evaluate_expression(model, "TVCL*1000", parameter_estimates=pe)
    1      6.573770
    2      6.573770
    3      6.573770
    4      6.573770
    5      6.573770
             ...
    740    5.165105
    741    5.165105
    742    5.165105
    743    5.165105
    744    5.165105
    Length: 744, dtype: float64

    """
    expression = Expr(expression)
    full_expr = model.statements.before_odes.full_expression(expression)
    inits = model.parameters.inits
    mapping = inits if parameter_estimates is None else {**inits, **parameter_estimates}
    expr = full_expr.subs(mapping)

    df = get_and_check_dataset(model)

    array = eval_expr(expr._sympy_(), len(df), DataFrameMapping(df))
    return pd.Series(array, df.index)


def evaluate_population_prediction(
    model: Model, parameters: Optional[ParameterMap] = None, dataset: Optional[pd.DataFrame] = None
):
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno_linear")
    >>> results = load_example_modelfit_results("pheno_linear")
    >>> pe = results.parameter_estimates
    >>> evaluate_population_prediction(model, parameters=dict(pe))
    1      17.529739
    2      28.179910
    3       9.688648
    4      17.798916
    5      25.023225
             ...
    151    22.459036
    152    29.223295
    153    20.217288
    154    28.472888
    155    34.226455
    Name: PRED, Length: 155, dtype: float64

    See also
    --------
    evaluate_individual_prediction : Evaluate the individual prediction
    """
    y = get_population_prediction_expression(model)
    mapping = model.parameters.inits if parameters is None else parameters
    expr = y.subs(mapping)

    df = get_and_check_dataset(model) if dataset is None else dataset

    pred = eval_expr(sympy.sympify(expr), len(df), DataFrameMapping(df))
    pred_series = pd.Series(pred, name='PRED', index=df.index)
    return pred_series


def evaluate_individual_prediction(
    model: Model,
    etas: Optional[pd.DataFrame] = None,
    parameters: Optional[ParameterMap] = None,
    dataset: Optional[pd.DataFrame] = None,
):
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno_linear")
    >>> results = load_example_modelfit_results("pheno_linear")
    >>> etas = results.individual_estimates
    >>> evaluate_individual_prediction(model, etas=etas)
    1      17.771084
    2      28.881859
    3      11.441728
    4      21.113050
    5      29.783055
             ...
    151    25.375041
    152    31.833395
    153    22.876707
    154    31.905095
    155    38.099690
    Name: IPRED, Length: 155, dtype: float64

    See also
    --------
    evaluate_population_prediction : Evaluate the population prediction
    """

    y = get_individual_prediction_expression(model)
    mapping = model.parameters.inits if parameters is None else parameters
    y = y.subs(mapping)

    df = get_and_check_dataset(model) if dataset is None else dataset

    idcol = model.datainfo.id_column.name

    _etas = (
        pd.DataFrame(
            0,
            index=df[idcol].unique(),
            columns=pd.Index(model.random_variables.etas.names),
        )
        if etas is None
        else etas
    )

    _df = df.join(_etas, on=idcol)

    ipred = eval_expr(sympy.sympify(y), len(_df), DataFrameMapping(_df))
    ipred_series = pd.Series(ipred, name='IPRED', index=df.index)
    return ipred_series


def _replace_parameters(model: Model, y: list[sympy.Expr], parameters: Optional[ParameterMap]):
    mapping = model.parameters.inits if parameters is None else parameters
    return [x.subs(mapping) for x in y]  # pyright: ignore [reportCallIssue, reportArgumentType]


def evaluate_eta_gradient(
    model: Model,
    etas: Optional[pd.DataFrame] = None,
    parameters: Optional[ParameterMap] = None,
    dataset: Optional[pd.DataFrame] = None,
):
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno_linear")
    >>> results = load_example_modelfit_results("pheno_linear")
    >>> etas = results.individual_estimates
    >>> evaluate_eta_gradient(model, etas=etas)
         dF/dETA_1  dF/dETA_2
    1     -0.159537 -17.609116
    2     -9.325893 -19.562289
    3     -0.104417 -11.346161
    4     -4.452951 -16.682310
    5    -10.838840 -18.981836
    ..          ...        ...
    151   -5.424423 -19.973013
    152  -14.497185 -17.344797
    153   -0.198714 -22.697161
    154   -7.987731 -23.941806
    155  -15.817067 -22.309945
    <BLANKLINE>
    [155 rows x 2 columns]

    See also
    --------
    evaluate_epsilon_gradient : Evaluate the epsilon gradient
    """

    y = calculate_eta_gradient_expression(model)
    y = _replace_parameters(model, y, parameters)

    df = get_and_check_dataset(model) if dataset is None else dataset
    idcol = model.datainfo.id_column.name

    if etas is not None:
        _etas = etas
    elif model.initial_individual_estimates is not None:
        _etas = model.initial_individual_estimates
    else:
        _etas = pd.DataFrame(
            0,
            index=df[idcol].unique(),
            columns=pd.Index(model.random_variables.etas.names),
        )

    derivative_names = [f'dF/d{eta}' for eta in model.random_variables.etas.names]

    _df = df.join(_etas, on=idcol)

    return pd.DataFrame(
        {
            name: eval_expr(sympy.sympify(expr), len(_df), DataFrameMapping(_df))
            for expr, name in zip(y, derivative_names)
        },
        index=df.index,
    )


def evaluate_epsilon_gradient(
    model: Model,
    etas: Optional[pd.DataFrame] = None,
    parameters: Optional[ParameterMap] = None,
    dataset: Optional[pd.DataFrame] = None,
):
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno_linear")
    >>> results = load_example_modelfit_results("pheno_linear")
    >>> etas = results.individual_estimates
    >>> evaluate_epsilon_gradient(model, etas=etas)
         dY/dEPS_1
    1     17.771084
    2     28.881859
    3     11.441728
    4     21.113050
    5     29.783055
    ..          ...
    151   25.375041
    152   31.833395
    153   22.876707
    154   31.905095
    155   38.099690
    <BLANKLINE>
    [155 rows x 1 columns]

    See also
    --------
    evaluate_eta_gradient : Evaluate the eta gradient
    """

    y = calculate_epsilon_gradient_expression(model)
    y = _replace_parameters(model, y, parameters)
    eps_names = model.random_variables.epsilons.names
    repl = {Expr.symbol(eps): 0 for eps in eps_names}
    y = [x.subs(repl) for x in y]

    df = get_and_check_dataset(model) if dataset is None else dataset

    idcol = model.datainfo.id_column.name

    if etas is not None:
        _etas = etas
    elif model.initial_individual_estimates is not None:
        _etas = model.initial_individual_estimates
    else:
        _etas = pd.DataFrame(
            0,
            index=df[idcol].unique(),
            columns=pd.Index(model.random_variables.etas.names),
        )

    _df = df.join(_etas, on=idcol)
    derivative_names = [f'dY/d{eps}' for eps in eps_names]

    return pd.DataFrame(
        {
            name: eval_expr(sympy.sympify(expr), len(_df), DataFrameMapping(_df))
            for expr, name in zip(y, derivative_names)
        },
        index=df.index,
    )


def evaluate_weighted_residuals(
    model: Model,
    parameters: Optional[ParameterMap] = None,
    dataset: Optional[pd.DataFrame] = None,
):
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
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> model = load_example_model("pheno_linear")
    >>> results = load_example_modelfit_results("pheno_linear")
    >>> parameters = results.parameter_estimates
    >>> evaluate_weighted_residuals(model, parameters=dict(parameters))
    1     -0.313859
    2      0.675721
    3     -1.544240
    4      1.921720
    5      1.517677
            ...
    151    1.223935
    152   -0.053334
    153   -0.007023
    154    0.931252
    155    0.778389
    Name: WRES, Length: 155, dtype: float64
    """

    omega = model.random_variables.etas.covariance_matrix
    sigma = model.random_variables.epsilons.covariance_matrix
    useparams = model.parameters.inits if parameters is None else parameters
    omega = omega.subs(useparams)
    sigma = sigma.subs(useparams)
    omega = omega.to_numpy()
    sigma = sigma.to_numpy()
    df = get_and_check_dataset(model) if dataset is None else dataset
    # FIXME: Could have option to gradients to set all etas 0
    etas = pd.DataFrame(
        0,
        index=df[model.datainfo.id_column.name].unique(),
        columns=pd.Index(model.random_variables.etas.names),
    )
    G = evaluate_eta_gradient(model, etas=etas, parameters=useparams, dataset=dataset)
    H = evaluate_epsilon_gradient(model, etas=etas, parameters=useparams, dataset=dataset)
    F = evaluate_population_prediction(model)
    index = df[model.datainfo.id_column.name]
    G.set_index(index, inplace=True)
    H.set_index(index, inplace=True)
    F.index = pd.Index(index)
    WRES = np.empty(0)
    for i in df[model.datainfo.id_column.name].unique():
        Gi = np.float64(G.loc[[i]])
        Hi = np.float64(H.loc[[i]])
        Fi = F.loc[i:i]
        DVi_df = (df['DV'][df[model.datainfo.id_column.name] == i]).astype(np.float64)
        DVi = DVi_df.values  # pyright: ignore [reportAttributeAccessIssue]
        Ci = Gi @ omega @ Gi.T + np.diag(np.diag(Hi @ sigma @ Hi.T))
        WRESi = linalg.sqrtm(linalg.inv(Ci)) @ (DVi - Fi)
        WRES = np.concatenate((WRES, WRESi))

    wres_series = pd.Series(WRES, name='WRES', index=df.index)
    return wres_series
