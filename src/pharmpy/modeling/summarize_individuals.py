import warnings
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd

from pharmpy.model import Model, ModelfitResultsError
from pharmpy.modeling.ml import predict_influential_individuals, predict_outliers


def summarize_individuals(models: List[Model]) -> Union[pd.DataFrame, None]:
    """Creates a summary dataframe keyed by model-individual pairs for an input
    list of models.

    Content of the various columns:

    +-------------------------+----------------------------------------------------------------------+
    | Column                  | Description                                                          |
    +=========================+======================================================================+
    | ``outlier_count``       | Number of observations with CWRES > 5                                |
    +-------------------------+----------------------------------------------------------------------+
    | ``ofv``                 | Individual OFV                                                       |
    +-------------------------+----------------------------------------------------------------------+
    | ``dofv_vs_parent``      | Difference in individual OFV between this model and its parent model |
    +-------------------------+----------------------------------------------------------------------+
    | ``predicted_dofv``      | Predicted dOFV if this individual was excluded                       |
    +-------------------------+----------------------------------------------------------------------+
    | ``predicted_residual``  | Predicted residual                                                   |
    +-------------------------+----------------------------------------------------------------------+

    Parameters
    ----------
    models : List[Model]
        Input models

    Return
    ------
    pd.DataFrame | None
        The summary as a dataframe

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> fit(model)
    <Pharmpy model object pheno>
    >>> results = run_tool(
    ...     'modelsearch',
    ...     model=model,
    ...     mfl='ABSORPTION(ZO);PERIPHERALS([1, 2])',
    ...     algorithm='reduced_stepwise'
    ... ) # doctest: +SKIP
    >>> summarize_individuals([results.start_model, *results.models]) # doctest: +SKIP

    """  # noqa: E501

    modelsDict = {model.name: model for model in models}

    df = pd.concat(
        map(
            lambda model: groupedByIDAddColumnsOneModel(modelsDict, model),
            models,
        ),
        keys=[model.name for model in models],
        names=['model'],
        axis=0,
        verify_integrity=True,
    )

    return df


def parent_model_name(model: Model) -> str:
    return model.parent_model


def model_name(model: Model) -> str:
    return model.name


def outlier_count_func(df: pd.DataFrame) -> float:
    # NOTE this returns a float because we will potentially concat this
    # with NaNs
    return float((abs(df) > 5).sum())


def outlier_count(model: Model) -> Union[pd.Series, float]:
    res = model.modelfit_results
    if res is None:
        return np.nan
    residuals = res.residuals
    if residuals is None:
        return np.nan
    else:
        groupedByID = residuals.groupby('ID')
        return groupedByID['CWRES'].agg(outlier_count_func)


def _predicted(
    predict: Callable[[Model], pd.DataFrame], model: Model, column: str
) -> Union[pd.Series, float]:
    try:
        predicted = predict(model)
    except ModelfitResultsError:
        return np.nan
    except ModuleNotFoundError:
        warnings.warn("tflite is not installed, returning nan")
        return np.nan
    except ImportError:
        warnings.warn("tflite cannot be imported, returning nan")
        return np.nan
    if predicted is None:
        return np.nan
    return predicted[column]


def predicted_residual(model: Model) -> Union[pd.Series, float]:
    return _predicted(predict_outliers, model, 'residual')


def predicted_dofv(model: Model) -> Union[pd.Series, float]:
    return _predicted(predict_influential_individuals, model, 'dofv')


def ofv(model: Model) -> Union[pd.Series, float]:
    res = model.modelfit_results
    return np.nan if res is None else res.individual_ofv


def dofv(parent_model: Union[Model, None], candidate_model: Model) -> Union[pd.Series, float]:
    return np.nan if parent_model is None else ofv(parent_model) - ofv(candidate_model)


def groupedByIDAddColumnsOneModel(modelsDict: Dict[str, Model], model: Model) -> pd.DataFrame:
    id_column_name = model.datainfo.id_column.name
    index = pd.Index(data=model.dataset[id_column_name].unique(), name=id_column_name)
    df = pd.DataFrame(
        {
            'parent_model': parent_model_name(model),
            'outlier_count': outlier_count(model),
            'ofv': ofv(model),
            'dofv_vs_parent': dofv(modelsDict.get(model.parent_model), model),
            'predicted_dofv': predicted_dofv(model),
            'predicted_residual': predicted_residual(model),
        },
        index=index,
    )
    return df


def summarize_individuals_count_table(models=None, df=None):
    r"""Create a count table for individual data

    Content of the various columns:

    +-------------------------+------------------------------------------------------------------------------------------------+
    | Column                  | Description                                                                                    |
    +=========================+================================================================================================+
    | ``inf_selection``       | Number of subjects influential on model selection.                                             |
    |                         | :math:`\mathrm{OFV}_{parent} - \mathrm{OFV} > 3.84 \veebar`                                    |
    |                         | :math:`\mathrm{OFV}_{parent} - \mathrm{iOFV}_{parent} - (\mathrm{OFV} - \mathrm{iOFV}) > 3.84` |
    +-------------------------+------------------------------------------------------------------------------------------------+
    | ``inf_params``          | Number of subjects influential on parameters. predicted_dofv > 3.84                            |
    +-------------------------+------------------------------------------------------------------------------------------------+
    | ``out_obs``             | Number of subjects having at least one outlying observation (CWRES > 5)                        |
    +-------------------------+------------------------------------------------------------------------------------------------+
    | ``out_ind``             | Number of outlying subjects. predicted_residual > 3.0                                          |
    +-------------------------+------------------------------------------------------------------------------------------------+
    | ``inf_outlier``         | Number of subjects both influential by any criteria and outlier by any criteria                |
    +-------------------------+------------------------------------------------------------------------------------------------+

    Parameters
    ----------
    models : list of models
        List of models to summarize.
    df : pd.DataFrame
        Output from a previous call to summarize_individuals.

    Returns
    -------
    pd.DataFrame
        Table with one row per model.

    See also
    --------
    summarize_individuals : Get raw individual data

    """  # noqa: E501
    if models:
        df = summarize_individuals(models)
    if df is None:
        return None

    is_out_obs = df['outlier_count'] > 0.0
    is_out_ind = df['predicted_residual'] > 3.0
    is_inf_params = df['predicted_dofv'] > 3.84

    out_obs = is_out_obs.groupby(level='model', sort=False).sum().astype('int32')
    out_ind = is_out_ind.groupby(level='model', sort=False).sum().astype('int32')
    inf_params = is_inf_params.groupby(level='model', sort=False).sum().astype('int32')

    ninds = len(df.index.unique(level='ID'))
    parents = df['parent_model'].iloc[::ninds]
    parent_ofvs = df.loc[parents]['ofv'].reset_index(drop=True)
    parent_ofvs.index = df.index

    for name in df.index.unique(level='model'):
        if name == df.loc[name]['parent_model'].iloc[0]:
            start_name = name
            break
    # FIXME: Doesn't have to have a start model

    ofv_sums = df['ofv'].groupby('model').sum()
    parent_sums = parent_ofvs.groupby('model').sum()
    full_ofv_diff = parent_sums - ofv_sums  # / len(df.index.unique(level='ID'))
    full_ofv_diff.loc[start_name] = 0

    removed_diff = (parent_sums - parent_ofvs) - (ofv_sums - df['ofv'])
    is_inf_selection = (full_ofv_diff > 3.84) ^ (removed_diff > 3.84)
    inf_selection = is_inf_selection.groupby(level='model', sort=False).sum().astype('int32')

    is_inf_outlier = (is_out_obs | is_out_ind) & (is_inf_params | is_inf_selection)
    inf_outlier = is_inf_outlier.groupby(level='model', sort=False).sum().astype('int32')
    parents.index = inf_selection.index
    res = pd.DataFrame(
        {
            'parent_model': parents,
            'inf_selection': inf_selection,
            'inf_params': inf_params,
            'out_obs': out_obs,
            'out_ind': out_ind,
            'inf_outlier': inf_outlier,
        }
    )
    return res
