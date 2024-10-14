from __future__ import annotations

from typing import Optional, Sequence, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import Model, ModelfitResultsError
from pharmpy.workflows import ModelEntry, ModelfitResults

from .ml import predict_influential_individuals, predict_outliers


def summarize_individuals(mes: Sequence[ModelEntry]) -> pd.DataFrame:
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
    models : list[Model]
        Input models
    models_res : list[ModelfitResults]
        Input results

    Return
    ------
    pd.DataFrame | None
        The summary as a dataframe

    """  # noqa: E501
    models = [me.model for me in mes]
    models_res = [me.modelfit_results for me in mes]
    if len(models) != len(models_res):
        raise ValueError('Different length of `models` and `models_res`')

    resDict = {model.name: res for model, res in zip(models, models_res)}

    df = pd.concat(
        map(
            lambda me: groupedByIDAddColumnsOneModel(resDict, me),
            mes,
        ),
        keys=[model.name for model in models],
        names=['model'],
        axis=0,
        verify_integrity=True,
    )

    assert df is not None
    return df


def model_name(model: Model) -> str:
    return model.name


def outlier_count_func(df: pd.DataFrame) -> float:
    # NOTE: This returns a float because we will potentially concat this
    # with NaNs
    return float((abs(df) > 5).sum())


def outlier_count(res: ModelfitResults, data) -> Union[pd.Series, float]:
    if res is None:
        return np.nan
    residuals = res.residuals
    if residuals is None:
        return np.nan
    else:
        residuals = residuals.join(data['ID']).set_index('ID')
        groupedByID = residuals.groupby('ID')
        return groupedByID['CWRES'].agg(outlier_count_func)


def _predicted(predict, model: Model, res: ModelfitResults, column: str) -> Union[pd.Series, float]:
    try:
        predicted = predict(model, res)
    except ModelfitResultsError:
        return np.nan
    except ImportError:
        return np.nan
    except SystemError:  # for numpy/tflite mismatch on MacOS
        return np.nan
    if predicted is None:
        return np.nan
    return predicted[column]


def predicted_residual(model: Model, res: ModelfitResults) -> Union[pd.Series, float]:
    return _predicted(predict_outliers, model, res, 'residual')


def predicted_dofv(model: Model, res: ModelfitResults) -> Union[pd.Series, float]:
    return _predicted(predict_influential_individuals, model, res, 'dofv')


def ofv(res: ModelfitResults) -> Union[pd.Series, float]:
    return np.nan if res is None or res.individual_ofv is None else res.individual_ofv


def dofv(
    parent_model_res: Optional[ModelfitResults], candidate_model_res: Optional[ModelfitResults]
) -> Union[pd.Series, float]:
    return np.nan if parent_model_res is None else ofv(parent_model_res) - ofv(candidate_model_res)


def groupedByIDAddColumnsOneModel(
    resDict: dict[str, ModelfitResults], me: ModelEntry
) -> pd.DataFrame:
    model = me.model
    model_res = me.modelfit_results
    id_column_name = model.datainfo.id_column.name
    index = pd.Index(data=model.dataset[id_column_name].unique(), name=id_column_name)
    parent_model_name = me.parent.name if me.parent is not None else None
    parent_model_res = None if parent_model_name is None else resDict.get(parent_model_name)
    df = pd.DataFrame(
        {
            'parent_model': parent_model_name,
            'outlier_count': outlier_count(model_res, model.dataset),
            'ofv': ofv(model_res),
            'dofv_vs_parent': dofv(parent_model_res, model_res),
            'predicted_dofv': predicted_dofv(model, model_res),
            'predicted_residual': predicted_residual(model, model_res),
        },
        index=index,
    )
    return df


def summarize_individuals_count_table(
    model_entries: Optional[Sequence[ModelEntry]] = None,
    df: pd.DataFrame = None,
):
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
    model_entries : list of model_entries
        List of model_entries to summarize.
    models_res : list[ModelfitResults]
        Input results
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
    if model_entries is None:
        models = None
    else:
        models = [me.model for me in model_entries]
    if model_entries is None:
        models_res = None
    else:
        models_res = [me.modelfit_results for me in model_entries]

    if models and models_res:
        if len(models) != len(models_res):
            raise ValueError('Different length of `models` and `models_res`')
        df = summarize_individuals(models, models_res)
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
    ofvs = df['ofv']
    parent_ofvs = ofvs.copy()
    for (name, _), parent in parents.items():
        if parent is None:
            parent_ofvs.loc[name] = np.nan
        else:
            parent_ofvs.loc[name] = list(ofvs.loc[parent])

    for name in df.index.unique(level='model'):
        if df.loc[name]['parent_model'].iloc[0] is None:
            start_name = name
            break
    else:
        # FIXME: Handle missing start model
        raise ValueError('Missing start model')

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
