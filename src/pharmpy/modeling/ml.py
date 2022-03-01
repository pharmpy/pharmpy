from pathlib import Path

import numpy as np
import pandas as pd

from .common import get_model_covariates
from .data import (
    get_ids,
    get_number_of_individuals,
    get_number_of_observations,
    get_number_of_observations_per_individual,
)


def _create_dataset(model):
    idcol = model.datainfo.id_column.name
    nids = get_number_of_individuals(model)
    nobs = get_number_of_observations(model)
    nobsi = get_number_of_observations_per_individual(model)
    cwres = model.modelfit_results.residuals['CWRES']
    iofv = model.modelfit_results.individual_estimates
    npar = len(model.parameters)
    ofv = model.modelfit_results.ofv

    # Max ratio of abs(ETAi) and omegai
    variance_omegas = model.random_variables.etas.variance_parameters
    omega_estimates = model.modelfit_results.parameter_estimates[variance_omegas]
    abs_ebes = model.modelfit_results.individual_estimates.abs()
    ebe_ratio = abs_ebes / list(omega_estimates)
    max_ebe_ratio = ebe_ratio.max(axis=1)

    # exp(OFVi / nobsi) / exp(OFV / nobs)
    iofv = model.modelfit_results.individual_ofv
    ofv_ratio = np.exp(iofv / nobsi) / np.exp(model.modelfit_results.ofv / nobs)

    # mean(ETA / OMEGA)
    cov = model.modelfit_results.individual_estimates_covariance
    etc_diag = pd.DataFrame([np.diag(y) for y in cov], columns=cov.iloc[0].columns)
    etc_ratio = etc_diag / list(omega_estimates)
    mean_etc_ratio = etc_ratio.mean(axis=1)
    mean_etc_ratio.index = ofv_ratio.index

    # max((abs(indcov - mean(cov))) / sd(cov))
    cov_names = get_model_covariates(model, strings=True)
    covariates = model.dataset[cov_names + [idcol]].set_index(idcol)
    mean_covs = covariates.groupby(idcol).mean()
    maxcov = (abs(mean_covs - mean_covs.mean()) / mean_covs.std()).max(axis=1)

    df = pd.DataFrame(
        {
            'nids': nids,
            'nobs': nobs,
            'nobs_subj': nobsi / (nobs / nids),
            'nobsi': nobsi,
            'ncovs': len(cov_names),
            'maxcov': maxcov,
            'max_cwres': abs(cwres).groupby(idcol).max(),
            'median_cwres': abs(cwres).groupby(idcol).median(),
            'max_ebe_ratio': max_ebe_ratio,
            'ofv_ratio': ofv_ratio,
            'etc_ratio': mean_etc_ratio,
            'iofv': iofv,
            'npar': npar,
            'ofv': ofv,
        }
    )
    return df


def predict_outliers(model):
    """Predict outliers for a model using a machine learning model.

    See the :ref:`simeval <Individual OFV summary>` documentation for a definition of the `residual`

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Dataframe
        Dataframe over the individuals with a `residual` column containing the raw predicted
        residuals and a `outlier` column with a boolean to tell whether the individual is
        an outlier or not.

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> predict_outliers(model)
        residual  outlier
    1  -0.172678    False
    2   0.675712    False
    3  -0.332006    False
    4  -0.710008    False
    5   0.600121    False
    6   1.155262    False
    7  -0.566974    False
    8  -0.643707    False
    9   0.147274    False
    10 -0.754759    False
    11  2.884859    False
    12 -0.070012    False
    13 -0.497592    False
    14  0.447700    False
    15 -0.605774    False
    16 -0.511041    False
    17 -0.445527    False
    18  2.485890    False
    19 -0.163881    False
    20 -0.706392    False
    21 -0.518767    False
    22 -0.485257    False
    23  0.708911    False
    24 -0.779624    False
    25  3.086815     True
    26 -0.794165    False
    27  0.741012    False
    28 -0.642436    False
    29 -0.696876    False
    30 -0.528595    False
    31 -0.758021    False
    32  1.979468    False
    33 -0.343283    False
    34 -0.513914    False
    35  1.893582    False
    36 -0.094610    False
    37 -0.610348    False
    38  0.139176    False
    39 -0.739828    False
    40 -0.747885    False
    41 -0.680807    False
    42  4.562005     True
    43  0.268274    False
    44 -0.603992    False
    45 -0.810457    False
    46 -0.735620    False
    47 -0.341139    False
    48  2.361898    False
    49 -0.356758    False
    50 -0.836002    False
    51  1.689565    False
    52  0.578408    False
    53  0.379588    False
    54  0.007405    False
    55  0.551320    False
    56 -0.655273    False
    57 -0.628927    False
    58 -0.670912    False
    59 -0.677908    False

    See also
    --------
    predict_influential_individuals
    predict_influential_outliers

    """
    model_path = Path(__file__).parent.resolve() / 'ml_models' / 'outliers.tflite'
    data = _create_dataset(model)
    output = _predict_with_tflite(model_path, data)
    df = pd.DataFrame({'residual': output, 'outlier': output > 3}, index=get_ids(model))
    return df


def predict_influential_individuals(model):
    """Predict influential individuals for a model using a machine learning model.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Dataframe
        Dataframe over the individuals with a `dofv` column containing the raw predicted
        delta-OFV and an `influential` column with a boolean to tell whether the individual is
        influential or not.

    See also
    --------
    predict_influential_outliers
    predict_outliers

    """

    model_path = Path(__file__).parent.resolve() / 'ml_models' / 'infinds.tflite'
    data = _create_dataset(model)
    output = _predict_with_tflite(model_path, data)
    df = pd.DataFrame({'dofv': output, 'influential': output > 3.84}, index=get_ids(model))
    return df


def predict_influential_outliers(model):
    """Predict influential outliers for a model using a machine learning model.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Dataframe
        Dataframe over the individuals with a `outliers` and `dofv` columns containing the raw
        predictions and `influential`, `outlier` and `influential_outlier` boolean columns.

    See also
    --------
    predict_influential_individuals
    predict_outliers

    """

    outliers = predict_outliers(model)
    infinds = predict_influential_individuals(model)
    df = pd.concat([outliers, infinds], axis=1)
    df['influential_outlier'] = df['outlier'] & df['influential']
    return df


def _predict_with_tflite(model_path, data):
    import tflite_runtime.interpreter as tflite

    interpreter = tflite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    npdata = data.astype('float32').values
    nrows = len(data)
    output = np.empty(nrows)

    for i in range(0, nrows):
        interpreter.set_tensor(input_details[0]['index'], npdata[i : (i + 1), :])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output[i] = output_data

    return output
