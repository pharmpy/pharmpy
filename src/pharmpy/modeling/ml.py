from pathlib import Path

import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite

from .common import get_model_covariates
from .data import (
    get_number_of_individuals,
    get_number_of_observations,
    get_number_of_observations_per_individual,
)


def _create_dataset(model):
    idcol = model.dataset.pharmpy.id_label
    nids = get_number_of_individuals(model)
    nobs = get_number_of_observations(model)
    nobsi = get_number_of_observations_per_individual(model)
    cwres = model.modelfit_results.residuals['CWRES']

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
    cov_ratio = (abs(mean_covs - mean_covs.mean()) / mean_covs.std()).mean(axis=1)

    df = pd.DataFrame(
        {
            'nids': nids,
            'nobs': nobs,
            'nobs_subj': nobsi / (nobs / nids),
            'ncovs': len(cov_names),
            'cov_ratio': cov_ratio,
            'max_cwres': abs(cwres).groupby(idcol).max(),
            'median_cwres': abs(cwres).groupby(idcol).median(),
            'max_ebe_ratio': max_ebe_ratio,
            'ofv_ratio': ofv_ratio,
            'etc_ratio': mean_etc_ratio,
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
    1  -0.272152    False
    2   1.723895    False
    3  -0.621257    False
    4  -0.426955    False
    5   1.639485    False
    6   1.147173    False
    7  -0.399760    False
    8   0.283858    False
    9   0.606595    False
    10 -0.505307    False
    11  5.413732     True
    12 -0.464585    False
    13  0.140510    False
    14  0.168691    False
    15 -0.472065    False
    16 -0.143271    False
    17 -0.225547    False
    18  3.730850     True
    19 -0.268493    False
    20 -0.616852    False
    21 -0.526294    False
    22 -0.393106    False
    23  1.242479    False
    24 -0.352115    False
    25  3.262130     True
    26 -0.309581    False
    27  2.062735    False
    28 -0.188629    False
    29 -0.434607    False
    30 -0.439220    False
    31 -0.227505    False
    32  3.245315     True
    33 -0.226902    False
    34  0.319238    False
    35  2.549845    False
    36 -0.086605    False
    37 -0.341915    False
    38  0.661809    False
    39 -0.678476    False
    40 -0.371514    False
    41 -0.901976    False
    42  6.411962     True
    43  1.326782    False
    44 -0.754095    False
    45 -0.453803    False
    46 -0.398793    False
    47 -0.133017    False
    48  2.460449    False
    49 -0.505857    False
    50 -0.077484    False
    51  2.011316    False
    52  1.789271    False
    53  0.579177    False
    54  0.892111    False
    55  1.943623    False
    56 -0.232907    False
    57 -0.345342    False
    58 -0.666367    False
    59 -0.578347    False

    See also
    --------
    predict_influential_individuals
    predict_influential_outliers

    """
    model_path = Path(__file__).parent.resolve() / 'ml_models' / 'outliers.tflite'
    data = _create_dataset(model)
    output = _predict_with_tflite(model_path, data)
    df = pd.DataFrame({'residual': output, 'outlier': output > 3}, index=model.dataset.pharmpy.ids)
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
    data['linearized'] = 0.0
    output = _predict_with_tflite(model_path, data)
    df = pd.DataFrame(
        {'dofv': output, 'influential': output > 3.84}, index=model.dataset.pharmpy.ids
    )
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
