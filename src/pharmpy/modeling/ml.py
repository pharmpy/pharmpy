from pathlib import Path

import numpy as np
import pandas as pd

from ..model import ModelfitResultsError
from .common import get_model_covariates
from .data import (
    get_ids,
    get_number_of_individuals,
    get_number_of_observations,
    get_number_of_observations_per_individual,
)


def _all_parameters(model):
    # All parameter estimates including fixed
    # This might warrant a function in modeling
    d = {p.name: p.init for p in model.parameters}
    d.update(dict(model.modelfit_results.parameter_estimates))
    return pd.Series(d)


def _create_dataset(model):
    res = model.modelfit_results
    if res is None:
        raise ModelfitResultsError("Need ModelfitResults for model")
    if res.residuals is None:
        raise ModelfitResultsError("No residuals available in ModelfitResults")
    if 'CWRES' not in res.residuals.columns:
        raise ModelfitResultsError("CWRES not an available residual in ModelfitResults")
    idcol = model.datainfo.id_column.name
    nids = get_number_of_individuals(model)
    nobs = get_number_of_observations(model)
    nobsi = get_number_of_observations_per_individual(model)
    cwres = res.residuals['CWRES']
    iofv = res.individual_estimates
    npar = len(model.parameters)
    ofv = res.ofv

    # Max ratio of abs(ETAi) and omegai
    variance_omegas = [
        model.random_variables.get_variance(rv.name).name for rv in model.random_variables.etas
    ]
    all_paramvals = _all_parameters(model)
    omega_estimates = np.sqrt(all_paramvals[variance_omegas])
    abs_ebes = res.individual_estimates.abs()
    ebe_ratio = abs_ebes / list(omega_estimates)
    max_ebe_ratio = ebe_ratio.max(axis=1).fillna(1.0)
    # Set to 1 if division by zero, e.g. from omega fix to 0

    # exp(OFVi / nobsi) / exp(OFV / nobs)
    iofv = res.individual_ofv
    ofv_ratio = np.exp(iofv / nobsi) / np.exp(model.modelfit_results.ofv / nobs)

    # mean(ETA / OMEGA)
    cov = res.individual_estimates_covariance
    etc_diag = np.sqrt(pd.DataFrame([np.diag(y) for y in cov], columns=cov.iloc[0].columns))
    etc_ratio = etc_diag / list(omega_estimates)
    mean_etc_ratio = etc_ratio.mean(axis=1).fillna(1.0)
    mean_etc_ratio.index = ofv_ratio.index

    # max((abs(indcov - mean(cov))) / sd(cov))
    cov_names = get_model_covariates(model, strings=True)
    if len(cov_names) > 0:
        covariates = model.dataset[cov_names + [idcol]].set_index(idcol)
        mean_covs = covariates.groupby(idcol).mean()
        maxcov = (abs(mean_covs - mean_covs.mean()) / mean_covs.std()).max(axis=1)
    else:
        maxcov = 0.0

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


def predict_outliers(model, cutoff=3.0):
    """Predict outliers for a model using a machine learning model.

    See the :ref:`simeval <Individual OFV summary>` documentation for a definition of the `residual`

    Parameters
    ----------
    model : Model
        Pharmpy model
    cutoff : float
        Cutoff threshold for a residual singalling an outlier

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
    >>> predict_outliers(model)     # doctest: +SKIP
        residual  outlier
    ID
    1  -0.281443    False
    2   1.057392    False
    3  -0.119105    False
    4  -0.846849    False
    5   0.600540    False
    6   1.014008    False
    7  -0.750734    False
    8   0.247175    False
    9   0.117002    False
    10 -0.835389    False
    11  3.529582     True
    12 -0.035670    False
    13  0.292333    False
    14  0.303278    False
    15 -0.565949    False
    16 -0.078192    False
    17 -0.291295    False
    18  2.466421    False
    19 -0.402343    False
    20 -0.699996    False
    21 -0.567987    False
    22 -0.526776    False
    23  0.303918    False
    24 -0.177588    False
    25  1.272142    False
    26 -0.390000    False
    27  0.775876    False
    28 -0.501528    False
    29 -0.700951    False
    30 -0.352599    False
    31  0.294196    False
    32  0.744014    False
    33 -0.215364    False
    34  0.208691    False
    35  1.713130    False
    36  0.300293    False
    37 -0.810736    False
    38  0.459877    False
    39 -0.675125    False
    40 -0.563835    False
    41 -0.526945    False
    42  4.449199     True
    43  0.720714    False
    44 -0.792248    False
    45 -0.860923    False
    46 -0.731858    False
    47 -0.247131    False
    48  1.894190    False
    49 -0.282737    False
    50 -0.153398    False
    51  1.200546    False
    52  0.902774    False
    53  0.586427    False
    54  0.183329    False
    55  1.036930    False
    56 -0.639868    False
    57 -0.765279    False
    58 -0.209665    False
    59 -0.225693    False

    See also
    --------
    predict_influential_individuals
    predict_influential_outliers

    """
    model_path = Path(__file__).parent.resolve() / 'ml_models' / 'outliers.tflite'
    data = _create_dataset(model)
    output = _predict_with_tflite(model_path, data)
    df = pd.DataFrame({'residual': output, 'outlier': output > cutoff}, index=get_ids(model))
    df.index.name = model.datainfo.id_column.name
    return df


def predict_influential_individuals(model, cutoff=3.84):
    """Predict influential individuals for a model using a machine learning model.

    Parameters
    ----------
    model : Model
        Pharmpy model
    cutoff : float
        Cutoff threshold for a dofv signalling an influential individual

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
    df = pd.DataFrame({'dofv': output, 'influential': output > cutoff}, index=get_ids(model))
    df.index.name = model.datainfo.id_column.name
    return df


def predict_influential_outliers(model, outlier_cutoff=3, influential_cutoff=3.84):
    """Predict influential outliers for a model using a machine learning model.

    Parameters
    ----------
    model : Model
        Pharmpy model
    outlier_cutoff : float
        Cutoff threshold for a residual singalling an outlier
    influential_cutoff : float
        Cutoff threshold for a dofv signalling an influential individual

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

    outliers = predict_outliers(model, cutoff=outlier_cutoff)
    infinds = predict_influential_individuals(model, cutoff=influential_cutoff)
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
