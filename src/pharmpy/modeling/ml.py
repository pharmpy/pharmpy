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
    1  -0.272168    False
    2   1.723904    False
    3  -0.621131    False
    4  -0.426917    False
    5   1.639699    False
    6   1.147001    False
    7  -0.399741    False
    8   0.283612    False
    9   0.606785    False
    10 -0.505261    False
    11  5.414047     True
    12 -0.464646    False
    13  0.140431    False
    14  0.168446    False
    15 -0.472035    False
    16 -0.143273    False
    17 -0.225539    False
    18  3.731170     True
    19 -0.268478    False
    20 -0.616696    False
    21 -0.526294    False
    22 -0.393129    False
    23  1.242754    False
    24 -0.351864    False
    25  3.261746     True
    26 -0.309609    False
    27  2.062392    False
    28 -0.188650    False
    29 -0.434584    False
    30 -0.439246    False
    31 -0.227535    False
    32  3.245340     True
    33 -0.226832    False
    34  0.319488    False
    35  2.549501    False
    36 -0.086467    False
    37 -0.341875    False
    38  0.661899    False
    39 -0.678489    False
    40 -0.371534    False
    41 -0.902069    False
    42  6.411995     True
    43  1.327096    False
    44 -0.754103    False
    45 -0.453768    False
    46 -0.398694    False
    47 -0.133043    False
    48  2.460399    False
    49 -0.505935    False
    50 -0.077702    False
    51  2.011274    False
    52  1.789052    False
    53  0.578962    False
    54  0.891918    False
    55  1.943221    False
    56 -0.232777    False
    57 -0.345403    False
    58 -0.666455    False
    59 -0.578287    False
    """
    model_path = Path(__file__).parent.resolve() / 'ml_models' / 'outliers.tflite'
    interpreter = tflite.Interpreter(str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    data = _create_dataset(model)
    nrows = len(data)
    npdata = data.astype('float32').values

    output = np.empty(nrows)

    for i in range(0, nrows):
        interpreter.set_tensor(input_details[0]['index'], npdata[i : (i + 1), :])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        output[i] = output_data

    df = pd.DataFrame({'residual': output, 'outlier': output > 3}, index=model.dataset.pharmpy.ids)

    return df
