import numpy as np
import pandas as pd


def calculate_se_from_cov(cov):
    """Calculate standard errors from a covariance matrix

    Parameters
    ----------
    cov : DataFrame
        Input covariance matrix

    Return
    ------
    Series
        Standard errors

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, calculate_se_from_cov
    >>> model = load_example_model("pheno")
    >>> cov = model.modelfit_results.covariance_matrix
    >>> cov
                    THETA(1)      THETA(2)  ...    OMEGA(2,2)    SIGMA(1,1)
    THETA(1)    4.411510e-08  4.010000e-08  ...  1.538630e-07  8.178090e-08
    THETA(2)    4.010000e-08  7.233530e-04  ...  7.171840e-05  1.461760e-05
    THETA(3)   -1.665010e-06 -8.040250e-04  ... -3.944800e-05  2.932950e-05
    OMEGA(1,1) -1.093430e-06  4.981380e-05  ... -1.856650e-05  4.867230e-06
    OMEGA(2,2)  1.538630e-07  7.171840e-05  ...  5.589820e-05 -4.685650e-07
    SIGMA(1,1)  8.178090e-08  1.461760e-05  ... -4.685650e-07  5.195640e-06
    <BLANKLINE>
    [6 rows x 6 columns]
    >>> calculate_se_from_cov(cov)
    THETA(1)      0.000210
    THETA(2)      0.026895
    THETA(3)      0.083762
    OMEGA(1,1)    0.013415
    OMEGA(2,2)    0.007477
    SIGMA(1,1)    0.002279
    dtype: float64

    """
    se = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)
    return se
