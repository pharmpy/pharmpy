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
                    THETA(1)      THETA(2)  THETA(3)  OMEGA(1,1)    OMEGA(2,2)    SIGMA(1,1)
    THETA(1)    4.411510e-08  4.010000e-08 -0.000002   -0.000001  1.538630e-07  8.178090e-08
    THETA(2)    4.010000e-08  7.233530e-04 -0.000804    0.000050  7.171840e-05  1.461760e-05
    THETA(3)   -1.665010e-06 -8.040250e-04  0.007016   -0.000108 -3.944800e-05  2.932950e-05
    OMEGA(1,1) -1.093430e-06  4.981380e-05 -0.000108    0.000180 -1.856650e-05  4.867230e-06
    OMEGA(2,2)  1.538630e-07  7.171840e-05 -0.000039   -0.000019  5.589820e-05 -4.685650e-07
    SIGMA(1,1)  8.178090e-08  1.461760e-05  0.000029    0.000005 -4.685650e-07  5.195640e-06
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


def calculate_se_from_inf(information_matrix):
    """Calculate standard errors from an information matrix

    Parameters
    ----------
    information_matrix : DataFrame
        Input information matrix

    Return
    ------
    Series
        Standard errors

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, calculate_se_from_inf
    >>> model = load_example_model("pheno")
    >>> inf = model.modelfit_results.information_matrix
    >>> inf
                    THETA(1)      THETA(2)      THETA(3)     OMEGA(1,1)    OMEGA(2,2)     SIGMA(1,1)
    THETA(1)    2.995567e+07  22660.028196  16057.855248  203511.614428 -39474.250514 -820118.299536
    THETA(2)    2.266003e+04   2129.904642    260.176234    -375.266233  -2800.816246   -7718.769557
    THETA(3)    1.605786e+04    260.176234    187.038903     177.207683   -205.808480   -2225.150449
    OMEGA(1,1)  2.035116e+05   -375.266233    177.207683    7527.530027   2462.974821   -9977.488860
    OMEGA(2,2) -3.947425e+04  -2800.816246   -205.808480    2462.974821  22343.198618    9370.758371
    SIGMA(1,1) -8.201183e+05  -7718.769557  -2225.150449   -9977.488860   9370.758371  249847.177845
    >>> calculate_se_from_inf(inf)
    THETA(1)      0.000210
    THETA(2)      0.026895
    THETA(3)      0.083762
    OMEGA(1,1)    0.013415
    OMEGA(2,2)    0.007477
    SIGMA(1,1)    0.002279
    dtype: float64
    """
    se = pd.Series(
        np.sqrt(np.diag(np.linalg.inv(information_matrix.values))), index=information_matrix.index
    )
    return se
