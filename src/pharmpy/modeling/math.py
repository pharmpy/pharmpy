from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.math import cov2corr


def calculate_se_from_cov(cov: pd.DataFrame):
    """Calculate standard errors from a covariance matrix

    Parameters
    ----------
    cov : pd.DataFrame
        Input covariance matrix

    Return
    ------
    pd.Series
        Standard errors

    Examples
    --------
    >>> from pharmpy.modeling import calculate_se_from_cov
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> results = load_example_modelfit_results("pheno")
    >>> cov = results.covariance_matrix
    >>> cov
                      PTVCL          PTVV   THETA_3      IVCL           IVV     SIGMA_1_1
    PTVCL      4.411510e-08  4.010000e-08 -0.000002 -0.000001  1.538630e-07  8.178090e-08
    PTVV       4.010000e-08  7.233530e-04 -0.000804  0.000050  7.171840e-05  1.461760e-05
    THETA_3   -1.665010e-06 -8.040250e-04  0.007016 -0.000108 -3.944800e-05  2.932950e-05
    IVCL      -1.093430e-06  4.981380e-05 -0.000108  0.000180 -1.856650e-05  4.867230e-06
    IVV        1.538630e-07  7.171840e-05 -0.000039 -0.000019  5.589820e-05 -4.685650e-07
    SIGMA_1_1  8.178090e-08  1.461760e-05  0.000029  0.000005 -4.685650e-07  5.195640e-06
    >>> calculate_se_from_cov(cov)
    PTVCL        0.000210
    PTVV         0.026895
    THETA_3      0.083762
    IVCL         0.013415
    IVV          0.007477
    SIGMA_1_1    0.002279
    dtype: float64

    See also
    --------
    calculate_se_from_prec : Standard errors from precision matrix
    calculate_corr_from_cov : Correlation matrix from covariance matrix
    calculate_cov_from_prec : Covariance matrix from precision matrix
    calculate_cov_from_corrse : Covariance matrix from correlation matrix and standard errors
    calculate_prec_from_cov : Precision matrix from covariance matrix
    calculate_prec_from_corrse : Precision matrix from correlation matrix and standard errors
    calculate_corr_from_prec : Correlation matrix from precision matrix
    """
    se = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)
    return se


def calculate_se_from_prec(precision_matrix: pd.DataFrame):
    """Calculate standard errors from a precision matrix

    Parameters
    ----------
    precision_matrix : pd.DataFrame
        Input precision matrix

    Return
    ------
    pd.Series
        Standard errors

    Examples
    --------
    >>> from pharmpy.modeling import calculate_se_from_prec
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> results = load_example_modelfit_results("pheno")
    >>> prec = results.precision_matrix
    >>> prec
                      PTVCL          PTVV       THETA_3           IVCL           IVV      SIGMA_1_1
    PTVCL      2.995567e+07  22660.028196  16057.855248  203511.614428 -39474.250514 -820118.299536
    PTVV       2.266003e+04   2129.904642    260.176234    -375.266233  -2800.816246   -7718.769557
    THETA_3    1.605786e+04    260.176234    187.038903     177.207683   -205.808480   -2225.150449
    IVCL       2.035116e+05   -375.266233    177.207683    7527.530027   2462.974821   -9977.488860
    IVV       -3.947425e+04  -2800.816246   -205.808480    2462.974821  22343.198618    9370.758371
    SIGMA_1_1 -8.201183e+05  -7718.769557  -2225.150449   -9977.488860   9370.758371  249847.177845
    >>> calculate_se_from_prec(prec)
    PTVCL        0.000210
    PTVV         0.026895
    THETA_3      0.083762
    IVCL         0.013415
    IVV          0.007477
    SIGMA_1_1    0.002279
    dtype: float64

    See also
    --------
    calculate_se_from_cov : Standard errors from covariance matrix
    calculate_corr_from_cov : Correlation matrix from covariance matrix
    calculate_cov_from_prec : Covariance matrix from precision matrix
    calculate_cov_from_corrse : Covariance matrix from correlation matrix and standard errors
    calculate_prec_from_cov : Precision matrix from covariance matrix
    calculate_prec_from_corrse : Precision matrix from correlation matrix and standard errors
    calculate_corr_from_prec : Correlation matrix from precision matrix
    """
    se = pd.Series(
        np.sqrt(np.diag(np.linalg.inv(precision_matrix.values))), index=precision_matrix.index
    )
    return se


def calculate_corr_from_cov(cov: pd.DataFrame):
    """Calculate correlation matrix from a covariance matrix

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix

    Return
    ------
    pd.DataFrame
        Correlation matrix

    Examples
    --------
    >>> from pharmpy.modeling import calculate_corr_from_cov
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> results = load_example_modelfit_results("pheno")
    >>> cov = results.covariance_matrix
    >>> cov
                      PTVCL          PTVV   THETA_3      IVCL           IVV     SIGMA_1_1
    PTVCL      4.411510e-08  4.010000e-08 -0.000002 -0.000001  1.538630e-07  8.178090e-08
    PTVV       4.010000e-08  7.233530e-04 -0.000804  0.000050  7.171840e-05  1.461760e-05
    THETA_3   -1.665010e-06 -8.040250e-04  0.007016 -0.000108 -3.944800e-05  2.932950e-05
    IVCL      -1.093430e-06  4.981380e-05 -0.000108  0.000180 -1.856650e-05  4.867230e-06
    IVV        1.538630e-07  7.171840e-05 -0.000039 -0.000019  5.589820e-05 -4.685650e-07
    SIGMA_1_1  8.178090e-08  1.461760e-05  0.000029  0.000005 -4.685650e-07  5.195640e-06
    >>> calculate_corr_from_cov(cov)
                  PTVCL      PTVV   THETA_3      IVCL       IVV  SIGMA_1_1
    PTVCL      1.000000  0.007099 -0.094640 -0.388059  0.097981   0.170820
    PTVV       0.007099  1.000000 -0.356899  0.138062  0.356662   0.238441
    THETA_3   -0.094640 -0.356899  1.000000 -0.096515 -0.062991   0.153616
    IVCL      -0.388059  0.138062 -0.096515  1.000000 -0.185111   0.159170
    IVV        0.097981  0.356662 -0.062991 -0.185111  1.000000  -0.027495
    SIGMA_1_1  0.170820  0.238441  0.153616  0.159170 -0.027495   1.000000

    See also
    --------
    calculate_se_from_cov : Standard errors from covariance matrix
    calculate_se_from_prec : Standard errors from precision matrix
    calculate_cov_from_prec : Covariance matrix from precision matrix
    calculate_cov_from_corrse : Covariance matrix from correlation matrix and standard errors
    calculate_prec_from_cov : Precision matrix from covariance matrix
    calculate_prec_from_corrse : Precision matrix from correlation matrix and standard errors
    calculate_corr_from_prec : Correlation matrix from precision matrix
    """

    corr = pd.DataFrame(cov2corr(cov.values), index=cov.index, columns=cov.columns)
    return corr


def calculate_cov_from_prec(precision_matrix: pd.DataFrame):
    """Calculate covariance matrix from a precision matrix

    Parameters
    ----------
    precision_matrix : pd.DataFrame
        Precision matrix

    Return
    ------
    pd.DataFrame
        Covariance matrix

    Examples
    --------
    >>> from pharmpy.modeling import calculate_cov_from_prec
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> results = load_example_modelfit_results("pheno")
    >>> prec = results.precision_matrix
    >>> prec
                      PTVCL          PTVV       THETA_3           IVCL           IVV      SIGMA_1_1
    PTVCL      2.995567e+07  22660.028196  16057.855248  203511.614428 -39474.250514 -820118.299536
    PTVV       2.266003e+04   2129.904642    260.176234    -375.266233  -2800.816246   -7718.769557
    THETA_3    1.605786e+04    260.176234    187.038903     177.207683   -205.808480   -2225.150449
    IVCL       2.035116e+05   -375.266233    177.207683    7527.530027   2462.974821   -9977.488860
    IVV       -3.947425e+04  -2800.816246   -205.808480    2462.974821  22343.198618    9370.758371
    SIGMA_1_1 -8.201183e+05  -7718.769557  -2225.150449   -9977.488860   9370.758371  249847.177845
    >>> calculate_cov_from_prec(prec)
                      PTVCL          PTVV   THETA_3      IVCL           IVV     SIGMA_1_1
    PTVCL      4.411510e-08  4.010000e-08 -0.000002 -0.000001  1.538630e-07  8.178090e-08
    PTVV       4.010000e-08  7.233530e-04 -0.000804  0.000050  7.171840e-05  1.461760e-05
    THETA_3   -1.665010e-06 -8.040250e-04  0.007016 -0.000108 -3.944800e-05  2.932950e-05
    IVCL      -1.093430e-06  4.981380e-05 -0.000108  0.000180 -1.856650e-05  4.867230e-06
    IVV        1.538630e-07  7.171840e-05 -0.000039 -0.000019  5.589820e-05 -4.685650e-07
    SIGMA_1_1  8.178090e-08  1.461760e-05  0.000029  0.000005 -4.685650e-07  5.195640e-06

    See also
    --------
    calculate_se_from_cov : Standard errors from covariance matrix
    calculate_se_from_prec : Standard errors from precision matrix
    calculate_corr_from_cov : Correlation matrix from covariance matrix
    calculate_cov_from_corrse : Covariance matrix from correlation matrix and standard errors
    calculate_prec_from_cov : Precision matrix from covariance matrix
    calculate_prec_from_corrse : Precision matrix from correlation matrix and standard errors
    calculate_corr_from_prec : Correlation matrix from precision matrix
    """

    cov = pd.DataFrame(
        np.linalg.inv(precision_matrix.values),
        index=precision_matrix.index,
        columns=precision_matrix.columns,
    )
    return cov


def calculate_cov_from_corrse(corr: pd.DataFrame, se: pd.Series):
    """Calculate covariance matrix from a correlation matrix and standard errors

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix
    se : pd.Series
        Standard errors

    Return
    ------
    pd.DataFrame
        Covariance matrix

    Examples
    --------
    >>> from pharmpy.modeling import calculate_cov_from_corrse
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> results = load_example_modelfit_results("pheno")
    >>> corr = results.correlation_matrix
    >>> se = results.standard_errors
    >>> corr
                  PTVCL      PTVV   THETA_3      IVCL       IVV  SIGMA_1_1
    PTVCL      1.000000  0.007099 -0.094640 -0.388059  0.097981   0.170820
    PTVV       0.007099  1.000000 -0.356899  0.138062  0.356662   0.238441
    THETA_3   -0.094640 -0.356899  1.000000 -0.096515 -0.062991   0.153616
    IVCL      -0.388059  0.138062 -0.096515  1.000000 -0.185111   0.159170
    IVV        0.097981  0.356662 -0.062991 -0.185111  1.000000  -0.027495
    SIGMA_1_1  0.170820  0.238441  0.153616  0.159170 -0.027495   1.000000
    >>> calculate_cov_from_corrse(corr, se)
                      PTVCL          PTVV   THETA_3      IVCL           IVV     SIGMA_1_1
    PTVCL      4.411512e-08  4.009998e-08 -0.000002 -0.000001  1.538630e-07  8.178111e-08
    PTVV       4.009998e-08  7.233518e-04 -0.000804  0.000050  7.171834e-05  1.461762e-05
    THETA_3   -1.665011e-06 -8.040245e-04  0.007016 -0.000108 -3.944801e-05  2.932957e-05
    IVCL      -1.093431e-06  4.981380e-05 -0.000108  0.000180 -1.856651e-05  4.867245e-06
    IVV        1.538630e-07  7.171834e-05 -0.000039 -0.000019  5.589820e-05 -4.685661e-07
    SIGMA_1_1  8.178111e-08  1.461762e-05  0.000029  0.000005 -4.685661e-07  5.195664e-06

    See also
    --------
    calculate_se_from_cov : Standard errors from covariance matrix
    calculate_se_from_prec : Standard errors from precision matrix
    calculate_corr_from_cov : Correlation matrix from covariance matrix
    calculate_cov_from_prec : Covariance matrix from precision matrix
    calculate_prec_from_cov : Precision matrix from covariance matrix
    calculate_prec_from_corrse : Precision matrix from correlation matrix and standard errors
    calculate_corr_from_prec : Correlation matrix from precision matrix
    """

    sd_matrix = np.diag(se.values)
    cov = sd_matrix @ corr.values @ sd_matrix
    cov_df = pd.DataFrame(cov, index=corr.index, columns=corr.columns)
    return cov_df


def calculate_prec_from_cov(cov: pd.DataFrame):
    """Calculate precision matrix from a covariance matrix

    Parameters
    ----------
    cov : pd.DataFrame
        Covariance matrix

    Return
    ------
    pd.DataFrame
        Precision matrix

    Examples
    --------
    >>> from pharmpy.modeling import calculate_prec_from_cov
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> results = load_example_modelfit_results("pheno")
    >>> cov = results.covariance_matrix
    >>> cov
                      PTVCL          PTVV   THETA_3      IVCL           IVV     SIGMA_1_1
    PTVCL      4.411510e-08  4.010000e-08 -0.000002 -0.000001  1.538630e-07  8.178090e-08
    PTVV       4.010000e-08  7.233530e-04 -0.000804  0.000050  7.171840e-05  1.461760e-05
    THETA_3   -1.665010e-06 -8.040250e-04  0.007016 -0.000108 -3.944800e-05  2.932950e-05
    IVCL      -1.093430e-06  4.981380e-05 -0.000108  0.000180 -1.856650e-05  4.867230e-06
    IVV        1.538630e-07  7.171840e-05 -0.000039 -0.000019  5.589820e-05 -4.685650e-07
    SIGMA_1_1  8.178090e-08  1.461760e-05  0.000029  0.000005 -4.685650e-07  5.195640e-06
    >>> calculate_prec_from_cov(cov)
                      PTVCL          PTVV       THETA_3           IVCL           IVV      SIGMA_1_1
    PTVCL      2.995567e+07  22660.028196  16057.855248  203511.614428 -39474.250514 -820118.299536
    PTVV       2.266003e+04   2129.904642    260.176234    -375.266233  -2800.816246   -7718.769557
    THETA_3    1.605786e+04    260.176234    187.038903     177.207683   -205.808480   -2225.150449
    IVCL       2.035116e+05   -375.266233    177.207683    7527.530027   2462.974821   -9977.488860
    IVV       -3.947425e+04  -2800.816246   -205.808480    2462.974821  22343.198618    9370.758371
    SIGMA_1_1 -8.201183e+05  -7718.769557  -2225.150449   -9977.488860   9370.758371  249847.177845

    See also
    --------
    calculate_se_from_cov : Standard errors from covariance matrix
    calculate_se_from_prec : Standard errors from precision matrix
    calculate_corr_from_cov : Correlation matrix from covariance matrix
    calculate_cov_from_prec : Covariance matrix from precision matrix
    calculate_cov_from_corrse : Covariance matrix from correlation matrix and standard errors
    calculate_prec_from_corrse : Precision matrix from correlation matrix and standard errors
    calculate_corr_from_prec : Correlation matrix from precision matrix
    """

    Pm = pd.DataFrame(np.linalg.inv(cov.values), index=cov.index, columns=cov.columns)
    return Pm


def calculate_prec_from_corrse(corr: pd.DataFrame, se: pd.Series):
    """Calculate precision matrix from a correlation matrix and standard errors

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix
    se : pd.Series
        Standard errors

    Return
    ------
    pd.DataFrame
        Precision matrix

    Examples
    --------
    >>> from pharmpy.modeling import calculate_prec_from_corrse
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> results = load_example_modelfit_results("pheno")
    >>> corr = results.correlation_matrix
    >>> se = results.standard_errors
    >>> corr
                  PTVCL      PTVV   THETA_3      IVCL       IVV  SIGMA_1_1
    PTVCL      1.000000  0.007099 -0.094640 -0.388059  0.097981   0.170820
    PTVV       0.007099  1.000000 -0.356899  0.138062  0.356662   0.238441
    THETA_3   -0.094640 -0.356899  1.000000 -0.096515 -0.062991   0.153616
    IVCL      -0.388059  0.138062 -0.096515  1.000000 -0.185111   0.159170
    IVV        0.097981  0.356662 -0.062991 -0.185111  1.000000  -0.027495
    SIGMA_1_1  0.170820  0.238441  0.153616  0.159170 -0.027495   1.000000
    >>> calculate_prec_from_corrse(corr, se)
                      PTVCL          PTVV       THETA_3           IVCL           IVV      SIGMA_1_1
    PTVCL      2.995565e+07  22660.041788  16057.848052  203511.410335 -39474.240358 -820116.179011
    PTVV       2.266004e+04   2129.908225    260.176399    -375.266263  -2800.818557   -7718.757955
    THETA_3    1.605785e+04    260.176399    187.038825     177.207512   -205.808434   -2225.144772
    IVCL       2.035114e+05   -375.266263    177.207512    7527.518562   2462.972906   -9977.457873
    IVV       -3.947424e+04  -2800.818557   -205.808434    2462.972906  22343.197906    9370.736254
    SIGMA_1_1 -8.201162e+05  -7718.757955  -2225.144772   -9977.457873   9370.736254  249846.006431

    See also
    --------
    calculate_se_from_cov : Standard errors from covariance matrix
    calculate_se_from_prec : Standard errors from precision matrix
    calculate_corr_from_cov : Correlation matrix from covariance matrix
    calculate_cov_from_prec : Covariance matrix from precision matrix
    calculate_cov_from_corrse : Covariance matrix from correlation matrix and standard errors
    calculate_prec_from_cov : Precision matrix from covariance matrix
    calculate_corr_from_prec : Correlation matrix from precision matrix
    """

    sd_matrix = np.diag(se.values)
    cov = sd_matrix @ corr.values @ sd_matrix
    Pm = pd.DataFrame(np.linalg.inv(cov), index=corr.index, columns=corr.columns)
    return Pm


def calculate_corr_from_prec(precision_matrix: pd.DataFrame):
    """Calculate correlation matrix from a precision matrix

    Parameters
    ----------
    precision_matrix : pd.DataFrame
        Precision matrix

    Return
    ------
    pd.DataFrame
        Correlation matrix

    Examples
    --------
    >>> from pharmpy.modeling import calculate_corr_from_prec
    >>> from pharmpy.tools import load_example_modelfit_results
    >>> results = load_example_modelfit_results("pheno")
    >>> prec = results.precision_matrix
    >>> prec
                      PTVCL          PTVV       THETA_3           IVCL           IVV      SIGMA_1_1
    PTVCL      2.995567e+07  22660.028196  16057.855248  203511.614428 -39474.250514 -820118.299536
    PTVV       2.266003e+04   2129.904642    260.176234    -375.266233  -2800.816246   -7718.769557
    THETA_3    1.605786e+04    260.176234    187.038903     177.207683   -205.808480   -2225.150449
    IVCL       2.035116e+05   -375.266233    177.207683    7527.530027   2462.974821   -9977.488860
    IVV       -3.947425e+04  -2800.816246   -205.808480    2462.974821  22343.198618    9370.758371
    SIGMA_1_1 -8.201183e+05  -7718.769557  -2225.150449   -9977.488860   9370.758371  249847.177845
    >>> calculate_corr_from_prec(prec)
                  PTVCL      PTVV   THETA_3      IVCL       IVV  SIGMA_1_1
    PTVCL      1.000000  0.007099 -0.094640 -0.388059  0.097981   0.170820
    PTVV       0.007099  1.000000 -0.356899  0.138062  0.356662   0.238441
    THETA_3   -0.094640 -0.356899  1.000000 -0.096515 -0.062991   0.153616
    IVCL      -0.388059  0.138062 -0.096515  1.000000 -0.185111   0.159170
    IVV        0.097981  0.356662 -0.062991 -0.185111  1.000000  -0.027495
    SIGMA_1_1  0.170820  0.238441  0.153616  0.159170 -0.027495   1.000000

    See also
    --------
    calculate_se_from_cov : Standard errors from covariance matrix
    calculate_se_from_prec : Standard errors from precision matrix
    calculate_corr_from_cov : Correlation matrix from covariance matrix
    calculate_cov_from_prec : Covariance matrix from precision matrix
    calculate_cov_from_corrse : Covariance matrix from correlation matrix and standard errors
    calculate_prec_from_cov : Precision matrix from covariance matrix
    calculate_prec_from_corrse : Precision matrix from correlation matrix and standard errors
    """

    corr = pd.DataFrame(
        cov2corr(np.linalg.inv(precision_matrix.values)),
        index=precision_matrix.index,
        columns=precision_matrix.columns,
    )
    return corr
