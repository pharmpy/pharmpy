from __future__ import annotations

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
                   POP_CL        POP_VC   COVAPGR    IIV_CL        IIV_VC         SIGMA
    POP_CL   4.408600e-08  4.761930e-08 -0.000002 -0.000001  1.552150e-07  8.042430e-08
    POP_VC   4.761930e-08  7.233190e-04 -0.000804  0.000050  7.174490e-05  1.467290e-05
    COVAPGR -1.679560e-06 -8.041750e-04  0.007015 -0.000108 -3.935790e-05  2.922260e-05
    IIV_CL  -1.090290e-06  4.989580e-05 -0.000108  0.000180 -1.863210e-05  5.049910e-06
    IIV_VC   1.552150e-07  7.174490e-05 -0.000039 -0.000019  5.588920e-05 -4.497590e-07
    SIGMA    8.042430e-08  1.467290e-05  0.000029  0.000005 -4.497590e-07  5.197970e-06
    >>> calculate_se_from_cov(cov)
    POP_CL     0.000210
    POP_VC     0.026895
    COVAPGR    0.083756
    IIV_CL     0.013416
    IIV_VC     0.007476
    SIGMA      0.002280
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
                   POP_CL        POP_VC       COVAPGR         IIV_CL        IIV_VC          SIGMA
    POP_CL   2.993428e+07  22261.039122  16027.859538  203633.930854 -39113.269503 -817314.801755
    POP_VC   2.226104e+04   2129.852212    260.115313    -373.896066  -2799.330946   -7697.921603
    COVAPGR  1.602786e+04    260.115313    187.053488     177.987340   -205.261483   -2224.522815
    IIV_CL   2.036339e+05   -373.896066    177.987340    7542.279597   2472.034556  -10209.416944
    IIV_VC  -3.911327e+04  -2799.330946   -205.261483    2472.034556  22348.216559    9193.203130
    SIGMA   -8.173148e+05  -7697.921603  -2224.522815  -10209.416944   9193.203130  249978.454601
    >>> calculate_se_from_prec(prec)
    POP_CL     0.000210
    POP_VC     0.026895
    COVAPGR    0.083756
    IIV_CL     0.013416
    IIV_VC     0.007476
    SIGMA      0.002280
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
                   POP_CL        POP_VC  COVAPGR    IIV_CL        IIV_VC         SIGMA
    POP_CL   4.408600e-08  4.761930e-08 -0.000002 -0.000001  1.552150e-07  8.042430e-08
    POP_VC   4.761930e-08  7.233190e-04 -0.000804  0.000050  7.174490e-05  1.467290e-05
    COVAPGR -1.679560e-06 -8.041750e-04  0.007015 -0.000108 -3.935790e-05  2.922260e-05
    IIV_CL  -1.090290e-06  4.989580e-05 -0.000108  0.000180 -1.863210e-05  5.049910e-06
    IIV_VC   1.552150e-07  7.174490e-05 -0.000039 -0.000019  5.588920e-05 -4.497590e-07
    SIGMA    8.042430e-08  1.467290e-05  0.000029  0.000005 -4.497590e-07  5.197970e-06
    >>> calculate_corr_from_cov(cov)
               POP_CL    POP_VC   COVAPGR    IIV_CL    IIV_VC     SIGMA
    POP_CL   1.000000  0.008433 -0.095506 -0.387063  0.098882  0.168004
    POP_VC   0.008433  1.000000 -0.357003  0.138290  0.356831  0.239295
    COVAPGR -0.095506 -0.357003  1.000000 -0.095767 -0.062857  0.153034
    IIV_CL  -0.387063  0.138290 -0.095767  1.000000 -0.185775  0.165104
    IIV_VC   0.098882  0.356831 -0.062857 -0.185775  1.000000 -0.026388
    SIGMA    0.168004  0.239295  0.153034  0.165104 -0.026388  1.000000

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
                   POP_CL        POP_VC       COVAPGR         IIV_CL        IIV_VC          SIGMA
    POP_CL   2.993428e+07  22261.039122  16027.859538  203633.930854 -39113.269503 -817314.801755
    POP_VC   2.226104e+04   2129.852212    260.115313    -373.896066  -2799.330946   -7697.921603
    COVAPGR  1.602786e+04    260.115313    187.053488     177.987340   -205.261483   -2224.522815
    IIV_CL   2.036339e+05   -373.896066    177.987340    7542.279597   2472.034556  -10209.416944
    IIV_VC  -3.911327e+04  -2799.330946   -205.261483    2472.034556  22348.216559    9193.203130
    SIGMA   -8.173148e+05  -7697.921603  -2224.522815  -10209.416944   9193.203130  249978.454601
    >>> calculate_cov_from_prec(prec)
                   POP_CL        POP_VC   COVAPGR    IIV_CL        IIV_VC         SIGMA
    POP_CL   4.408600e-08  4.761930e-08 -0.000002 -0.000001  1.552150e-07  8.042430e-08
    POP_VC   4.761930e-08  7.233190e-04 -0.000804  0.000050  7.174490e-05  1.467290e-05
    COVAPGR -1.679560e-06 -8.041750e-04  0.007015 -0.000108 -3.935790e-05  2.922260e-05
    IIV_CL  -1.090290e-06  4.989580e-05 -0.000108  0.000180 -1.863210e-05  5.049910e-06
    IIV_VC   1.552150e-07  7.174490e-05 -0.000039 -0.000019  5.588920e-05 -4.497590e-07
    SIGMA    8.042430e-08  1.467290e-05  0.000029  0.000005 -4.497590e-07  5.197970e-06

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
               POP_CL    POP_VC   COVAPGR    IIV_CL    IIV_VC     SIGMA
    POP_CL   1.000000  0.008433 -0.095506 -0.387063  0.098882  0.168004
    POP_VC   0.008433  1.000000 -0.357003  0.138290  0.356831  0.239295
    COVAPGR -0.095506 -0.357003  1.000000 -0.095767 -0.062857  0.153034
    IIV_CL  -0.387063  0.138290 -0.095767  1.000000 -0.185775  0.165104
    IIV_VC   0.098882  0.356831 -0.062857 -0.185775  1.000000 -0.026388
    SIGMA    0.168004  0.239295  0.153034  0.165104 -0.026388  1.000000
    >>> calculate_cov_from_corrse(corr, se)
                   POP_CL        POP_VC   COVAPGR    IIV_CL        IIV_VC         SIGMA
    POP_CL   4.408614e-08  4.761939e-08 -0.000002 -0.000001  1.552153e-07  8.042458e-08
    POP_VC   4.761939e-08  7.233195e-04 -0.000804  0.000050  7.174494e-05  1.467293e-05
    COVAPGR -1.679562e-06 -8.041749e-04  0.007015 -0.000108 -3.935789e-05  2.922264e-05
    IIV_CL  -1.090293e-06  4.989586e-05 -0.000108  0.000180 -1.863212e-05  5.049924e-06
    IIV_VC   1.552153e-07  7.174494e-05 -0.000039 -0.000019  5.588923e-05 -4.497600e-07
    SIGMA    8.042458e-08  1.467293e-05  0.000029  0.000005 -4.497600e-07  5.197990e-06

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

    sd_matrix = np.diag(se.to_numpy())
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
                   POP_CL        POP_VC   COVAPGR    IIV_CL        IIV_VC         SIGMA
    POP_CL   4.408600e-08  4.761930e-08 -0.000002 -0.000001  1.552150e-07  8.042430e-08
    POP_VC   4.761930e-08  7.233190e-04 -0.000804  0.000050  7.174490e-05  1.467290e-05
    COVAPGR -1.679560e-06 -8.041750e-04  0.007015 -0.000108 -3.935790e-05  2.922260e-05
    IIV_CL  -1.090290e-06  4.989580e-05 -0.000108  0.000180 -1.863210e-05  5.049910e-06
    IIV_VC   1.552150e-07  7.174490e-05 -0.000039 -0.000019  5.588920e-05 -4.497590e-07
    SIGMA    8.042430e-08  1.467290e-05  0.000029  0.000005 -4.497590e-07  5.197970e-06
    >>> calculate_prec_from_cov(cov)
                   POP_CL        POP_VC       COVAPGR         IIV_CL        IIV_VC          SIGMA
    POP_CL   2.993428e+07  22261.039122  16027.859538  203633.930854 -39113.269503 -817314.801755
    POP_VC   2.226104e+04   2129.852212    260.115313    -373.896066  -2799.330946   -7697.921603
    COVAPGR  1.602786e+04    260.115313    187.053488     177.987340   -205.261483   -2224.522815
    IIV_CL   2.036339e+05   -373.896066    177.987340    7542.279597   2472.034556  -10209.416944
    IIV_VC  -3.911327e+04  -2799.330946   -205.261483    2472.034556  22348.216559    9193.203130
    SIGMA   -8.173148e+05  -7697.921603  -2224.522815  -10209.416944   9193.203130  249978.454601

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
               POP_CL    POP_VC   COVAPGR    IIV_CL    IIV_VC     SIGMA
    POP_CL   1.000000  0.008433 -0.095506 -0.387063  0.098882  0.168004
    POP_VC   0.008433  1.000000 -0.357003  0.138290  0.356831  0.239295
    COVAPGR -0.095506 -0.357003  1.000000 -0.095767 -0.062857  0.153034
    IIV_CL  -0.387063  0.138290 -0.095767  1.000000 -0.185775  0.165104
    IIV_VC   0.098882  0.356831 -0.062857 -0.185775  1.000000 -0.026388
    SIGMA    0.168004  0.239295  0.153034  0.165104 -0.026388  1.000000
    >>> calculate_prec_from_corrse(corr, se)
                   POP_CL        POP_VC       COVAPGR         IIV_CL        IIV_VC          SIGMA
    POP_CL   2.993418e+07  22260.995666  16027.840996  203633.422078 -39113.196303 -817311.952371
    POP_VC   2.226100e+04   2129.850713    260.115336    -373.895598  -2799.329201   -7697.904374
    COVAPGR  1.602784e+04    260.115336    187.053654     177.987259   -205.261518   -2224.519605
    IIV_CL   2.036334e+05   -373.895598    177.987259    7542.266046   2472.031665  -10209.388516
    IIV_VC  -3.911320e+04  -2799.329201   -205.261518    2472.031665  22348.204432    9193.183296
    SIGMA   -8.173120e+05  -7697.904374  -2224.519605  -10209.388516   9193.183296  249977.511621

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

    sd_matrix = np.diag(se.to_numpy())
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
                   POP_CL        POP_VC       COVAPGR         IIV_CL        IIV_VC          SIGMA
    POP_CL   2.993428e+07  22261.039122  16027.859538  203633.930854 -39113.269503 -817314.801755
    POP_VC   2.226104e+04   2129.852212    260.115313    -373.896066  -2799.330946   -7697.921603
    COVAPGR  1.602786e+04    260.115313    187.053488     177.987340   -205.261483   -2224.522815
    IIV_CL   2.036339e+05   -373.896066    177.987340    7542.279597   2472.034556  -10209.416944
    IIV_VC  -3.911327e+04  -2799.330946   -205.261483    2472.034556  22348.216559    9193.203130
    SIGMA   -8.173148e+05  -7697.921603  -2224.522815  -10209.416944   9193.203130  249978.454601
    >>> calculate_corr_from_prec(prec)
               POP_CL    POP_VC   COVAPGR    IIV_CL    IIV_VC     SIGMA
    POP_CL   1.000000  0.008433 -0.095506 -0.387063  0.098882  0.168004
    POP_VC   0.008433  1.000000 -0.357003  0.138290  0.356831  0.239295
    COVAPGR -0.095506 -0.357003  1.000000 -0.095767 -0.062857  0.153034
    IIV_CL  -0.387063  0.138290 -0.095767  1.000000 -0.185775  0.165104
    IIV_VC   0.098882  0.356831 -0.062857 -0.185775  1.000000 -0.026388
    SIGMA    0.168004  0.239295  0.153034  0.165104 -0.026388  1.000000

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
