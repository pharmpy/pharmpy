# Functional interface to extract dataset information
import numpy as np
import pandas as pd

from pharmpy.data import ColumnType, DatasetError


def get_ids(model):
    """Retrieve a list of all subject ids of the dataset

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    list
        All subject ids

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, get_ids
    >>> model = load_example_model("pheno")
    >>> get_ids(model)      # doctest: +ELLIPSIS
    [1, 2, 3, ..., 57, 58, 59]
    """
    idcol = model.datainfo.id_label
    ids = list(model.dataset[idcol].unique())
    return ids


def get_number_of_individuals(model):
    """Retrieve the number of individuals in the model dataset

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    int
        Number of individuals in the model dataset

    Examples
    --------
    >>> from pharmpy.modeling import get_number_of_individuals, load_example_model
    >>> model = load_example_model("pheno")
    >>> get_number_of_individuals(model)
    59

    Notes
    -----
    For NONMEM models this is the number of individuals of the active dataset, i.e. after filtering
    of IGNORE and ACCEPT and removal of individuals with no observations.

    See also
    --------
    get_number_of_observations : Get the number of observations in a dataset
    get_number_of_observations_per_individual : Get the number of observations per individual in a
        dataset

    """
    return len(get_ids(model))


def get_number_of_observations(model):
    """Retrieve the total number of observations in the model dataset

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    int
        Number of observations in the model dataset

    Examples
    --------
    >>> from pharmpy.modeling import get_number_of_observations, load_example_model
    >>> model = load_example_model("pheno")
    >>> get_number_of_observations(model)
    155

    Notes
    -----
    For NONMEM models this is the number of observations of the active dataset, i.e. after filtering
    of IGNORE and ACCEPT and removal of individuals with no observations.

    See also
    --------
    get_number_of_individuals : Get the number of individuals in a dataset
    get_number_of_observations_per_individual : Get the number of observations per individual in a
        dataset

    """
    return len(get_observations(model))


def get_number_of_observations_per_individual(model):
    """Number of observations for each individual

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Series
        Number of observations in the model dataset

    Examples
    --------
    >>> from pharmpy.modeling import get_number_of_observations_per_individual, load_example_model
    >>> model = load_example_model("pheno")
    >>> get_number_of_observations_per_individual(model)
    ID
    1     2
    2     3
    3     3
    4     3
    5     3
    6     3
    7     3
    8     3
    9     4
    10    3
    11    1
    12    3
    13    2
    14    4
    15    2
    16    3
    17    3
    18    4
    19    3
    20    3
    21    3
    22    2
    23    3
    24    3
    25    6
    26    2
    27    2
    28    1
    29    1
    30    2
    31    1
    32    3
    33    2
    34    2
    35    2
    36    3
    37    2
    38    4
    39    3
    40    2
    41    3
    42    2
    43    1
    44    3
    45    3
    46    1
    47    1
    48    5
    49    3
    50    4
    51    3
    52    3
    53    2
    54    4
    55    1
    56    1
    57    2
    58    3
    59    3
    Name: observation_count, dtype: int64

    Notes
    -----
    For NONMEM models this is the individuals and number of observations of the active dataset, i.e.
    after filtering of IGNORE and ACCEPT and removal of individuals with no observations.

    See also
    --------
    get_number_of_individuals : Get the number of individuals in a dataset
    get_number_of_observations_per_individual : Get the number of observations per individual in a
        dataset

    """
    ser = get_observations(model).groupby(model.datainfo.id_label).count()
    ser.name = "observation_count"
    return ser


def get_observations(model):
    """Get observations from dataset

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Series
        Observations indexed over ID and TIME

    Examples
    --------
    >>> from pharmpy.modeling import get_observations, load_example_model
    >>> model = load_example_model("pheno")
    >>> get_observations(model)
        ID  TIME
    1   2.0      17.3
        112.5    31.0
    2   2.0       9.7
        63.5     24.6
        135.5    33.0
                 ...
    58  47.5     27.9
        131.8    31.0
    59  1.8      22.6
        73.8     34.3
        146.8    40.2
    Name: DV, Length: 155, dtype: float64

    See also
    --------
    get_number_of_observations : get the number of observations
    get_number_of_observations_per_individual : get the number of observations per individual
    """
    label = model.datainfo.get_column_label('event')
    if label is None:
        label = model.datainfo.get_column_label('dose')
        if label is None:
            raise DatasetError('Could not identify observation rows in dataset')

    idcol = model.datainfo.id_label
    idvcol = model.datainfo.idv_label
    df = model.dataset.query(f'{label} == 0')

    if df.empty:
        df = model.dataset.astype({label: 'float'})
        df = df.query(f'{label} == 0')

    df = df[[idcol, idvcol, model.datainfo.dv_label]]
    try:
        # FIXME: This shouldn't be needed
        df = df.astype({idvcol: np.float64})
    except ValueError:
        # TIME could not be converted to float (e.g. 10:15)
        pass
    df.set_index([idcol, idvcol], inplace=True)
    return df.squeeze()


def get_baselines(model):
    """Baselines for each subject.

    Baseline is taken to be the first row even if that has a missing value.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.DataFrame
        Dataset with the baselines

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_baselines
    >>> model = load_example_model("pheno")
    >>> get_baselines(model)
           ID TIME   AMT  WGT  APGR   DV  FA1  FA2
    pheno
    1       1   0.  25.0  1.4   7.0  0.0  1.0  1.0
    2       2   0.  15.0  1.5   9.0  0.0  1.0  1.0
    3       3   0.  30.0  1.5   6.0  0.0  1.0  1.0
    4       4   0.  18.6  0.9   6.0  0.0  1.0  1.0
    5       5   0.  27.0  1.4   7.0  0.0  1.0  1.0
    6       6   0.  24.0  1.2   5.0  0.0  1.0  1.0
    7       7   0.  19.0  1.0   5.0  0.0  1.0  1.0
    8       8   0.  24.0  1.2   7.0  0.0  1.0  1.0
    9       9   0.  27.0  1.4   8.0  0.0  1.0  1.0
    10     10   0.  27.0  1.4   7.0  0.0  1.0  1.0
    11     11   0.  24.0  1.2   7.0  0.0  1.0  1.0
    12     12   0.  26.0  1.3   6.0  0.0  1.0  1.0
    13     13   0.  11.0  1.1   6.0  0.0  1.0  1.0
    14     14   0.  22.0  1.1   7.0  0.0  1.0  1.0
    15     15   0.  26.0  1.3   7.0  0.0  1.0  1.0
    16     16   0.  12.0  1.2   9.0  0.0  1.0  1.0
    17     17   0.  22.0  1.1   5.0  0.0  1.0  1.0
    18     18   0.  20.0  1.0   5.0  0.0  1.0  1.0
    19     19   0.  10.0  1.0   1.0  0.0  1.0  1.0
    20     20   0.  24.0  1.2   6.0  0.0  1.0  1.0
    21     21   0.  17.5  1.8   7.0  0.0  1.0  1.0
    22     22   0.  15.0  1.5   8.0  0.0  1.0  1.0
    23     23   0.  60.0  3.1   3.0  0.0  1.0  1.0
    24     24   0.  63.0  3.2   2.0  0.0  1.0  1.0
    25     25   0.  15.0  0.7   1.0  0.0  1.0  1.0
    26     26   0.  70.0  3.5   9.0  0.0  1.0  1.0
    27     27   0.  35.0  1.9   5.0  0.0  1.0  1.0
    28     28   0.  60.0  3.2   9.0  0.0  1.0  1.0
    29     29   0.  20.0  1.0   7.0  0.0  1.0  1.0
    30     30   0.  18.0  1.8   8.0  0.0  1.0  1.0
    31     31   0.  30.0  1.4   8.0  0.0  1.0  1.0
    32     32   0.  70.0  3.6   9.0  0.0  1.0  1.0
    33     33   0.  17.0  1.7   8.0  0.0  1.0  1.0
    34     34   0.  34.0  1.7   4.0  0.0  1.0  1.0
    35     35   0.  25.0  2.5   5.0  0.0  1.0  1.0
    36     36   0.  30.0  1.5   5.0  0.0  1.0  1.0
    37     37   0.  24.0  1.2   9.0  0.0  1.0  1.0
    38     38   0.  26.0  1.3   8.0  0.0  1.0  1.0
    39     39   0.  56.0  1.9  10.0  0.0  1.0  1.0
    40     40   0.  19.0  1.1   3.0  0.0  1.0  1.0
    41     41   0.  34.0  1.7   7.0  0.0  1.0  1.0
    42     42   0.  28.0  2.8   9.0  0.0  1.0  1.0
    43     43   0.  18.0  0.9   1.0  0.0  1.0  1.0
    44     44   0.  14.0  1.4   7.0  0.0  1.0  1.0
    45     45   0.  16.0  0.8   2.0  0.0  1.0  1.0
    46     46   0.  11.0  1.1   8.0  0.0  1.0  1.0
    47     47   0.  40.0  2.6   9.0  0.0  1.0  1.0
    48     48   0.  14.0  0.7   8.0  0.0  1.0  1.0
    49     49   0.  26.0  1.3   8.0  0.0  1.0  1.0
    50     50   0.  20.0  1.1   6.0  0.0  1.0  1.0
    51     51   0.  18.0  0.9   9.0  0.0  1.0  1.0
    52     52   0.   9.5  0.9   7.0  0.0  1.0  1.0
    53     53   0.  17.0  1.7   8.0  0.0  1.0  1.0
    54     54   0.  18.0  1.8   8.0  0.0  1.0  1.0
    55     55   0.  25.0  1.1   4.0  0.0  1.0  1.0
    56     56   0.  12.0  0.6   4.0  0.0  1.0  1.0
    57     57   0.  20.0  2.1   6.0  0.0  1.0  1.0
    58     58   0.  14.0  1.4   8.0  0.0  1.0  1.0
    59     59   0.  22.8  1.1   6.0  0.0  1.0  1.0
    """
    idlab = model.datainfo.id_label
    baselines = model.dataset.groupby(idlab).nth(0)
    return baselines


def get_covariate_baselines(model):
    """Return a dataframe with baselines of all covariates for each id.

    Baseline is taken to be the first row even if that has a missing value.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.DataFrame
        covariate baselines

    See also
    --------
    get_baselines : baselines for all data columns

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, get_covariate_baselines
    >>> model = load_example_model("pheno")
    >>> model.datainfo.set_column_type(["WGT", "APGR"], "covariate")
    >>> get_covariate_baselines(model)
        WGT  APGR
    ID
    1   1.4   7.0
    2   1.5   9.0
    3   1.5   6.0
    4   0.9   6.0
    5   1.4   7.0
    6   1.2   5.0
    7   1.0   5.0
    8   1.2   7.0
    9   1.4   8.0
    10  1.4   7.0
    11  1.2   7.0
    12  1.3   6.0
    13  1.1   6.0
    14  1.1   7.0
    15  1.3   7.0
    16  1.2   9.0
    17  1.1   5.0
    18  1.0   5.0
    19  1.0   1.0
    20  1.2   6.0
    21  1.8   7.0
    22  1.5   8.0
    23  3.1   3.0
    24  3.2   2.0
    25  0.7   1.0
    26  3.5   9.0
    27  1.9   5.0
    28  3.2   9.0
    29  1.0   7.0
    30  1.8   8.0
    31  1.4   8.0
    32  3.6   9.0
    33  1.7   8.0
    34  1.7   4.0
    35  2.5   5.0
    36  1.5   5.0
    37  1.2   9.0
    38  1.3   8.0
    39  1.9  10.0
    40  1.1   3.0
    41  1.7   7.0
    42  2.8   9.0
    43  0.9   1.0
    44  1.4   7.0
    45  0.8   2.0
    46  1.1   8.0
    47  2.6   9.0
    48  0.7   8.0
    49  1.3   8.0
    50  1.1   6.0
    51  0.9   9.0
    52  0.9   7.0
    53  1.7   8.0
    54  1.8   8.0
    55  1.1   4.0
    56  0.6   4.0
    57  2.1   6.0
    58  1.4   8.0
    59  1.1   6.0
    """
    covariates = model.datainfo.get_column_labels('covariate')
    idlab = model.datainfo.id_label
    df = model.dataset[covariates + [idlab]]
    df.set_index(idlab, inplace=True)
    return df.groupby(idlab).nth(0)


def list_time_varying_covariates(model):
    """Return a list of names of all time varying covariates

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    list
        Names of all time varying covariates

    See also
    --------
    get_covariate_baselines : get baselines for all covariates

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, list_time_varying_covariates
    >>> model = load_example_model("pheno")
    >>> list_time_varying_covariates(model)
    []

    """
    cov_labels = model.datainfo.get_column_labels('covariate')
    if len(cov_labels) == 0:
        return []
    else:
        time_var = (
            model.dataset.groupby(by=model.datainfo.id_label)[cov_labels].nunique().gt(1).any()
        )
        return list(time_var.index[time_var])


def get_doses(model):
    """Get a series of all doses

    Indexed with ID and TIME

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Series
        doses

    Example
    -------
    >>> from pharmpy.modeling import load_example_model, get_doses
    >>> model = load_example_model("pheno")
    >>> get_doses(model)
    ID  TIME
    1   0.0      25.0
        12.5      3.5
        24.5      3.5
        37.0      3.5
        48.0      3.5
                 ...
    59  96.0      3.0
        108.3     3.0
        120.5     3.0
        132.3     3.0
        144.8     3.0
    Name: AMT, Length: 589, dtype: float64

    """
    try:
        label = model.datainfo.get_column_label('dose')
    except KeyError:
        raise DatasetError('Could not identify dosing rows in dataset')

    idcol = model.datainfo.id_label
    idvcol = model.datainfo.idv_label
    df = model.dataset.query(f'{label} != 0')
    df = df[[idcol, idvcol, label]]
    try:
        # FIXME: This shouldn't be needed
        df = df.astype({idvcol: np.float64})
    except ValueError:
        # TIME could not be converted to float (e.g. 10:15)
        pass
    df.set_index([idcol, idvcol], inplace=True)
    return df.squeeze()


def get_mdv(model):
    """Get MDVs from dataset

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Series
        MDVs
    """
    try:
        label = model.dataset.pharmpy.labels_by_type[ColumnType.EVENT]
    except KeyError:
        try:
            label = model.dataset.pharmpy.labels_by_type[ColumnType.DOSE]
        except KeyError:
            label = model.dataset.pharmpy.labels_by_type[ColumnType.DV]
            data = model.dataset[label].astype('float64').squeeze()
            mdv = pd.Series(np.zeros(len(data))).astype('int64').rename('MDV')
            return mdv

    data = model.dataset[label].astype('float64').squeeze()
    mdv = data.where(data == 0, other=1).astype('int64').rename('MDV')
    return mdv
