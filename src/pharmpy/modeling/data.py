# Functional interface to extract dataset information
import numpy as np
import pandas as pd

from pharmpy.data import DatasetError
from pharmpy.datainfo import ColumnInfo


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
    idcol = model.datainfo.id_column.name
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
    ser = get_observations(model).groupby(model.datainfo.id_column.name).count()
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
    try:
        label = model.datainfo.typeix['event'][0].name
    except IndexError:
        try:
            label = model.datainfo.typeix['dose'][0].name
        except IndexError:
            raise DatasetError('Could not identify observation rows in dataset')

    idcol = model.datainfo.id_column.name
    idvcol = model.datainfo.idv_column.name
    df = model.dataset.query(f'{label} == 0')

    if df.empty:
        df = model.dataset.astype({label: 'float'})
        df = df.query(f'{label} == 0')

    df = df[[idcol, idvcol, model.datainfo.dv_column.name]]
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
       TIME   AMT  WGT  APGR   DV  FA1  FA2
    ID
    1    0.  25.0  1.4   7.0  0.0  1.0  1.0
    2    0.  15.0  1.5   9.0  0.0  1.0  1.0
    3    0.  30.0  1.5   6.0  0.0  1.0  1.0
    4    0.  18.6  0.9   6.0  0.0  1.0  1.0
    5    0.  27.0  1.4   7.0  0.0  1.0  1.0
    6    0.  24.0  1.2   5.0  0.0  1.0  1.0
    7    0.  19.0  1.0   5.0  0.0  1.0  1.0
    8    0.  24.0  1.2   7.0  0.0  1.0  1.0
    9    0.  27.0  1.4   8.0  0.0  1.0  1.0
    10   0.  27.0  1.4   7.0  0.0  1.0  1.0
    11   0.  24.0  1.2   7.0  0.0  1.0  1.0
    12   0.  26.0  1.3   6.0  0.0  1.0  1.0
    13   0.  11.0  1.1   6.0  0.0  1.0  1.0
    14   0.  22.0  1.1   7.0  0.0  1.0  1.0
    15   0.  26.0  1.3   7.0  0.0  1.0  1.0
    16   0.  12.0  1.2   9.0  0.0  1.0  1.0
    17   0.  22.0  1.1   5.0  0.0  1.0  1.0
    18   0.  20.0  1.0   5.0  0.0  1.0  1.0
    19   0.  10.0  1.0   1.0  0.0  1.0  1.0
    20   0.  24.0  1.2   6.0  0.0  1.0  1.0
    21   0.  17.5  1.8   7.0  0.0  1.0  1.0
    22   0.  15.0  1.5   8.0  0.0  1.0  1.0
    23   0.  60.0  3.1   3.0  0.0  1.0  1.0
    24   0.  63.0  3.2   2.0  0.0  1.0  1.0
    25   0.  15.0  0.7   1.0  0.0  1.0  1.0
    26   0.  70.0  3.5   9.0  0.0  1.0  1.0
    27   0.  35.0  1.9   5.0  0.0  1.0  1.0
    28   0.  60.0  3.2   9.0  0.0  1.0  1.0
    29   0.  20.0  1.0   7.0  0.0  1.0  1.0
    30   0.  18.0  1.8   8.0  0.0  1.0  1.0
    31   0.  30.0  1.4   8.0  0.0  1.0  1.0
    32   0.  70.0  3.6   9.0  0.0  1.0  1.0
    33   0.  17.0  1.7   8.0  0.0  1.0  1.0
    34   0.  34.0  1.7   4.0  0.0  1.0  1.0
    35   0.  25.0  2.5   5.0  0.0  1.0  1.0
    36   0.  30.0  1.5   5.0  0.0  1.0  1.0
    37   0.  24.0  1.2   9.0  0.0  1.0  1.0
    38   0.  26.0  1.3   8.0  0.0  1.0  1.0
    39   0.  56.0  1.9  10.0  0.0  1.0  1.0
    40   0.  19.0  1.1   3.0  0.0  1.0  1.0
    41   0.  34.0  1.7   7.0  0.0  1.0  1.0
    42   0.  28.0  2.8   9.0  0.0  1.0  1.0
    43   0.  18.0  0.9   1.0  0.0  1.0  1.0
    44   0.  14.0  1.4   7.0  0.0  1.0  1.0
    45   0.  16.0  0.8   2.0  0.0  1.0  1.0
    46   0.  11.0  1.1   8.0  0.0  1.0  1.0
    47   0.  40.0  2.6   9.0  0.0  1.0  1.0
    48   0.  14.0  0.7   8.0  0.0  1.0  1.0
    49   0.  26.0  1.3   8.0  0.0  1.0  1.0
    50   0.  20.0  1.1   6.0  0.0  1.0  1.0
    51   0.  18.0  0.9   9.0  0.0  1.0  1.0
    52   0.   9.5  0.9   7.0  0.0  1.0  1.0
    53   0.  17.0  1.7   8.0  0.0  1.0  1.0
    54   0.  18.0  1.8   8.0  0.0  1.0  1.0
    55   0.  25.0  1.1   4.0  0.0  1.0  1.0
    56   0.  12.0  0.6   4.0  0.0  1.0  1.0
    57   0.  20.0  2.1   6.0  0.0  1.0  1.0
    58   0.  14.0  1.4   8.0  0.0  1.0  1.0
    59   0.  22.8  1.1   6.0  0.0  1.0  1.0
    """
    idlab = model.datainfo.id_column.name
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
    >>> model.datainfo[["WGT", "APGR"]].types = "covariate"
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
    covariates = model.datainfo.typeix['covariate'].names
    idlab = model.datainfo.id_column.name
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
    cov_labels = model.datainfo.typeix['covariate'].names
    if len(cov_labels) == 0:
        return []
    else:
        time_var = (
            model.dataset.groupby(by=model.datainfo.id_column.name)[cov_labels]
            .nunique()
            .gt(1)
            .any()
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
        label = model.datainfo.typeix['dose'][0].name
    except IndexError:
        raise DatasetError('Could not identify dosing rows in dataset')

    idcol = model.datainfo.id_column.name
    idvcol = model.datainfo.idv_column.name
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


def get_doseid(model):
    """Get a DOSEID series from the dataset with an id of each dose period starting from 1

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Series
        DOSEIDs

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_doseid
    >>> model = load_example_model("pheno")
    >>> get_doseid(model)
    0       1
    1       1
    2       2
    3       3
    4       4
           ..
    739    10
    740    11
    741    12
    742    13
    743    13
    Name: DOSEID, Length: 744, dtype: int64
    """
    try:
        dose = model.datainfo.typeix['dose'][0].name
    except IndexError:
        raise DatasetError('Could not identify dosing rows in dataset')
    df = model.dataset.copy()
    df['DOSEID'] = df[dose]
    df.loc[df['DOSEID'] > 0, 'DOSEID'] = 1
    df['DOSEID'] = df['DOSEID'].astype(int)
    df['DOSEID'] = df.groupby(model.datainfo.id_column.name)['DOSEID'].cumsum()
    return df['DOSEID'].copy()


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
        label = model.datainfo.typeix['mdv'][0].name
    except IndexError:
        try:
            label = model.datainfo.typeix['event'][0].name
        except IndexError:
            try:
                label = model.datainfo.typeix['dose'][0].name
            except IndexError:
                label = model.datainfo.dv_column.name
                data = model.dataset[label].astype('float64').squeeze()
                mdv = pd.Series(np.zeros(len(data))).astype('int64').rename('MDV')
                return mdv
    data = model.dataset[label].astype('float64').squeeze()
    mdv = data.where(data == 0, other=1).astype('int64').rename('MDV')
    return mdv


def add_time_after_dose(model):
    """Calculate and add a TAD column to the dataset"

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Reference to the same model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, add_time_after_dose
    >>> model = load_example_model("pheno")
    >>> add_time_after_dose(model)  # doctest: +ELLIPSIS
    <...>

    """
    # FIXME: Should not rely on name here. Use coltypes for TAD
    # FIXME: TIME is converted to float. Should be handled when reading in dataset
    df = model.dataset
    doseid = get_doseid(model)
    df['DOSEID'] = doseid
    idv = model.datainfo.idv_column.name
    idlab = model.datainfo.id_column.name
    df[idv] = df[idv].astype(np.float64)
    df['TAD'] = df.groupby([idlab, 'DOSEID'])[idv].diff().fillna(0)
    df['TAD'] = df.groupby([idlab, 'DOSEID'])['TAD'].cumsum()
    df.drop('DOSEID', axis=1, inplace=True)
    ci = ColumnInfo('TAD')
    model.datainfo.append(ci)
    return model


def get_concentration_parameters_from_data(model):
    """Create a dataframe with concentration parameters

    Note that all values are directly calculated from the dataset

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Results
    -------
    pd.DataFrame
        Concentration parameters

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, get_concentration_parameters_from_data
    >>> model = load_example_model("pheno")
    >>> get_concentration_parameters_from_data(model)
               Cmax  Tmax  Cmin  Tmin
    ID DOSEID
    1  1       17.3   2.0   NaN   NaN
       2        NaN   NaN   NaN   NaN
       3        NaN   NaN   NaN   NaN
       4        NaN   NaN   NaN   NaN
       5        NaN   NaN   NaN   NaN
    ...         ...   ...   ...   ...
    59 9        NaN   NaN   NaN   NaN
       10       NaN   NaN   NaN   NaN
       11       NaN   NaN   NaN   NaN
       12       NaN   NaN   NaN   NaN
       13      40.2   2.0   NaN   NaN
    <BLANKLINE>
    [589 rows x 4 columns]
    """
    model = model.copy()
    df = model.dataset
    add_time_after_dose(model)
    doseid = get_doseid(model)
    df['DOSEID'] = doseid
    idlab = model.datainfo.id_column.name
    dv = model.datainfo.dv_column.name
    noobs = df.groupby([idlab, 'DOSEID']).size() == 1
    idx = df.groupby([idlab, 'DOSEID'])[dv].idxmax()
    params = df.loc[idx].set_index([idlab, 'DOSEID'])
    params = params[[dv, 'TAD']]
    params.rename(columns={dv: 'Cmax', 'TAD': 'Tmax'}, inplace=True)
    params.loc[noobs] = np.nan

    grpind = df.groupby(['ID', 'DOSEID']).indices
    keep = []
    for ind, rows in grpind.items():
        index = idx.loc[ind]
        p = params.loc[ind]
        if not np.isnan(p['Tmax']):
            keep += [row for row in rows if row > index]
    minidx = df.iloc[keep].groupby([idlab, 'DOSEID'])[dv].idxmin()
    params2 = df.loc[minidx].set_index([idlab, 'DOSEID'])
    params2 = params2[[dv, 'TAD']]
    params2.rename(columns={dv: 'Cmin', 'TAD': 'Tmin'}, inplace=True)
    res = params.join(params2)
    return res
