import re
from pathlib import Path
from typing import List, Optional, Union

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.deps.rich import box as rich_box
from pharmpy.deps.rich import console as rich_console
from pharmpy.deps.rich import table as rich_table
from pharmpy.model import ColumnInfo, CompartmentalSystem, DataInfo, DatasetError, Model
from pharmpy.model.model import update_datainfo

from .iterators import resample_data


def get_ids(model: Model):
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


def get_number_of_individuals(model: Model):
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


def get_number_of_observations(model: Model):
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


def get_number_of_observations_per_individual(model: Model):
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


def get_observations(model: Model):
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
        label = model.datainfo.typeix['mdv'][0].name
    except IndexError:
        try:
            label = model.datainfo.typeix['event'][0].name
        except IndexError:
            try:
                label = model.datainfo.typeix['dose'][0].name
            except IndexError:
                label = None  # All data records are observations

    idcol = model.datainfo.id_column.name
    idvcol = model.datainfo.idv_column.name

    if label:
        df = model.dataset.query(f'{label} == 0')
        if df.empty:
            df = model.dataset.astype({label: 'float'})
            df = df.query(f'{label} == 0')
    else:
        df = model.dataset.copy()

    df = df[[idcol, idvcol, model.datainfo.dv_column.name]]
    try:
        # FIXME: This shouldn't be needed
        df = df.astype({idvcol: np.float64})
    except ValueError:
        # TIME could not be converted to float (e.g. 10:15)
        pass
    df.set_index([idcol, idvcol], inplace=True)
    return df.squeeze()


def get_baselines(model: Model):
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
         TIME  AMT  WGT  APGR   DV  FA1  FA2
    ID
    1    0.0  25.0  1.4   7.0  0.0  1.0  1.0
    2    0.0  15.0  1.5   9.0  0.0  1.0  1.0
    3    0.0  30.0  1.5   6.0  0.0  1.0  1.0
    4    0.0  18.6  0.9   6.0  0.0  1.0  1.0
    5    0.0  27.0  1.4   7.0  0.0  1.0  1.0
    6    0.0  24.0  1.2   5.0  0.0  1.0  1.0
    7    0.0  19.0  1.0   5.0  0.0  1.0  1.0
    8    0.0  24.0  1.2   7.0  0.0  1.0  1.0
    9    0.0  27.0  1.4   8.0  0.0  1.0  1.0
    10   0.0  27.0  1.4   7.0  0.0  1.0  1.0
    11   0.0  24.0  1.2   7.0  0.0  1.0  1.0
    12   0.0  26.0  1.3   6.0  0.0  1.0  1.0
    13   0.0  11.0  1.1   6.0  0.0  1.0  1.0
    14   0.0  22.0  1.1   7.0  0.0  1.0  1.0
    15   0.0  26.0  1.3   7.0  0.0  1.0  1.0
    16   0.0  12.0  1.2   9.0  0.0  1.0  1.0
    17   0.0  22.0  1.1   5.0  0.0  1.0  1.0
    18   0.0  20.0  1.0   5.0  0.0  1.0  1.0
    19   0.0  10.0  1.0   1.0  0.0  1.0  1.0
    20   0.0  24.0  1.2   6.0  0.0  1.0  1.0
    21   0.0  17.5  1.8   7.0  0.0  1.0  1.0
    22   0.0  15.0  1.5   8.0  0.0  1.0  1.0
    23   0.0  60.0  3.1   3.0  0.0  1.0  1.0
    24   0.0  63.0  3.2   2.0  0.0  1.0  1.0
    25   0.0  15.0  0.7   1.0  0.0  1.0  1.0
    26   0.0  70.0  3.5   9.0  0.0  1.0  1.0
    27   0.0  35.0  1.9   5.0  0.0  1.0  1.0
    28   0.0  60.0  3.2   9.0  0.0  1.0  1.0
    29   0.0  20.0  1.0   7.0  0.0  1.0  1.0
    30   0.0  18.0  1.8   8.0  0.0  1.0  1.0
    31   0.0  30.0  1.4   8.0  0.0  1.0  1.0
    32   0.0  70.0  3.6   9.0  0.0  1.0  1.0
    33   0.0  17.0  1.7   8.0  0.0  1.0  1.0
    34   0.0  34.0  1.7   4.0  0.0  1.0  1.0
    35   0.0  25.0  2.5   5.0  0.0  1.0  1.0
    36   0.0  30.0  1.5   5.0  0.0  1.0  1.0
    37   0.0  24.0  1.2   9.0  0.0  1.0  1.0
    38   0.0  26.0  1.3   8.0  0.0  1.0  1.0
    39   0.0  56.0  1.9  10.0  0.0  1.0  1.0
    40   0.0  19.0  1.1   3.0  0.0  1.0  1.0
    41   0.0  34.0  1.7   7.0  0.0  1.0  1.0
    42   0.0  28.0  2.8   9.0  0.0  1.0  1.0
    43   0.0  18.0  0.9   1.0  0.0  1.0  1.0
    44   0.0  14.0  1.4   7.0  0.0  1.0  1.0
    45   0.0  16.0  0.8   2.0  0.0  1.0  1.0
    46   0.0  11.0  1.1   8.0  0.0  1.0  1.0
    47   0.0  40.0  2.6   9.0  0.0  1.0  1.0
    48   0.0  14.0  0.7   8.0  0.0  1.0  1.0
    49   0.0  26.0  1.3   8.0  0.0  1.0  1.0
    50   0.0  20.0  1.1   6.0  0.0  1.0  1.0
    51   0.0  18.0  0.9   9.0  0.0  1.0  1.0
    52   0.0   9.5  0.9   7.0  0.0  1.0  1.0
    53   0.0  17.0  1.7   8.0  0.0  1.0  1.0
    54   0.0  18.0  1.8   8.0  0.0  1.0  1.0
    55   0.0  25.0  1.1   4.0  0.0  1.0  1.0
    56   0.0  12.0  0.6   4.0  0.0  1.0  1.0
    57   0.0  20.0  2.1   6.0  0.0  1.0  1.0
    58   0.0  14.0  1.4   8.0  0.0  1.0  1.0
    59   0.0  22.8  1.1   6.0  0.0  1.0  1.0
    """
    idlab = model.datainfo.id_column.name
    baselines = model.dataset.groupby(idlab).nth(0)
    return baselines


def set_covariates(model: Model, covariates: List[str]):
    """Set columns in the dataset to be covariates in the datainfo

    Parameters
    ----------
    model : Model
        Pharmpy model
    covariates : list
        List of column names

    Returns
    -------
    Model
        Pharmpy model object
    """
    di = model.datainfo
    newcols = []
    for col in di:
        if col.name in covariates:
            newcol = col.replace(type='covariate')
            newcols.append(newcol)
        else:
            newcols.append(col)
    model = model.replace(datainfo=di.replace(columns=newcols))
    return model.update_source()


def set_dvid(model: Model, name: str):
    """Set a column to act as DVID. Replace DVID if one is already set.

    Parameters
    ----------
    model : Model
        Pharmpy model
    name : str
        Name of DVID column

    Returns
    -------
    Model
        Pharmpy model object
    """
    di = model.datainfo
    col = di[name]
    if col.type == 'dvid':
        return model

    try:
        curdvid = di.typeix['dvid'][0]
    except IndexError:
        pass
    else:
        curdvid = curdvid.replace(type='unknown')
        di = di.set_column(curdvid)

    col = col.replace(
        type='dvid',
        unit=1,
        scale='nominal',
        continuous=False,
        drop=False,
        descriptor='observation identifier',
    )
    df = model.dataset
    if not col.is_integer():
        ser = df[name]
        converted = pd.to_numeric(ser, downcast='integer')
        if not pd.api.types.is_integer_dtype(converted):
            raise ValueError(
                f"Could not use column {name} as DVID because it contains non-integral values"
            )
        df = df.assign(**{name: converted})
        col = col.replace(datatype=ColumnInfo.convert_pd_dtype_to_datatype(converted.dtype))
        new_dataset = True
    else:
        new_dataset = False

    col = col.replace(categories=sorted(df[name].unique()))

    di = di.set_column(col)

    if new_dataset:
        model = model.replace(datainfo=di, dataset=df)
    else:
        model = model.replace(datainfo=di)
    return model.update_source()


def get_covariate_baselines(model: Model):
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
    >>> from pharmpy.modeling import load_example_model, get_covariate_baselines, set_covariates
    >>> model = load_example_model("pheno")
    >>> model = set_covariates(model, ["WGT", "APGR"])
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
    df = df.set_index(idlab)
    return df.groupby(idlab).nth(0)


def list_time_varying_covariates(model: Model):
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


def get_doses(model: Model):
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


def expand_additional_doses(model: Model, flag: bool = False):
    """Expand additional doses into separate dose records

    Parameters
    ----------
    model : Model
        Pharmpy model object
    flag : bool
        True to add a boolean EXPANDED column to mark added records. In this case all
        columns in the original dataset will be kept. Care needs to be taken to handle
        the new dataset.

    Returns
    -------
    Model
        Pharmpy model object
    """
    try:
        addl = model.datainfo.typeix['additional'][0].name
        ii = model.datainfo.typeix['ii'][0].name
    except IndexError:
        return model
    idv = model.datainfo.idv_column.name
    idcol = model.datainfo.id_column.name

    df = model.dataset.copy()

    try:
        event = model.datainfo.typeix['event'][0].name
    except IndexError:
        df['_RESETGROUP'] = 1.0
    else:
        df['_FLAG'] = df[event] >= 3
        df['_RESETGROUP'] = df.groupby('ID')['_FLAG'].cumsum()
        df.drop('_FLAG', axis=1, inplace=True)

    def fn(a):
        if a[addl] == 0:
            times = [a[idv]]
            expanded = [False]
        else:
            length = int(a[addl])
            times = [a[ii] * x + a[idv] for x in range(length + 1)]
            expanded = [False] + [True] * length
        a['_TIMES'] = times
        a['_EXPANDED'] = expanded
        return a

    df = df.apply(fn, axis=1)
    df = df.apply(lambda x: x.explode() if x.name in ['_TIMES', '_EXPANDED'] else x)
    df = df.astype({'_EXPANDED': np.bool_})
    df = df.groupby([idcol, '_RESETGROUP'], group_keys=False).apply(
        lambda x: x.sort_values(by='_TIMES', kind='stable')
    )
    df[idv] = df['_TIMES']
    df.drop(['_TIMES', '_RESETGROUP'], axis=1, inplace=True)
    if flag:
        df.rename(columns={'_EXPANDED': 'EXPANDED'}, inplace=True)
    else:
        df.drop([addl, ii, '_EXPANDED'], axis=1, inplace=True)
    model = model.replace(dataset=df.reset_index(drop=True))
    return model.update_source()


def get_doseid(model: Model):
    """Get a DOSEID series from the dataset with an id of each dose period starting from 1

    If a a dose and observation exist at the same time point the observation will be counted
    towards the previous dose.

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
    >>> get_doseid(model)  # doctest: +ELLIPSIS
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
    Name: DOSEID, Length: 744, dtype: int...
    """
    try:
        dose = model.datainfo.typeix['dose'][0].name
    except IndexError:
        raise DatasetError('Could not identify dosing rows in dataset')

    df = model.dataset.copy()
    df['DOSEID'] = df[dose]
    df.loc[df['DOSEID'] > 0, 'DOSEID'] = 1
    df['DOSEID'] = df['DOSEID'].astype(int)
    idcol = model.datainfo.id_column.name
    df['DOSEID'] = df.groupby(idcol)['DOSEID'].cumsum()

    # Adjust for dose and observation at the same time point
    # Observation is moved to previous dose group
    # Except for steady state dose where the dose group is kept
    try:
        eventcol = model.datainfo.typeix['event'][0].name
    except IndexError:
        df['_RESETGROUP'] = 1.0
    else:
        df['_FLAG'] = df[eventcol] >= 3
        df['_RESETGROUP'] = df.groupby('ID')['_FLAG'].cumsum()

    try:
        ss = model.datainfo.typeix['ss'][0].name
    except IndexError:
        ss = None

    idvcol = model.datainfo.idv_column.name
    ser = df.groupby([idcol, idvcol, '_RESETGROUP']).size()
    nonunique = ser[ser > 1]

    for i, time, _ in nonunique.index:
        groupind = df[(df[idcol] == i) & (df[idvcol] == time)].index
        obsind = df[(df[idcol] == i) & (df[idvcol] == time) & (df[dose] == 0)].index
        doseind = set(groupind) - set(obsind)
        if not doseind:
            continue
        maxind = max(doseind)
        for index in obsind:
            if 0 in groupind:  # This is the first dose
                continue
            if maxind > index:  # Dose record is after the observation
                continue
            if ss and df.loc[maxind, ss] > 0:  # No swap for SS dosing
                continue
            curdoseid = df.loc[index, 'DOSEID']
            df.loc[index, 'DOSEID'] = curdoseid - 1

    return df['DOSEID'].copy()


def get_mdv(model: Model):
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
    found = False
    for key in ['mdv', 'event', 'dose']:
        try:
            label = model.datainfo.typeix[key][0].name
            found = True
            break
        except IndexError:
            pass
    else:
        label = model.datainfo.dv_column.name

    data = model.dataset[label].astype('float64').squeeze()

    series = data.where(data == 0, other=1) if found else pd.Series(np.zeros(len(data)))

    return series.astype('int32').rename('MDV')


def get_evid(model: Model):
    """Get the evid from model dataset

    If an event column is present this will be extracted otherwise
    an evid column will be created.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Series
        EVID
    """
    di = model.datainfo
    try:
        eventcols = di.typeix['event']
    except IndexError:
        pass
    else:
        return model.dataset[eventcols[0].name]
    mdv = get_mdv(model)
    return mdv.rename('EVID')


def get_admid(model: Model):
    """Get the admid from model dataset

    If an administration column is present this will be extracted otherwise
    an admid column will be created.
    1 : Oral dose
    2 : IV dose

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Series
        ADMID
    """
    di = model.datainfo
    try:
        admidcols = di.typeix["admid"]
    except IndexError:
        pass
    else:
        return model.dataset[admidcols[0].name]

    oral = iv = None
    odes = model.statements.ode_system
    names = odes.compartment_names
    if isinstance(odes, CompartmentalSystem):
        for dosing in odes.dosing_compartment:
            if dosing == odes.central_compartment:
                iv = names.index(dosing.name) + 1
            else:
                oral = names.index(dosing.name) + 1
    adm = get_cmt(model)
    adm = adm.replace({oral: 1, iv: 2})
    adm.name = "ADMID"
    return adm


def add_admid(model: Model):
    """
    Add an admid column to the model dataset and datainfo. Dependent on the
    presence of a CMT column in order to add admid correctly.
    1 : Oral dose
    2 : IV dose

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    model : Model
        Pharmpy model

    See also
    --------
    get_admid : Get or create an admid column
    get_cmt : Get or create a cmt column
    """
    di = model.datainfo
    if "admid" not in di.types:
        adm = get_admid(model)
        dataset = model.dataset
        dataset["ADMID"] = adm
        di = update_datainfo(model.datainfo, dataset)
        colinfo = di['ADMID'].replace(type='admid')
        model = model.replace(datainfo=di.set_column(colinfo), dataset=dataset)

    return model.update_source()


def get_cmt(model: Model):
    """Get the cmt (compartment) column from the model dataset

    If a cmt column is present this will be extracted otherwise
    a cmt column will be created. If created, multiple dose compartments are
    not supported.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    pd.Series
        CMT
    """
    di = model.datainfo
    try:
        cmtcols = di.typeix['compartment']
    except IndexError:
        pass
    else:
        return model.dataset[cmtcols[0].name]
    odes = model.statements.ode_system
    if isinstance(odes, CompartmentalSystem):
        dosing = odes.dosing_compartment[0]
        names = odes.compartment_names
        dose_cmt = names.index(dosing.name) + 1
    else:
        dose_cmt = 1
    cmt = get_evid(model)
    cmt = cmt.replace({1: dose_cmt, 2: 0, 3: 0, 4: dose_cmt})  # Only consider dose/non-dose
    cmt.name = "CMT"
    return cmt


def add_time_after_dose(model: Model):
    """Calculate and add a TAD column to the dataset"

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import load_example_model, add_time_after_dose
    >>> model = load_example_model("pheno")
    >>> model = add_time_after_dose(model)

    """
    try:
        model.datainfo.descriptorix['time after dose']
    except IndexError:
        pass
    else:
        # Already have time after dose
        return model
    temp = translate_nmtran_time(model)
    idv = temp.datainfo.idv_column.name
    idlab = temp.datainfo.id_column.name
    df = model.dataset.copy()
    df['_NEWTIME'] = temp.dataset[idv]

    try:
        addl = temp.datainfo.typeix['additional'][0].name
    except IndexError:
        addl = None
    else:
        # FIXME: temp workaround, should be canonicalized in Model.replace
        di = update_datainfo(temp.datainfo, df)
        new_idvcol = di.idv_column.replace(type='unknown')
        new_timecol = di['_NEWTIME'].replace(type='idv')
        di = di.set_column(new_idvcol).set_column(new_timecol)
        temp = temp.replace(datainfo=di, dataset=df)
        temp = expand_additional_doses(temp, flag=True)
        df = temp.dataset

    df['_DOSEID'] = get_doseid(temp)

    # Sort in case DOSEIDs are non-increasing
    df = (
        df.groupby(idlab)
        .apply(lambda x: x.sort_values(by=['_DOSEID'], kind='stable', ignore_index=True))
        .reset_index(drop=True)
    )

    df['TAD'] = df.groupby([idlab, '_DOSEID'])['_NEWTIME'].diff().fillna(0)
    df['TAD'] = df.groupby([idlab, '_DOSEID'])['TAD'].cumsum()

    if addl:
        df = df[~df['EXPANDED']].reset_index(drop=True)
        df.drop(columns=['EXPANDED'], inplace=True)

    # Handle case for observation at same timepoint as SS dose
    # In this case II should be used as TAD (imaginary previous dose)
    try:
        ss = model.datainfo.typeix['ss'][0].name
        ii = model.datainfo.typeix['ii'][0].name
    except IndexError:
        pass
    else:

        def fn(df):
            if len(df) < 2:
                return df
            ii_time = None
            for i in df.index:
                if df.loc[i, ss] > 0:
                    ii_time = df.loc[i, ii]
                else:
                    assert ii_time is not None
                    df.loc[i, 'TAD'] = ii_time
            return df

        df = df.groupby([idlab, idv, '_DOSEID'], group_keys=False).apply(fn)

    df.drop(columns=['_NEWTIME', '_DOSEID'], inplace=True)

    # FIXME: temp workaround, should be canonicalized in Model.replace
    di = update_datainfo(model.datainfo, df)
    colinfo = di['TAD'].replace(descriptor='time after dose', unit=di[idv].unit)
    model = model.replace(datainfo=di.set_column(colinfo), dataset=df)
    return model.update_source()


def get_concentration_parameters_from_data(model: Model):
    """Create a dataframe with concentration parameters

    Note that all values are directly calculated from the dataset

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
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
    model = add_time_after_dose(model)
    doseid = get_doseid(model)
    df = model.dataset.copy()
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


def drop_dropped_columns(model: Model):
    """Drop columns marked as dropped from the dataset

    NM-TRAN date columns will not be dropped by this function
    even if marked as dropped.
    Columns not specified in the datainfo ($INPUT for NONMEM)
    will also be dropped from the dataset.

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = drop_dropped_columns(model)
    >>> list(model.dataset.columns)
    ['ID', 'TIME', 'AMT', 'WGT', 'APGR', 'DV', 'FA1', 'FA2']

    See also
    --------
    drop_columns : Drop specific columns or mark them as drop
    """
    datainfo = model.datainfo
    todrop = [
        colname
        for colname in datainfo.names
        if datainfo[colname].drop and datainfo[colname].datatype != 'nmtran-date'
    ]
    todrop += list(set(model.dataset.columns) - set(datainfo.names))
    model = drop_columns(model, todrop)
    return model.update_source()


def drop_columns(model: Model, column_names: Union[List[str], str], mark: bool = False):
    """Drop columns from the dataset or mark as dropped

    Parameters
    ----------
    model : Model
        Pharmpy model object
    column_names : list or str
        List of column names or one column name to drop or mark as dropped
    mark : bool
        Default is to remove column from dataset. Set this to True to only mark as dropped

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = drop_columns(model, ['WGT', 'APGR'])
    >>> list(model.dataset.columns)
    ['ID', 'TIME', 'AMT', 'DV', 'FA1', 'FA2']

    See also
    --------
    drop_dropped_columns : Drop all columns marked as drop
    undrop_columns : Undrop columns of model
    """
    if isinstance(column_names, str):
        column_names = [column_names]
    di = model.datainfo
    newcols, to_drop = [], []
    for col in di:
        if col.name in column_names:
            if mark:
                newcol = col.replace(drop=True)
                newcols.append(newcol)
            else:
                to_drop.append(col.name)
        else:
            newcols.append(col)
    replace_dict = {'datainfo': di.replace(columns=newcols)}
    if to_drop:
        df = model.dataset.copy()
        replace_dict['dataset'] = df.drop(to_drop, axis=1)
    model = model.replace(**replace_dict)
    return model.update_source()


def undrop_columns(model: Model, column_names: Union[List[str], str]):
    """Undrop columns of model

    Parameters
    ----------
    model : Model
        Pharmpy model object
    column_names : list or str
        List of column names or one column name to undrop

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = drop_columns(model, ['WGT', 'APGR'], mark=True)
    >>> model = undrop_columns(model, 'WGT')

    See also
    --------
    drop_dropped_columns : Drop all columns marked as drop
    drop_columns : Drop or mark columns as dropped
    """
    if isinstance(column_names, str):
        column_names = [column_names]
    di = model.datainfo
    newcols = []
    for col in di:
        if col.name in column_names:
            newcol = col.replace(drop=False)
            newcols.append(newcol)
        else:
            newcols.append(col)
    model = model.replace(datainfo=di.replace(columns=newcols))
    return model.update_source()


def _translate_nonmem_time_value(time):
    if ':' in time:
        components = time.split(':')
        if len(components) != 2:
            raise DatasetError(f'Bad TIME format: {time}')
        hours = float(components[0]) + float(components[1]) / 60
        return hours
    else:
        return float(time)


def _translate_time_column(df, timecol, idcol):
    if df[timecol].dtype != np.float64:
        df[timecol] = df[timecol].apply(_translate_nonmem_time_value)
        df[timecol] = df[timecol] - df.groupby(idcol)[timecol].transform('first')
    return df


def _translate_nonmem_time_and_date_value(ser, timecol, datecol):
    timeval = _translate_nonmem_time_value(ser[timecol])
    date = ser[datecol]
    a = re.split(r'[^0-9]', date)
    if date.startswith('-') or len(a) == 1:
        return timeval + float(date) * 24
    elif len(a) == 2:
        year = 2001  # Non leap year
        month = a[1]
        day = a[0]
    elif len(a) == 3:
        if datecol.endswith('E'):
            month = a[0]
            day = a[1]
            year = a[2]
        elif datecol.endswith('1'):
            day = a[0]
            month = a[1]
            year = a[2]
        elif datecol.endswith('3'):
            year = a[0]
            day = a[1]
            month = a[2]
        else:  # Let DAT2 be default if other name
            year = a[0]
            month = a[1]
            day = a[2]
        if len(year) < 3:
            year = int(year)
            if year > 50:
                year += 1900
            else:
                year += 2000
        else:
            year = int(year)
        month = int(month)
        day = int(day)
        hour = int(timeval)
        timeval = (timeval - hour) * 60
        minute = int(timeval)
        timeval = (timeval - minute) * 60
        second = int(timeval)
        timeval = (timeval - second) * 1000000
        microsecond = int(timeval)
        timeval = (timeval - microsecond) * 1000
        nanosecond = int(timeval)
        ts = pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond,
            nanosecond=nanosecond,
        )
        return ts
    else:
        raise DatasetError(f'Bad DATE value: {date}')


def _translate_time_and_date_columns(df, timecol, datecol, idcol):
    df[timecol] = df.apply(
        _translate_nonmem_time_and_date_value, axis=1, timecol=timecol, datecol=datecol
    )
    timediff = df[timecol] - df.groupby(idcol)[timecol].transform('first')
    if df[timecol].dtype != np.float64:
        df[timecol] = timediff.dt.total_seconds() / 3600
    return df


def _find_time_and_date_columns(model):
    # Both time and date can be None. If date is None time must be not None
    time = None
    date = None
    di = model.datainfo
    for col in di:
        if col.datatype == 'nmtran-time' and not col.drop:
            if time is None:
                time = col
            else:
                raise ValueError(f"Multiple time columns found {time} and {col.name}")
        elif col.datatype == 'nmtran-date' and not col.drop:
            if date is None:
                date = col
            else:
                raise ValueError(f"Multiple date columns found {date} and {col.name}")
    if time is None and date is not None:
        raise ValueError(f"Found date column {date}, but no time column")
    return time, date


def translate_nmtran_time(model: Model):
    """Translate NM-TRAN TIME and DATE column into one TIME column

    If dataset of model have special NM-TRAN TIME and DATE columns these
    will be translated into one single time column with time in hours.

    Warnings
    --------
    Use this function with caution. For example reset events are currently not taken into account.

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Model
        Pharmpy model object
    """
    timecol, datecol = _find_time_and_date_columns(model)
    df = model.dataset.copy()
    di = model.datainfo
    idname = di.id_column.name
    if datecol is None:
        if timecol is None:
            return model
        else:
            df = _translate_time_column(df, timecol.name, idname)
    else:
        assert timecol is not None
        df = _translate_time_and_date_columns(df, timecol.name, datecol.name, idname)
        model = drop_columns(model, datecol.name)
        timecol = timecol.replace(unit='h')
    timecol = timecol.replace(datatype='float64')
    di = di.set_column(timecol)
    model = model.replace(datainfo=di, dataset=df)
    return model.update_source()


def remove_loq_data(model: Model, lloq: Optional[float] = None, uloq: Optional[float] = None):
    """Remove loq data records from the dataset

    Does nothing if none of the limits is specified.

    Parameters
    ----------
    model : Model
        Pharmpy model object
    lloq : float
        Lower limit of quantification. Default not specified.
    uloq : float
        Upper limit of quantification. Default not specified.

    Returns
    -------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_loq_data(model, lloq=10, uloq=40)
    >>> len(model.dataset)
    736
    """
    df = model.dataset
    dv = model.datainfo.dv_column.name
    mdv = get_mdv(model)
    keep = pd.Series(True, index=df.index)
    if lloq:
        keep &= (df[dv] >= lloq) | mdv
    if uloq:
        keep &= (df[dv] <= uloq) | mdv
    model = model.replace(dataset=df[keep])
    return model.update_source()


class Checker:
    _all_checks = (
        ('A1', 'Body weight has unit'),
        ('A2', 'Body weight has mass unit'),
        ('A3', 'Body weight >0 and <700kg'),
        ('A4', 'Age has unit'),
        ('A5', 'Age has time unit'),
        ('A6', 'Age >=0 and <130 years'),
        ('A7', 'Lean body mass has unit'),
        ('A8', 'Lean body mass has mass unit'),
        ('A9', 'Lean body mass >0 and <700kg'),
        ('A10', 'Fat free mass has unit'),
        ('A11', 'Fat free mass has mass unit'),
        ('A12', 'Fat free mass >0 and <700kg'),
        ('D1', 'Time after dose has unit'),
        ('D2', 'Time after dose has time unit'),
        ('D3', 'Time after dose >=0'),
        ('D4', 'Plasma concentration has unit'),
        ('D5', 'Plasma concentration has mass/volume unit'),
        ('D6', 'Plasma concentration >= 0'),
        ('I1', 'Subject identifier is unitless'),
    )

    def __init__(self, datainfo, dataset, verbose=False):
        self.datainfo = datainfo
        self.dataset = dataset
        self.verbose = verbose
        self.check_results = {}
        self.violations = []

    def set_result(self, code, test=False, violation=None, skip=False, warn=False):
        if skip:
            result = "SKIP"
        elif test:
            result = "OK"
        else:
            if warn:
                result = "WARN"
            else:
                result = "FAIL"
        if code not in self.check_results or (
            code in self.check_results
            and (
                self.check_results[code] == 'SKIP'
                or self.check_results[code] == 'OK'
                and result in ('WARN', 'FAIL')
                or self.check_results[code] == 'WARN'
                and result == 'FAIL'
            )
        ):
            self.check_results[code] = result
        if result in ('WARN', 'FAIL'):
            self.violations.append((code, result, violation))

    def check_has_unit(self, code, col):
        has_unit = col.unit is not None
        self.set_result(code, test=has_unit, violation=col.name, warn=True)
        return has_unit

    def check_is_unitless(self, code, col):
        is_unitless = col.unit == sympy.Integer(1)
        self.set_result(code, test=is_unitless, violation=col.name, warn=True)

    def check_dimension(self, code, column, dim):
        if column.unit is None:
            self.set_result(code, skip=True)
            return False
        else:
            dim2 = sympy.physics.units.Dimension(
                sympy.physics.units.si.SI.get_dimensional_expr(column.unit)
            )
            self.set_result(
                code,
                test=dim == dim2,
                violation=f"Unit {column.unit} of {column.name} is not a {dim} unit",
            )
            return dim == dim2

    def check_range(self, code, col, lower, upper, unit, lower_included=True, upper_included=True):
        name = col.name
        if lower == 0:
            scaled_lower = lower
        else:
            scaled_lower = float(sympy.physics.units.convert_to(lower * unit, col.unit) / col.unit)
        if upper == 0:
            scaled_upper = upper
        else:
            scaled_upper = float(sympy.physics.units.convert_to(upper * unit, col.unit) / col.unit)
        if lower_included:
            lower_viol = self.dataset[name] < scaled_lower
        else:
            lower_viol = self.dataset[name] <= scaled_lower
        if upper_included:
            upper_viol = self.dataset[name] > scaled_upper
        else:
            upper_viol = self.dataset[name] >= scaled_upper
        all_viol = lower_viol | upper_viol
        violations = all_viol[all_viol]
        if not violations.empty:
            for i in violations.index:
                self.set_result(
                    code,
                    test=False,
                    violation=f"{col.name} index={i} value={self.dataset[name].loc[i]}",
                )
        else:
            self.set_result(code, test=True)

    def get_dataframe(self):
        codes = []
        checks = []
        results = []
        violations = []

        for code, msg in Checker._all_checks:
            if code not in self.check_results:
                self.check_results[code] = "SKIP"

            if self.check_results[code] in ['OK', 'SKIP']:
                if self.verbose:
                    codes.append(code)
                    checks.append(msg)
                    results.append(self.check_results[code])
                    violations.append(None)
            else:
                for viol in self.violations:
                    if (
                        viol[0] == code
                        and viol[1] == "FAIL"
                        or (viol[1] == "WARN" and self.verbose)
                    ):
                        codes.append(code)
                        checks.append(msg)
                        results.append(viol[1])
                        violations.append(viol[2])
        df = pd.DataFrame(
            {'code': codes, 'check': checks, 'result': results, 'violation': violations}
        )
        return df

    def print(self):
        table = rich_table.Table(title="Dataset checks", box=rich_box.SQUARE)
        table.add_column("Code")
        table.add_column("Check")
        table.add_column("Result")
        table.add_column("Violation")

        for code, msg in Checker._all_checks:
            if code not in self.check_results:
                self.check_results[code] = "SKIP"

            if self.check_results[code] in ['OK', 'SKIP']:
                if self.verbose:
                    table.add_row(code, msg, f'[bold green]{self.check_results[code]}', "")
            else:
                for viol in self.violations:
                    if (
                        viol[0] == code
                        and viol[1] == "FAIL"
                        or (viol[1] == "WARN" and self.verbose)
                    ):
                        result = viol[1]
                        if result == "FAIL":
                            result = f"[bold red]{result}"
                        else:
                            result = f"[bold yellow]{result}"
                        table.add_row(code, msg, result, viol[2])

        if table.rows:  # Do not print an empty table
            console = rich_console.Console()
            console.print(table)


def check_dataset(model: Model, dataframe: bool = False, verbose: bool = False):
    """Check dataset for consistency across a set of rules

    Parameters
    ----------
    model : Model
        Pharmpy model object
    dataframe : bool
        True to return a DataFrame instead of printing to the console
    verbose : bool
        Print out all rules checked if True else print only failed rules

    Returns
    -------
    pd.DataFrame
        Only returns a DataFrame is dataframe=True
    """
    di = model.datainfo
    df = model.dataset
    checker = Checker(di, df, verbose=verbose)

    for col in di:
        if col.descriptor == "body weight":
            checker.check_has_unit("A1", col)
            samedim = checker.check_dimension("A2", col, sympy.physics.units.mass)
            if samedim:
                checker.check_range("A3", col, 0, 700, sympy.physics.units.kg, False, False)

        if col.descriptor == "age":
            checker.check_has_unit("A4", col)
            samedim = checker.check_dimension("A5", col, sympy.physics.units.time)
            if samedim:
                checker.check_range("A6", col, 0, 130, sympy.physics.units.year, True, False)

        if col.descriptor == "lean body mass":
            checker.check_has_unit("A7", col)
            samedim = checker.check_dimension("A8", col, sympy.physics.units.mass)
            if samedim:
                checker.check_range("A9", col, 0, 700, sympy.physics.units.kg, False, False)

        if col.descriptor == "fat free mass":
            checker.check_has_unit("A10", col)
            samedim = checker.check_dimension("A11", col, sympy.physics.units.mass)
            if samedim:
                checker.check_range("A12", col, 0, 700, sympy.physics.units.kg, False, False)

        if col.descriptor == "time after dose":
            checker.check_has_unit("D1", col)
            samedim = checker.check_dimension("D2", col, sympy.physics.units.time)
            if samedim:
                checker.check_range(
                    "D3", col, 0, float('inf'), sympy.physics.units.second, True, False
                )

        if col.descriptor == "plasma concentration":
            checker.check_has_unit("D4", col)
            samedim = checker.check_dimension(
                "D5", col, sympy.physics.units.mass / sympy.physics.units.length**3
            )
            if samedim:
                checker.check_range(
                    "D6",
                    col,
                    0,
                    float('inf'),
                    sympy.physics.units.kg / sympy.physics.units.L,
                    True,
                    False,
                )

        if col.descriptor == "subject identifier":
            checker.check_is_unitless("I1", col)

    if dataframe:
        return checker.get_dataframe()
    else:
        checker.print()


def read_dataset_from_datainfo(
    datainfo: Union[DataInfo, Path, str], datatype: Optional[str] = None
):
    """Read a dataset given a datainfo object or path to a datainfo file

    Parameters
    ----------
    datainfo : DataInfo | Path | str
        A datainfo object or a path to a datainfo object
    datatype : str
        A string to specify dataset type

    Returns
    -------
    pd.DataFrame
        The dataset
    """
    if not isinstance(datainfo, DataInfo):
        datainfo = DataInfo.read_json(datainfo)

    if datainfo.path is None:
        raise ValueError('datainfo.path is None')
    from pharmpy.model.external.nonmem.dataset import read_nonmem_dataset
    from pharmpy.model.external.nonmem.parsing import filter_observations

    if datatype == 'nonmem':
        drop = [col.drop for col in datainfo]
        df = read_nonmem_dataset(
            datainfo.path,
            ignore_character='@',
            drop=drop,
            colnames=datainfo.names,
            dtype=datainfo.get_dtype_dict(),
        )
        # This assumes a PK model
        df = filter_observations(df, list(df.columns))
    else:
        df = pd.read_csv(
            datainfo.path,
            sep=datainfo.separator,
            dtype=datainfo.get_dtype_dict(),
            float_precision='round_trip',
        )
    return df


def deidentify_data(
    df: pd.DataFrame, id_column: str = 'ID', date_columns: Optional[List[str]] = None
):
    """Deidentify a dataset

    Two operations are performed on the dataset:

    1. All ID numbers are randomized from the range 1 to n
    2. All columns containing dates will have the year changed

    The year change is done by letting the earliest year in the dataset
    be used as a reference and by maintaining leap years. The reference year
    will either be 1901, 1902, 1903 or 1904 depending on its distance to the closest
    preceeding leap year.

    Parameters
    ----------
    df : pd.DataFrame
        A dataset
    id_column : str
        Name of the id column
    date_columns : list
        Names of all date columns

    Returns
    -------
    pd.DataFrame
        Deidentified dataset
    """
    df = df.copy()
    df[id_column] = pd.to_numeric(df[id_column])
    resampler = resample_data(df, id_column)
    df, _ = next(resampler)

    if date_columns is None:
        return df
    for datecol in date_columns:
        if pd.api.types.is_datetime64_any_dtype(df[datecol]):
            pass
        elif df[datecol].dtype == 'object':
            # assume string
            df[datecol] = pd.to_datetime(df[datecol])
        else:
            raise ValueError(f"Column {datecol} does not seem to contain a date")
    earliest_date = df[date_columns].min().min()

    # Handle leap year modulo
    earliest_year_modulo = earliest_date.year % 4
    reference_offset = 4 if earliest_year_modulo == 0 else earliest_year_modulo
    reference_year = 1900 + reference_offset
    delta = earliest_date.year - reference_year

    def convert(x):
        new = x.replace(year=x.year - delta)
        return new

    for datecol in date_columns:
        df[datecol] = df[datecol].transform(convert)

    return df
