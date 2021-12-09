# Functional interface to extract dataset information
from pharmpy.data import ColumnType, DatasetError


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
    return model.dataset.pharmpy.ninds


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
    return model.dataset.pharmpy.nobs


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
    return model.dataset.pharmpy.nobsi


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
    get_number_of_observations
    get_number_of_observations_per_individual

    """
    return model.dataset.pharmpy.observations


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
            raise DatasetError('Could not find EVID or AMT column in dataset')

    dose = model.dataset[label].astype('float64').squeeze()
    mdv = dose.where(dose == 0, other=1).astype('int64').rename('MDV')
    return mdv
