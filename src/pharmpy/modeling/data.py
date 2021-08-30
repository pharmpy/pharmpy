# Functional interface to extract dataset information


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
    get_number_of_observations_per_individual : Get the number of observations per individual in a dataset

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
    get_number_of_observations_per_individual : Get the number of observations per individual in a dataset

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
    1.0     2
    2.0     3
    3.0     3
    4.0     3
    5.0     3
    6.0     3
    7.0     3
    8.0     3
    9.0     4
    10.0    3
    11.0    1
    12.0    3
    13.0    2
    14.0    4
    15.0    2
    16.0    3
    17.0    3
    18.0    4
    19.0    3
    20.0    3
    21.0    3
    22.0    2
    23.0    3
    24.0    3
    25.0    6
    26.0    2
    27.0    2
    28.0    1
    29.0    1
    30.0    2
    31.0    1
    32.0    3
    33.0    2
    34.0    2
    35.0    2
    36.0    3
    37.0    2
    38.0    4
    39.0    3
    40.0    2
    41.0    3
    42.0    2
    43.0    1
    44.0    3
    45.0    3
    46.0    1
    47.0    1
    48.0    5
    49.0    3
    50.0    4
    51.0    3
    52.0    3
    53.0    2
    54.0    4
    55.0    1
    56.0    1
    57.0    2
    58.0    3
    59.0    3
    Name: DV, dtype: int64

    Notes
    -----
    For NONMEM models this is the individuals and number of observations of the active dataset, i.e. after filtering
    of IGNORE and ACCEPT and removal of individuals with no observations.

    See also
    --------
    get_number_of_individuals : Get the number of individuals in a dataset
    get_number_of_observations_per_individual : Get the number of observations per individual in a dataset

    """
    return model.dataset.pharmpy.nobsi
