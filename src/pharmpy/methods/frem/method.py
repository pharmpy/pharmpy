import itertools
import warnings
from pathlib import Path

from pharmpy import Model
from pharmpy.data import ColumnType
from pharmpy.data_structures import OrderedSet

from .models import create_model3b


def setup(model_path, covariates):
    input_model = Model(model_path)
    covariates = check_covariates(input_model, covariates)
    return covariates


def check_covariates(input_model, covariates):
    """Perform checks of covariates and filter out inappropriate ones

    1. A covariate has the same baseline value for all individuals
    2. Two or more covariates have the exact same base line values
    3. Warn for time varying covariates

    Return a new list of covariates with "bad" ones removed
    """
    data = input_model.dataset
    data.pharmpy.column_type[covariates] = ColumnType.COVARIATE
    cov_bls = data.pharmpy.covariate_baselines

    tvar = data.pharmpy.time_varying_covariates
    if tvar:
        warnings.warn(
            f'The covariates {tvar} are time varying, but FREM will only use the '
            f'baseline values.'
        )

    # Covariates with only one baseline value
    unique = (cov_bls.iloc[0] == cov_bls.iloc[1:]).all()
    covset = OrderedSet(covariates)
    unique_covariates = OrderedSet(list((unique[unique].index)))
    covset -= unique_covariates
    if unique_covariates:
        warnings.warn(
            f'The covariates {list(unique_covariates)} have the same baseline value for '
            f'all individuals and has been removed from the analysis.'
        )

    # Remove covariates that are equal
    new_covset = OrderedSet(covset)
    for col1, col2 in itertools.combinations(covset, 2):
        if cov_bls[col1].equals(cov_bls[col2]):
            new_covset.discard(col2)
            warnings.warn(
                f'The baselines for covariates {col1} and {col2} are equal for all '
                f'individuals. {col2} has been removed from the analysis.'
            )

    return list(new_covset)


def update_model3b_for_psn(rundir, ncovs):
    """Function to update model3b from psn

    NOTE: This function lets pharmpy tie in to the PsN workflow
          and is a temporary solution
    """
    model_path = Path(rundir) / 'm1'
    model1b = Model(model_path / 'model_1b.mod')
    model3 = Model(model_path / 'model_3.mod')
    model3b = create_model3b(model1b, model3, int(ncovs))
    model3b.write(model_path, force=True)
