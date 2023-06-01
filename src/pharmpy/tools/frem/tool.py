import itertools
import warnings
from pathlib import Path

from pharmpy.internals.ds.ordered_set import OrderedSet
from pharmpy.model import Model
from pharmpy.modeling import (
    get_covariate_baselines,
    list_time_varying_covariates,
    set_covariates,
    write_model,
)
from pharmpy.tools import read_modelfit_results

from .models import create_model3b


def setup(model_path, covariates):
    input_model = Model.parse_model(model_path)
    covariates = check_covariates(input_model, covariates)
    return covariates


def check_covariates(input_model, covariates):
    """Perform checks of covariates and filter out inappropriate ones

    1. A covariate has the same baseline value for all individuals
    2. Two or more covariates have the exact same base line values
    3. Warn for time varying covariates

    Return a new list of covariates with "bad" ones removed
    """
    kept = []
    for cov in covariates:
        if cov not in input_model.datainfo.names:
            warnings.warn(f'The covariate {cov} is not available in the dataset. Will be skipped.')
        else:
            kept.append(cov)
    covariates = kept

    input_model = set_covariates(input_model, covariates)
    cov_bls = get_covariate_baselines(input_model)

    tvar = list_time_varying_covariates(input_model)
    if tvar:
        warnings.warn(
            f'The covariates {tvar} are time varying, but FREM will only use the '
            f'baseline values.'
        )

    # Covariates with only one baseline value
    unique = (cov_bls.iloc[0] == cov_bls.iloc[1:]).all()
    covset = OrderedSet(covariates)
    unique_covariates = OrderedSet(unique[unique].index)
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
    model1b = Model.parse_model(model_path / 'model_1b.mod')
    model3 = Model.parse_model(model_path / 'model_3.mod')
    model3_res = read_modelfit_results(model_path / 'model_3.mod')
    model3b = create_model3b(model1b, model3, model3_res, int(ncovs))
    write_model(model3b, model_path, force=True)
