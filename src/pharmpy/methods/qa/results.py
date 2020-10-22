from pathlib import Path

import pandas as pd

import pharmpy.random_variables
from pharmpy import Model
from pharmpy.random_variables import VariabilityLevel
from pharmpy.results import Results


class QAResults(Results):
    def __init__(self, dofv=None, fullblock_parameters=None):
        self.dofv = dofv
        self.fullblock_parameters = fullblock_parameters


def calculate_results(original_model, fullblock_model=None):
    fullblock_table, fullblock_dofv = calc_fullblock(original_model, fullblock_model)
    dofv_table = fullblock_dofv
    res = QAResults(dofv=dofv_table, fullblock_parameters=fullblock_table)
    return res


def calc_boxcox(original_model, boxcox_model=None):
    """Retrieve new and old parameters of boxcox"""
    if boxcox_model is None:
        return None
    origres = original_model.modelfit_results
    boxcoxres = boxcox_model.modelfit_results
    if boxcoxres is None:
        return None
    boxcoxres.random_variables.etas
    origres


def calc_fullblock(original_model, fullblock_model):
    """Retrieve new and old parameters of full block"""
    if fullblock_model is None:
        return None
    origres = original_model.modelfit_results
    fullres = fullblock_model.modelfit_results
    if fullres is None:
        return None
    _, dist = list(fullblock_model.random_variables.distributions(level=VariabilityLevel.IIV))[0]
    fullblock_parameters = [str(symb) for symb in dist.free_symbols]
    origres.reparameterize(
        [
            pharmpy.random_variables.NormalParametrizationSd,
            pharmpy.random_variables.MultivariateNormalParametrizationSdCorr,
        ]
    )
    fullres.reparameterize(
        [
            pharmpy.random_variables.NormalParametrizationSd,
            pharmpy.random_variables.MultivariateNormalParametrizationSdCorr,
        ]
    )
    new_params = (
        fullres.parameter_estimates[fullblock_parameters]
        .reindex(index=fullres.parameter_estimates.index)
        .dropna()
    )
    old_params = origres.parameter_estimates
    table = pd.DataFrame({'new': new_params, 'old': old_params}).reindex(index=new_params.index)

    degrees = table['old'].isna().sum()
    dofv = origres.ofv - fullres.ofv
    dofv_tab = pd.DataFrame({'dofv': dofv, 'df': degrees}, index=['fullblock'])
    return table, dofv_tab


def psn_qa_results(path):
    """Create qa results from a PsN qa run

    :param path: Path to PsN qa run directory
    :return: A :class:`QAResults` object
    """
    path = Path(path)

    original_model = Model(path / 'linearize_run' / 'scm_dir1' / 'derivatives.mod')
    fullblock_path = path / 'modelfit_run' / 'fullblock.mod'
    if fullblock_path.is_file():
        fullblock_model = Model(fullblock_path)
    else:
        fullblock_model = None

    res = calculate_results(original_model, fullblock_model=fullblock_model)
    return res
