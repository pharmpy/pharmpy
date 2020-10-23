from pathlib import Path

import numpy as np
import pandas as pd

import pharmpy.methods.psn_helpers as psn_helpers
import pharmpy.random_variables
import pharmpy.symbols
from pharmpy import Model
from pharmpy.random_variables import VariabilityLevel
from pharmpy.results import Results, read_results


class QAResults(Results):
    def __init__(
        self,
        dofv=None,
        fullblock_parameters=None,
        boxcox_parameters=None,
        tdist_parameters=None,
        add_etas_parameters=None,
    ):
        self.dofv = dofv
        self.fullblock_parameters = fullblock_parameters
        self.boxcox_parameters = boxcox_parameters
        self.tdist_parameters = tdist_parameters
        self.add_etas_parameters = add_etas_parameters


def calculate_results(
    original_model,
    base_model,
    fullblock_model=None,
    boxcox_model=None,
    tdist_model=None,
    add_etas_model=None,
    etas_added_to=None,
    frem_results=None,
):
    fullblock_table, fullblock_dofv = calc_fullblock(original_model, fullblock_model)
    boxcox_table, boxcox_dofv = calc_transformed_etas(
        original_model, boxcox_model, 'boxcox', 'lambda'
    )
    tdist_table, tdist_dofv = calc_transformed_etas(original_model, tdist_model, 'tdist', 'df')
    addetas_table, addetas_dofv = calc_add_etas(original_model, add_etas_model, etas_added_to)
    frem_dofv = calc_frem_dofv(base_model, fullblock_model, frem_results)
    dofv_table = pd.concat([fullblock_dofv, boxcox_dofv, tdist_dofv, addetas_dofv, frem_dofv])
    res = QAResults(
        dofv=dofv_table,
        fullblock_parameters=fullblock_table,
        boxcox_parameters=boxcox_table,
        tdist_parameters=tdist_table,
        add_etas_parameters=addetas_table,
    )
    return res


def calc_add_etas(original_model, add_etas_model, etas_added_to):
    dofv_tab = pd.DataFrame({'dofv': np.nan, 'df': np.nan}, index=['add_etas'])
    if add_etas_model is None:
        return None, dofv_tab
    add_etas_res = add_etas_model.modelfit_results
    if add_etas_res is None:
        return None, dofv_tab
    origres = original_model.modelfit_results
    original_etas = [rv.name for rv in original_model.random_variables.etas]
    all_etas = original_etas + etas_added_to
    added = [True] * len(original_etas) + [False] * len(etas_added_to)
    params = add_etas_model.random_variables.variance_parameters(exclude_level=VariabilityLevel.RUV)
    params = [pharmpy.symbols.symbol(p.name) for p in params]
    orig_params = original_model.random_variables.variance_parameters(
        exclude_level=VariabilityLevel.RUV
    )
    orig_params = [pharmpy.symbols.symbol(p.name) for p in orig_params]
    origres.reparameterize(
        [
            pharmpy.random_variables.NormalParametrizationSd,
            pharmpy.random_variables.MultivariateNormalParametrizationSdCorr,
        ]
    )
    add_etas_res.reparameterize(
        [
            pharmpy.random_variables.NormalParametrizationSd,
            pharmpy.random_variables.MultivariateNormalParametrizationSdCorr,
        ]
    )
    add_etas_sds = [add_etas_res.parameter_estimates[p.name] for p in params]
    orig_sds = [origres.parameter_estimates[p.name] for p in orig_params]
    orig_sds += [np.nan] * len(etas_added_to)
    table = pd.DataFrame(
        {'added': added, 'new_sd': add_etas_sds, 'orig_sd': orig_sds}, index=all_etas
    )
    dofv = origres.ofv - add_etas_res.ofv
    dofv_tab = pd.DataFrame({'dofv': dofv, 'df': len(etas_added_to)}, index=['add_etas'])
    return table, dofv_tab


def calc_frem_dofv(base_model, fullblock_model, frem_results):
    """Calculate the dOFV for the frem model"""
    dofv_tab = pd.DataFrame({'dofv': np.nan, 'df': np.nan}, index=['frem'])
    if base_model is None or frem_results is None:
        return dofv_tab
    baseres = base_model.modelfit_results
    if baseres is not None:
        base_ofv = baseres.ofv
    else:
        return dofv_tab
    if fullblock_model is not None:
        fullres = fullblock_model.modelfit_results
        if fullres is not None:
            full_ofv = fullres.ofv
        else:
            return dofv_tab
    else:
        full_ofv = 0

    model2_ofv = frem_results.ofv['ofv']['model_2']
    model4_ofv = frem_results.ofv['ofv']['model_4']

    dofv = model2_ofv - model4_ofv - (base_ofv - full_ofv)
    npar = len(frem_results.covariate_effects.index.get_level_values('parameter').unique())
    ncov = len(frem_results.covariate_effects.index.get_level_values('covariate').unique())
    dofv_tab = pd.DataFrame({'dofv': dofv, 'df': npar * ncov}, index=['frem'])
    return dofv_tab


def calc_transformed_etas(original_model, new_model, transform_name, parameter_name):
    """Retrieve new and old parameters of boxcox"""
    dofv_tab = pd.DataFrame({'dofv': np.nan, 'df': np.nan}, index=[transform_name])
    if new_model is None:
        return None, dofv_tab
    origres = original_model.modelfit_results
    newres = new_model.modelfit_results
    if newres is None:
        return None, dofv_tab
    params = new_model.random_variables.variance_parameters(exclude_level=VariabilityLevel.RUV)
    params = [pharmpy.symbols.symbol(p.name) for p in params]
    origres.reparameterize(
        [
            pharmpy.random_variables.NormalParametrizationSd,
            pharmpy.random_variables.MultivariateNormalParametrizationSdCorr,
        ]
    )
    newres.reparameterize(
        [
            pharmpy.random_variables.NormalParametrizationSd,
            pharmpy.random_variables.MultivariateNormalParametrizationSdCorr,
        ]
    )
    boxcox_sds = [newres.parameter_estimates[p.name] for p in params]
    orig_sds = [origres.parameter_estimates[p.name] for p in params]
    thetas = newres.parameter_estimates[0 : len(params)]
    eta_names = [rv.name for rv in new_model.random_variables.etas]

    table = pd.DataFrame(
        {parameter_name: thetas.values, 'new_sd': boxcox_sds, 'old_sd': orig_sds}, index=eta_names
    )

    dofv = origres.ofv - newres.ofv
    dofv_tab = pd.DataFrame({'dofv': dofv, 'df': len(eta_names)}, index=[transform_name])
    return table, dofv_tab


def calc_fullblock(original_model, fullblock_model):
    """Retrieve new and old parameters of full block"""
    dofv_tab = pd.DataFrame({'dofv': np.nan, 'df': np.nan}, index=['fullblock'])
    if fullblock_model is None:
        return None, dofv_tab
    origres = original_model.modelfit_results
    fullres = fullblock_model.modelfit_results
    if fullres is None:
        return None, dofv_tab
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
    base_path = list(path.glob('*_linbase.mod'))[0]
    base_model = Model(base_path)
    fullblock_path = path / 'modelfit_run' / 'fullblock.mod'
    if fullblock_path.is_file():
        fullblock_model = Model(fullblock_path)
    else:
        fullblock_model = None
    boxcox_path = path / 'modelfit_run' / 'boxcox.mod'
    if boxcox_path.is_file():
        boxcox_model = Model(boxcox_path)
    else:
        boxcox_model = None
    tdist_path = path / 'modelfit_run' / 'tdist.mod'
    if tdist_path.is_file():
        tdist_model = Model(tdist_path)
    else:
        tdist_model = None
    addetas_path = path / 'add_etas_run' / 'add_etas_linbase.mod'
    if addetas_path.is_file():
        addetas_model = Model(addetas_path)
    else:
        addetas_model = None

    frem_path = path / 'frem_run' / 'results.json'
    if frem_path.is_file():
        frem_res = read_results(frem_path)
    else:
        frem_res = None

    args = psn_helpers.options_from_command(psn_helpers.psn_command(path))
    etas_added_to = args['add_etas'].split(',')

    res = calculate_results(
        original_model,
        base_model,
        fullblock_model=fullblock_model,
        boxcox_model=boxcox_model,
        tdist_model=tdist_model,
        add_etas_model=addetas_model,
        etas_added_to=etas_added_to,
        frem_results=frem_res,
    )
    return res
