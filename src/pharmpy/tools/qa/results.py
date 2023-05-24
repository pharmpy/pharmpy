from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

import pharmpy.tools.psn_helpers as psn_helpers
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.model import Model, Results
from pharmpy.results import read_results
from pharmpy.tools import read_modelfit_results


@dataclass(frozen=True)
class QAResults(Results):
    dofv: Optional[Any] = None
    fullblock_parameters: Optional[Any] = None
    boxcox_parameters: Optional[Any] = None
    tdist_parameters: Optional[Any] = None
    add_etas_parameters: Optional[Any] = None
    iov_parameters: Optional[Any] = None
    influential_individuals: Optional[Any] = None
    covariate_effects: Optional[Any] = None
    univariate_sum: Optional[Any] = None
    residual_error: Optional[Any] = None
    structural_bias: Optional[Any] = None


def calculate_results(
    original_model,
    base_model,
    fullblock_model=None,
    boxcox_model=None,
    tdist_model=None,
    add_etas_model=None,
    iov_model=None,
    etas_added_to=None,
    frem_results=None,
    cdd_results=None,
    scm_results=None,
    simeval_results=None,
    resmod_idv_results=None,
    **kwargs,
):
    fullblock_table, fullblock_dofv = calc_fullblock(original_model, fullblock_model)
    boxcox_table, boxcox_dofv = calc_transformed_etas(
        original_model, boxcox_model, 'boxcox', 'lambda'
    )
    tdist_table, tdist_dofv = calc_transformed_etas(original_model, tdist_model, 'tdist', 'df')
    addetas_table, addetas_dofv = calc_add_etas(original_model, add_etas_model, etas_added_to)
    iov_table, iov_dofv = calc_iov(original_model, iov_model)
    frem_dofv = calc_frem_dofv(base_model, fullblock_model, frem_results)
    univariate_sum, scm_table, scm_dofv = calc_scm_dofv(scm_results)
    infinds, cdd_dofv = influential_individuals(cdd_results)
    _, simeval_dofv = outliers(simeval_results, cdd_results)
    resmodtab, resmod_dofv = resmod(resmod_idv_results)

    dofv_table = pd.concat(
        [
            fullblock_dofv,
            boxcox_dofv,
            tdist_dofv,
            addetas_dofv,
            iov_dofv,
            frem_dofv,
            scm_dofv,
            cdd_dofv,
            simeval_dofv,
            resmod_dofv,
        ]
    )
    dofv_table.set_index(['section', 'run', 'dvid'], inplace=True)

    return QAResults(
        dofv=dofv_table,
        fullblock_parameters=fullblock_table,
        boxcox_parameters=boxcox_table,
        tdist_parameters=tdist_table,
        add_etas_parameters=addetas_table,
        iov_parameters=iov_table,
        influential_individuals=infinds,
        covariate_effects=scm_table,
        univariate_sum=univariate_sum,
        residual_error=resmodtab,
        **kwargs,
    )


def outliers(simeval_res, cdd_res):
    dofv_tab = pd.DataFrame(
        {
            'section': ['outliers'],
            'run': ['1'],
            'dofv': [np.nan],
            'df': [np.nan],
        }
    )
    if simeval_res is None or cdd_res is None:
        return None, dofv_tab
    iofv = simeval_res.iofv_summary
    outliers = list(iofv.loc[iofv['residual_outlier']].index)
    cases = cdd_res.case_results.copy()
    cases['skipped_individuals'] = cases['skipped_individuals'].transform(lambda x: int(x[0]))
    cases.reset_index(inplace=True)
    cases.set_index('skipped_individuals', inplace=True)
    dofv = cases.loc[outliers].delta_ofv
    top_three = dofv.sort_values(ascending=False).iloc[:3]
    dofv_tab = pd.DataFrame(
        {
            'section': ['outliers'] * len(top_three),
            'run': list(top_three.index),
            'dofv': top_three,
            'df': [np.nan] * len(top_three),
        }
    )

    return None, dofv_tab


def influential_individuals(cdd_res):
    dofv_tab = pd.DataFrame(
        {
            'section': ['influential_individuals'],
            'run': ['1'],
            'dofv': [np.nan],
            'df': [np.nan],
        }
    )
    if cdd_res is None:
        return None, dofv_tab
    df = cdd_res.case_results
    df = df.loc[df['delta_ofv'] > 3.84]
    skipped = [e[0] for e in df['skipped_individuals']]
    influentials = pd.DataFrame({'delta_ofv': df['delta_ofv']}, index=skipped)
    influentials.index.name = 'ID'
    top_three = influentials.sort_values(by=['delta_ofv']).iloc[:3]
    dofv_tab = pd.DataFrame(
        {
            'section': ['influential_individuals'] * len(top_three),
            'run': list(top_three.index),
            'dofv': top_three['delta_ofv'],
            'df': [np.nan] * len(top_three),
        }
    )
    return influentials, dofv_tab


def resmod(res):
    dofv_tab = pd.DataFrame(
        {
            'section': ['residual_error_model'],
            'run': [np.nan],
            'dvid': [np.nan],
            'dofv': [np.nan],
            'df': [np.nan],
        }
    )
    if res is None:
        return None, dofv_tab
    df = res.cwres_models.copy()
    df = df.droplevel(0)
    df.drop('sum', level='DVID', errors='ignore', inplace=True)
    df['dOFV'] = -df['dOFV']
    df.drop(
        ['idv_varying_RUV', 'idv_varying_combined', 'idv_varying_theta'],
        level='model',
        inplace=True,
    )

    # Select the best idv_varying cut for each DVID
    remaining_timevar = []
    for dvid in df.index.unique(level='DVID'):
        subdf = df.loc[dvid]
        idvvar = subdf.index.str.startswith("idv_varying")
        best_timevar_idx = subdf.loc[idvvar]['dOFV'].idxmax()
        for name in subdf.loc[idvvar].index:
            if name != best_timevar_idx:
                df.drop((dvid, name), inplace=True)
        remaining_timevar.append(best_timevar_idx)
    for ind in remaining_timevar:
        df.rename(index={ind: 'time_varying'}, level='model', inplace=True)

    df = df.groupby(level='DVID', sort=False).apply(
        lambda x: x.sort_values(['dOFV'], ascending=False)
    )
    df = df.droplevel(0)  # FIXME: Why was an extra DVID level added?
    n = df['parameters'].apply(lambda x: len(x))
    df.insert(1, 'additional_parameters', n)
    for dvid in df.index.unique(level='DVID'):
        df.loc[(dvid, 'time_varying'), 'additional_parameters'] = 2

    inds = list(df.index[:2])
    dvid2 = [dvid for dvid, _ in inds]
    run2 = [run for _, run in inds]

    dofv_tab = pd.DataFrame(
        {
            'section': ['residual_error_model'] * 2,
            'run': run2,
            'dvid': dvid2,
            'dofv': list(df['dOFV'].iloc[:2]),
            'df': list(df['additional_parameters'].iloc[:2]),
        }
    )

    return df, dofv_tab


def calc_iov(original_model, iov_model):
    dofv_tab = pd.DataFrame(
        {
            'section': ['parameter_variability'],
            'run': ['iov'],
            'dvid': [np.nan],
            'dofv': [np.nan],
            'df': [np.nan],
        }
    )
    if iov_model is None:
        return None, dofv_tab
    iov_res = iov_model.modelfit_results
    if iov_res is None:
        return None, dofv_tab
    origres = original_model.modelfit_results
    iov_params = iov_model.random_variables.iov.variance_parameters
    iov_sds = [iov_res.parameter_estimates_sdcorr[param] for param in iov_params]
    iiv_params = iov_model.random_variables.iiv.variance_parameters
    iiv_params = [param for param in iiv_params if not original_model.parameters[param].fix]
    new_iiv_sds = [iov_res.parameter_estimates_sdcorr[param] for param in iiv_params]
    old_iiv_sds = [origres.parameter_estimates_sdcorr[param] for param in iiv_params]

    etas = []
    for dist in original_model.random_variables.iiv:
        if not set(iiv_params).isdisjoint({s.name for s in dist.free_symbols}):
            etas.extend(dist.names)

    table = pd.DataFrame(
        {'new_iiv_sd': new_iiv_sds, 'orig_iiv_sd': old_iiv_sds, 'iov_sd': iov_sds}, index=etas
    )

    dofv = origres.ofv - iov_res.ofv
    dofv_tab = pd.DataFrame(
        {
            'section': ['parameter_variability'],
            'run': ['iov'],
            'dvid': [np.nan],
            'dofv': [dofv],
            'df': [len(iov_params)],
        }
    )
    return table, dofv_tab


def calc_add_etas(original_model, add_etas_model, etas_added_to):
    dofv_tab = pd.DataFrame(
        {
            'section': ['parameter_variability'],
            'run': ['add_etas'],
            'dvid': [np.nan],
            'dofv': [np.nan],
            'df': [np.nan],
        }
    )
    if add_etas_model is None:
        return None, dofv_tab
    add_etas_res = add_etas_model.modelfit_results
    if add_etas_res is None:
        return None, dofv_tab
    origres = original_model.modelfit_results
    original_etas = original_model.random_variables.etas.names
    all_etas = original_etas + etas_added_to
    added = [True] * len(original_etas) + [False] * len(etas_added_to)
    params = add_etas_model.random_variables.etas.variance_parameters
    params = [sympy.Symbol(p) for p in params]
    orig_params = original_model.random_variables.etas.variance_parameters
    orig_params = [sympy.Symbol(p) for p in orig_params]
    add_etas_sds = [add_etas_res.parameter_estimates_sdcorr[p.name] for p in params]
    orig_sds = [origres.parameter_estimates_sdcorr[p.name] for p in orig_params]
    orig_sds += [np.nan] * len(etas_added_to)
    table = pd.DataFrame(
        {'added': added, 'new_sd': add_etas_sds, 'orig_sd': orig_sds}, index=all_etas
    )
    dofv = origres.ofv - add_etas_res.ofv
    dofv_tab = pd.DataFrame(
        {
            'section': ['parameter_variability'],
            'run': ['add_etas'],
            'dvid': [np.nan],
            'dofv': [dofv],
            'df': [len(etas_added_to)],
        }
    )
    return table, dofv_tab


def calc_scm_dofv(scm_results):
    dofv_tab = pd.DataFrame(
        {
            'section': ['covariates'],
            'run': ['scm'],
            'dvid': [np.nan],
            'dofv': [np.nan],
            'df': [np.nan],
        }
    )
    if scm_results is None:
        return None, None, dofv_tab
    table = scm_results.steps.copy()
    table.index = table.index.droplevel(3)
    table.index = table.index.droplevel(0)
    univariate_sum = table['ofv_drop'].sum()
    top = table.sort_values(by=['ofv_drop']).iloc[-1:]
    table['coeff'] = [list(coveff.values())[0] for coveff in table['covariate_effects']]
    table = table[['ofv_drop', 'coeff']]
    table.columns = ['dofv', 'coeff']
    table.rename(mapper=lambda name: f'ETA({name[2:]})', level=0, inplace=True)
    dofv_tab = pd.DataFrame(
        {
            'section': ['covariates'],
            'run': top['model'].values,
            'dvid': [np.nan],
            'dofv': top['ofv_drop'].values,
            'df': top['delta_df'].values,
        }
    )
    return univariate_sum, table, dofv_tab


def calc_frem_dofv(base_model, fullblock_model, frem_results):
    """Calculate the dOFV for the frem model"""
    dofv_tab = pd.DataFrame(
        {
            'section': ['covariates'],
            'run': ['frem'],
            'dvid': [np.nan],
            'dofv': [np.nan],
            'df': [np.nan],
        }
    )
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
    dofv_tab = pd.DataFrame(
        {
            'section': ['covariates'],
            'run': ['frem'],
            'dvid': [np.nan],
            'dofv': [dofv],
            'df': [npar * ncov],
        }
    )
    return dofv_tab


def calc_transformed_etas(original_model, new_model, transform_name, parameter_name):
    """Retrieve new and old parameters of boxcox"""
    dofv_tab = pd.DataFrame(
        {
            'section': ['parameter_variability'],
            'run': [transform_name],
            'dvid': [np.nan],
            'dofv': [np.nan],
            'df': [np.nan],
        }
    )
    if new_model is None:
        return None, dofv_tab
    origres = original_model.modelfit_results
    newres = new_model.modelfit_results
    if newres is None:
        return None, dofv_tab
    params = new_model.random_variables.etas.variance_parameters
    params = [sympy.Symbol(p) for p in params]
    boxcox_sds = [newres.parameter_estimates_sdcorr[p.name] for p in params]
    orig_sds = [origres.parameter_estimates_sdcorr[p.name] for p in params]
    thetas = newres.parameter_estimates_sdcorr[0 : len(params)]
    eta_names = new_model.random_variables.etas.names

    table = pd.DataFrame(
        {parameter_name: thetas.values, 'new_sd': boxcox_sds, 'old_sd': orig_sds}, index=eta_names
    )

    dofv = origres.ofv - newres.ofv
    dofv_tab = pd.DataFrame(
        {
            'section': ['parameter_variability'],
            'run': [transform_name],
            'dofv': [dofv],
            'df': [len(eta_names)],
        }
    )
    return table, dofv_tab


def calc_fullblock(original_model, fullblock_model):
    """Retrieve new and old parameters of full block"""
    dofv_tab = pd.DataFrame(
        {
            'section': ['parameter_variability'],
            'run': ['fullblock'],
            'dvid': [np.nan],
            'dofv': [np.nan],
            'df': [np.nan],
        }
    )
    if fullblock_model is None:
        return None, dofv_tab
    origres = original_model.modelfit_results
    fullres = fullblock_model.modelfit_results
    if fullres is None:
        return None, dofv_tab
    dist = fullblock_model.random_variables.iiv[0]
    fullblock_parameters = [str(symb) for symb in dist.variance.free_symbols]
    new_params = (
        fullres.parameter_estimates_sdcorr[fullblock_parameters]
        .reindex(index=fullres.parameter_estimates_sdcorr.index)
        .dropna()
    )
    old_params = origres.parameter_estimates_sdcorr
    table = pd.DataFrame({'new': new_params, 'old': old_params}).reindex(index=new_params.index)

    degrees = table['old'].isna().sum()
    dofv = origres.ofv - fullres.ofv
    dofv_tab = pd.DataFrame(
        {
            'section': ['parameter_variability'],
            'run': ['fullblock'],
            'dvid': [np.nan],
            'dofv': [dofv],
            'df': [degrees],
        }
    )
    return table, dofv_tab


def read_results_summary(path):
    summary_path = path / 'results_summary.yaml'
    if not summary_path.is_file():
        return
    with open(summary_path, 'r') as fh:
        summary = yaml.safe_load(fh)
    dfs = []
    for idvtab in summary['structural']:
        idv = idvtab['idv']
        dvid = idvtab['dvid']
        if dvid == 'NA':
            dvid = 1
        cwres = [float(num) for num in idvtab['second_table']['CWRES']]
        cpred = [int(num) for num in idvtab['second_table']['%CPRED']]
        binmin = idvtab['table']['bin_min']
        binmax = idvtab['table']['bin_max']
        df = pd.DataFrame(
            {
                'idv': idv,
                'dvid': dvid,
                'binid': list(range(1, len(cpred) + 1)),
                'binmin': binmin,
                'binmax': binmax,
                'cwres': cwres,
                'cpred': cpred,
            }
        )
        dfs.append(df)
    bias = pd.concat(dfs)
    bias.set_index(['idv', 'dvid', 'binid'], inplace=True)
    return bias


def psn_qa_results(path):
    """Create qa results from a PsN qa run

    :param path: Path to PsN qa run directory
    :return: A :class:`QAResults` object
    """
    path = Path(path)

    original_model = Model.parse_model(path / 'linearize_run' / 'scm_dir1' / 'derivatives.mod')
    orig_res = read_modelfit_results(path / 'linearize_run' / 'scm_dir1' / 'derivatives.mod')
    original_model = original_model.replace(modelfit_results=orig_res)

    base_path = list(path.glob('*_linbase.mod'))[0]
    base_model = Model.parse_model(base_path)
    base_res = Model.parse_model(base_path)
    base_model = base_model.replace(modelfit_results=base_res)

    fullblock_path = path / 'modelfit_run' / 'fullblock.mod'
    if fullblock_path.is_file():
        fullblock_model = Model.parse_model(fullblock_path)
        fb_res = read_modelfit_results(fullblock_path)
        fullblock_model = fullblock_model.replace(modelfit_results=fb_res)
    else:
        fullblock_model = None

    boxcox_path = path / 'modelfit_run' / 'boxcox.mod'
    if boxcox_path.is_file():
        boxcox_model = Model.parse_model(boxcox_path)
        bc_res = read_modelfit_results(boxcox_path)
        boxcox_model = boxcox_model.replace(modelfit_results=bc_res)
    else:
        boxcox_model = None

    tdist_path = path / 'modelfit_run' / 'tdist.mod'
    if tdist_path.is_file():
        tdist_model = Model.parse_model(tdist_path)
        td_res = read_modelfit_results(tdist_path)
        tdist_model = tdist_model.replace(modelfit_results=td_res)
    else:
        tdist_model = None

    addetas_path = path / 'add_etas_run' / 'add_etas_linbase.mod'
    if addetas_path.is_file():
        addetas_model = Model.parse_model(addetas_path)
        ae_res = read_modelfit_results(addetas_path)
        addetas_model = addetas_model.replace(modelfit_results=ae_res)
    else:
        addetas_model = None

    iov_path = path / 'modelfit_run' / 'iov.mod'
    if iov_path.is_file():
        iov_model = Model.parse_model(iov_path)
        iov_res = read_modelfit_results(iov_path)
        iov_model = iov_model.replace(modelfit_results=iov_res)
    else:
        iov_model = None

    frem_path = path / 'frem_run' / 'results.json'
    if frem_path.is_file():
        frem_res = read_results(frem_path)
    else:
        frem_res = None

    cdd_path = path / 'cdd_run' / 'results.json'
    if cdd_path.is_file():
        cdd_res = read_results(cdd_path)
    else:
        cdd_res = None

    scm_path = path / 'scm_run' / 'results.json'
    if scm_path.is_file():
        scm_res = read_results(scm_path)
    else:
        scm_res = None

    simeval_path = path / 'simeval_run' / 'results.json'
    if simeval_path.is_file():
        simeval_res = read_results(simeval_path)
    else:
        simeval_res = None

    args = psn_helpers.options_from_command(psn_helpers.psn_command(path))
    if 'add_etas' not in args:
        etas_added_to = None
    else:
        etas_added_to = args['add_etas'].split(',')

    idv = args.get('resmod_idv', 'TIME')
    resmod_idv_path = path / f'resmod_{idv}' / 'results.json'
    if resmod_idv_path.is_file():
        resmod_idv_res = read_results(resmod_idv_path)
    else:
        resmod_idv_res = None

    bias = read_results_summary(path)

    return calculate_results(
        original_model,
        base_model,
        fullblock_model=fullblock_model,
        boxcox_model=boxcox_model,
        tdist_model=tdist_model,
        add_etas_model=addetas_model,
        iov_model=iov_model,
        etas_added_to=etas_added_to,
        frem_results=frem_res,
        cdd_results=cdd_res,
        scm_results=scm_res,
        simeval_results=simeval_res,
        resmod_idv_results=resmod_idv_res,
        structural_bias=bias,
    )
