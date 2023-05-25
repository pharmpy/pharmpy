from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import pharmpy.modeling as modeling
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model import EstimationSteps, Model, Parameters, RandomVariables
from pharmpy.model.external.nonmem.nmtran_parser import NMTranControlStream
from pharmpy.model.external.nonmem.parsing import parse_table_columns
from pharmpy.model.external.nonmem.table import ExtTable, NONMEMTableFile, PhiTable
from pharmpy.model.external.nonmem.update import create_name_map
from pharmpy.results import ModelfitResults
from pharmpy.workflows.log import Log

from .results_file import NONMEMResultsFile


def _parse_modelfit_results(
    path: Optional[Union[str, Path]],
    control_stream: NMTranControlStream,
    name_map,
    model: Model,
    subproblem: Optional[int] = None,
):
    # Path to model file or results file
    if path is None:
        return None

    path = Path(path)

    name = model.name
    description = model.description
    estimation_steps = model.estimation_steps
    parameters = model.parameters
    etas = model.random_variables.etas

    log = Log()
    try:
        try:
            ext_path = path.with_suffix('.ext')
            ext_tables = NONMEMTableFile(ext_path)
        except ValueError:
            log.log_error(f"Broken ext-file {path.with_suffix('.ext')}")
            return None

        for table in ext_tables:
            try:
                table.data_frame
            except ValueError:
                log.log_error(
                    f"Broken table in ext-file {path.with_suffix('.ext')}, "
                    f"table no. {table.number}"
                )
    except (FileNotFoundError, OSError):
        return None

    (
        table_numbers,
        final_ofv,
        ofv_iterations,
        final_pe,
        sdcorr,
        pe_iterations,
        ses,
        ses_sdcorr,
    ) = _parse_ext(control_stream, name_map, ext_tables, subproblem, parameters)

    table_df = _parse_tables(path, control_stream, netas=len(model.random_variables.etas.names))
    residuals = _parse_residuals(table_df)
    predictions = _parse_predictions(table_df)
    iofv, ie, iec = _parse_phi(path, control_stream, name_map, etas, subproblem)
    rse = _calculate_relative_standard_errors(final_pe, ses)
    (
        runtime_total,
        log_likelihood,
        covstatus,
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
    ) = _parse_lst(len(estimation_steps), path, table_numbers, log)

    if table_numbers:
        eststeps = table_numbers
    else:
        eststeps = list(range(1, len(estimation_steps) + 1))
    last_est_ind = _get_last_est(estimation_steps)
    minsucc_iters = pd.Series(
        minimization_successful, index=eststeps, name='minimization_successful'
    )
    esttime_iters = pd.Series(estimation_runtime, index=eststeps, name='estimation_runtime')
    funcevals_iters = pd.Series(function_evaluations, index=eststeps, name='function_evaluations')
    termcause_iters = pd.Series(termination_cause, index=eststeps, name='termination_cause')
    sigdigs_iters = pd.Series(significant_digits, index=eststeps, name='significant_digits')

    if covstatus and ses is not None:
        cov = _parse_matrix(path.with_suffix(".cov"), control_stream, name_map, table_numbers)
        cor = _parse_matrix(path.with_suffix(".cor"), control_stream, name_map, table_numbers)
        if cor is not None:
            np.fill_diagonal(cor.values, 1)
        coi = _parse_matrix(path.with_suffix(".coi"), control_stream, name_map, table_numbers)
    else:
        cov, cor, coi = None, None, None

    cov, cor, coi, ses = calculate_cov_cor_coi_ses(cov, cor, coi, ses)

    evaluation = _parse_evaluation(estimation_steps)

    res = ModelfitResults(
        name=name,
        description=description,
        minimization_successful=minimization_successful[last_est_ind],
        minimization_successful_iterations=minsucc_iters,
        estimation_runtime=estimation_runtime[last_est_ind],
        estimation_runtime_iterations=esttime_iters,
        function_evaluations=function_evaluations[last_est_ind],
        function_evaluations_iterations=funcevals_iters,
        termination_cause=termination_cause[last_est_ind],
        termination_cause_iterations=termcause_iters,
        significant_digits=significant_digits[-1],
        significant_digits_iterations=sigdigs_iters,
        relative_standard_errors=rse,
        individual_estimates=ie,
        individual_estimates_covariance=iec,
        runtime_total=runtime_total,
        log_likelihood=log_likelihood,
        covariance_matrix=cov,
        correlation_matrix=cor,
        precision_matrix=coi,
        standard_errors=ses,
        standard_errors_sdcorr=ses_sdcorr,
        individual_ofv=iofv,
        parameter_estimates=final_pe,
        parameter_estimates_sdcorr=sdcorr,
        parameter_estimates_iterations=pe_iterations,
        ofv=final_ofv,
        ofv_iterations=ofv_iterations,
        predictions=predictions,
        residuals=residuals,
        evaluation=evaluation,
        log=log,
    )
    return res


def calculate_cov_cor_coi_ses(cov, cor, coi, ses):
    if cov is None:
        if cor is not None:
            cov = modeling.calculate_cov_from_corrse(cor, ses)
        elif coi is not None:
            cov = modeling.calculate_cov_from_prec(coi)
    if cor is None:
        if cov is not None:
            cor = modeling.calculate_corr_from_cov(cov)
        elif coi is not None:
            cor = modeling.calculate_corr_from_prec(coi)
    if coi is None:
        if cov is not None:
            coi = modeling.calculate_prec_from_cov(cov)
        elif cor is not None:
            coi = modeling.calculate_prec_from_corrse(cor, ses)
    if ses is None:
        if cov is not None:
            ses = modeling.calculate_se_from_cov(cov)
        elif coi is not None:
            ses = modeling.calculate_se_from_prec(coi)
    return cov, cor, coi, ses


def _parse_matrix(
    path: Path,
    control_stream: NMTranControlStream,
    name_map,
    table_numbers,
):
    try:
        tables = NONMEMTableFile(path)
    except OSError:
        return None

    last_table = tables.table_no(table_numbers[-1])
    assert last_table is not None
    df = last_table.data_frame
    if df is not None:
        df = df.rename(index=name_map)
        df.columns = df.index
    return df


def _empty_lst_results(n: int):
    false_vec = [False] * n
    nan_vec = [np.nan] * n
    none_vec = [None] * n
    return None, np.nan, False, false_vec, nan_vec, nan_vec, none_vec, nan_vec


def _parse_lst(n: int, path: Path, table_numbers, log: Log):
    try:
        rfile = NONMEMResultsFile(path.with_suffix('.lst'), log=log)
    except OSError:
        return _empty_lst_results(n)

    if not table_numbers:
        return _empty_lst_results(n)

    runtime_total = rfile.runtime_total

    try:
        log_likelihood = rfile.table[table_numbers[-1]]['ofv_with_constant']
    except (KeyError, FileNotFoundError):
        log_likelihood = np.nan

    status = rfile.covariance_status(table_numbers[-1])
    covstatus = status['covariance_step_ok']

    (
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
    ) = parse_estimation_status(rfile, table_numbers)

    return (
        runtime_total,
        log_likelihood,
        covstatus,
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
    )


def parse_estimation_status(results_file, table_numbers):
    minimization_successful = []
    function_evaluations = []
    significant_digits = []
    termination_cause = []
    estimation_runtime = []
    for tabno in table_numbers:
        if results_file is not None:
            estimation_status = results_file.estimation_status(tabno)
        else:
            estimation_status = NONMEMResultsFile.unknown_termination()
        minimization_successful.append(estimation_status['minimization_successful'])
        function_evaluations.append(estimation_status['function_evaluations'])
        significant_digits.append(estimation_status['significant_digits'])
        if estimation_status['maxevals_exceeded'] is True:
            tc = 'maxevals_exceeded'
        elif estimation_status['rounding_errors'] is True:
            tc = 'rounding_errors'
        else:
            tc = None
        termination_cause.append(tc)
        try:
            er = results_file.table[tabno]['estimation_runtime']
        except (KeyError, FileNotFoundError):
            er = np.nan
        estimation_runtime.append(er)

    return (
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
    )


def _get_last_est(estimation_steps: EstimationSteps):
    # Find last estimation
    for i in range(len(estimation_steps) - 1, -1, -1):
        step = estimation_steps[i]
        if not step.evaluation:
            return i
    # If all steps were evaluation the last evaluation step is relevant
    return len(estimation_steps) - 1


def _parse_phi(
    path: Path,
    control_stream: NMTranControlStream,
    name_map,
    etas: RandomVariables,
    subproblem=None,
):
    try:
        phi_tables = NONMEMTableFile(path.with_suffix('.phi'))
    except FileNotFoundError:
        return None, None, None
    if subproblem is None:
        table = phi_tables.tables[-1]
    else:
        table = phi_tables.tables[subproblem - 1]

    if table is None:
        return None, None, None

    assert isinstance(table, PhiTable)

    eta_names = set(name_map.values())
    rv_names = list(filter(eta_names.__contains__, etas.names))
    try:
        individual_ofv = table.iofv
        individual_estimates = table.etas.rename(columns=name_map)[rv_names]
        ids, eta_col_names, matrix_array = table.etc_data()
        index = {name_map[x]: i for i, x in enumerate(eta_col_names)}
        indices = tuple(map(index.__getitem__, rv_names))
        selector = np.ix_(indices, indices)
        etc_frames = [
            pd.DataFrame(matrix[selector], columns=rv_names, index=rv_names)
            for matrix in matrix_array
        ]
        covs = pd.Series(etc_frames, index=ids, dtype='object')
        return individual_ofv, individual_estimates, covs
    except KeyError:
        return None, None, None


def _parse_tables(path: Path, control_stream: NMTranControlStream, netas) -> pd.DataFrame:
    """Parse $TABLE and table files into one large dataframe of useful columns"""
    interesting_columns = {
        'ID',
        'TIME',
        'PRED',
        'CIPREDI',
        'CPRED',
        'IPRED',
        'RES',
        'WRES',
        'CWRES',
        'MDV',
    }

    table_recs = control_stream.get_records('TABLE')
    colnames_list = parse_table_columns(control_stream, netas)
    found = set()
    df = pd.DataFrame()
    for table_rec, colnames in zip(table_recs, colnames_list):
        columns_in_table = []
        colnames_in_table = []
        for i, name in enumerate(colnames):
            if name in interesting_columns and name not in found:
                found.add(name)
                colnames_in_table.append(name)
                columns_in_table.append(i)

        noheader = table_rec.has_option("NOHEADER")
        notitle = table_rec.has_option("NOTITLE") or noheader
        nolabel = table_rec.has_option("NOLABEL") or noheader
        table_path = path.parent / table_rec.path
        try:
            table_file = NONMEMTableFile(table_path, notitle=notitle, nolabel=nolabel)
        except IOError:
            continue
        table = table_file.tables[0]

        df[colnames_in_table] = table.data_frame.iloc[:, columns_in_table]

    if 'ID' in df.columns:
        df['ID'] = df['ID'].convert_dtypes()
    return df


def _extract_from_df(df: pd.DataFrame, mandatory, optional):
    # Extract all mandatory and at least one optional column from df
    columns = set(df.columns)
    if not (set(mandatory) <= columns):
        return None

    found_optionals = [col for col in optional if col in columns]
    if not found_optionals:
        return None
    return df[mandatory + found_optionals]


def _parse_residuals(df: pd.DataFrame):
    index_cols = ['ID', 'TIME']
    cols = ['RES', 'WRES', 'CWRES']
    df = _extract_from_df(df, index_cols, cols)
    if df is not None:
        df.set_index(['ID', 'TIME'], inplace=True)
        df = df.loc[(df != 0).any(axis=1)]  # Simple way of removing non-observations
    return df


def _parse_predictions(df: pd.DataFrame):
    index_cols = ['ID', 'TIME']
    cols = ['PRED', 'CIPREDI', 'CPRED', 'IPRED']
    df = _extract_from_df(df, index_cols, cols)
    if df is not None:
        df.set_index(['ID', 'TIME'], inplace=True)
    return df


def _create_failed_ofv_iterations(n: int):
    steps = list(range(1, n + 1))
    iterations = [0] * n
    ofvs = [np.nan] * n
    ofv_iterations = create_ofv_iterations_series(ofvs, steps, iterations)
    return ofv_iterations


def create_ofv_iterations_series(ofv, steps, iterations):
    step_series = pd.Series(steps, dtype='int32', name='steps')
    iteration_series = pd.Series(iterations, dtype='int32', name='iteration')
    ofv_iterations = pd.Series(
        ofv, name='OFV', dtype='float64', index=[step_series, iteration_series]
    )
    return ofv_iterations


def _create_failed_parameter_estimates(parameters: Parameters):
    return pd.Series(np.nan, name='estimates', index=list(parameters.nonfixed.inits.keys()))


def _parse_ext(
    control_stream: NMTranControlStream,
    name_map,
    ext_tables: NONMEMTableFile,
    subproblem: Optional[int],
    parameters: Parameters,
):
    table_numbers = _parse_table_numbers(ext_tables, subproblem)

    final_ofv, ofv_iterations = _parse_ofv(ext_tables, subproblem)
    final_pe, sdcorr, pe_iterations = _parse_parameter_estimates(
        control_stream, name_map, ext_tables, subproblem, parameters
    )
    ses, ses_sdcorr = _parse_standard_errors(control_stream, name_map, ext_tables, parameters)
    return (
        table_numbers,
        final_ofv,
        ofv_iterations,
        final_pe,
        sdcorr,
        pe_iterations,
        ses,
        ses_sdcorr,
    )


def _parse_table_numbers(ext_tables: NONMEMTableFile, subproblem: Optional[int]):
    table_numbers = []
    for table in ext_tables.tables:
        if subproblem and table.subproblem != subproblem:
            continue
        table_numbers.append(table.number)
    return table_numbers


def _parse_ofv(ext_tables: NONMEMTableFile, subproblem: Optional[int]):
    step = []
    iteration = []
    ofv = []
    final_table = None
    for i, table in enumerate(ext_tables.tables, start=1):
        if subproblem and table.subproblem != subproblem:
            continue
        df = _get_iter_df(table.data_frame)
        n = len(df)
        step += [i] * n
        iteration += list(df['ITERATION'])
        ofv += list(df['OBJ'])
        final_table = table

    assert isinstance(final_table, ExtTable)
    if np.isnan(ofv[-1]):
        final_ofv = ofv[-1]
    else:
        final_ofv = final_table.final_ofv
    ofv_iterations = create_ofv_iterations_series(ofv, step, iteration)
    return final_ofv, ofv_iterations


def _get_iter_df(df):
    final_iter = -(10**9)
    iters = df['ITERATION']
    if 0 not in iters.values and final_iter in iters.values:
        df = df[iters == -(10**9)]
        df.at[0, 'ITERATION'] = 0
    else:
        try:
            final_iter_ofv = df[iters == final_iter].iloc[-1].loc['OBJ']
        except IndexError:
            final_iter_ofv = np.nan
        last_iter_ofv = df[iters >= 0].iloc[-1].loc['OBJ']
        if final_iter_ofv != last_iter_ofv:
            new_iter_no = int(df[iters >= 0].iloc[-1].loc['ITERATION']) + 1
            if final_iter in iters.values:
                idx = df[iters == final_iter].index.values[0]
                df.at[idx, 'ITERATION'] = new_iter_no
                new_iter_no += 1
            nan_row_dict = {
                colname: (new_iter_no if colname == 'ITERATION' else np.nan)
                for colname in df.columns
            }
            nan_row = pd.DataFrame(nan_row_dict, index=[len(df)])
            df = pd.concat([df, nan_row])
        df = df[df['ITERATION'] >= 0]
    return df


def _calculate_relative_standard_errors(pe, se):
    if pe is None or se is None:
        ser = None
    else:
        ser = se / pe
        ser.name = 'RSE'
    return ser


def _parse_parameter_estimates(
    control_stream: NMTranControlStream,
    name_map,
    ext_tables: NONMEMTableFile,
    subproblem: Optional[int],
    parameters: Parameters,
):
    pe = pd.DataFrame()
    fixed_param_names = []
    final_table = None
    fix = None
    for i, table in enumerate(ext_tables.tables, start=1):
        if subproblem and table.subproblem != subproblem:
            continue
        df = _get_iter_df(table.data_frame)
        assert isinstance(table, ExtTable)
        fix = _get_fixed_parameters(table, parameters, name_map)
        fixed_param_names = [name for name in list(df.columns)[1:-1] if fix[name]]
        df = df.drop(fixed_param_names + ['OBJ'], axis=1)
        df['step'] = i
        df = df.rename(columns=name_map)
        pe = pd.concat([pe, df])
        final_table = table

    assert fix is not None
    assert final_table is not None
    if pe.iloc[-1].drop(['ITERATION', 'step']).isnull().all():
        final = pe.iloc[-1].drop(['ITERATION', 'step']).rename('estimates')
    else:
        final = final_table.final_parameter_estimates
        final = final.drop(fixed_param_names)
        final = final.rename(index=name_map)
    pe = pe.rename(columns={'ITERATION': 'iteration'}).set_index(['step', 'iteration'])

    try:
        sdcorr = final_table.omega_sigma_stdcorr[~fix]
    except KeyError:
        sdcorr_ests = pd.Series(np.nan, index=pe.index)
    else:
        sdcorr = sdcorr.rename(index=name_map)
        sdcorr_ests = final.copy()
        sdcorr_ests.update(sdcorr)
    return final, sdcorr_ests, pe


def _parse_standard_errors(
    control_stream: NMTranControlStream,
    name_map,
    ext_tables: NONMEMTableFile,
    parameters: Parameters,
):
    table = ext_tables.tables[-1]
    assert isinstance(table, ExtTable)
    try:
        ses = table.standard_errors
    except KeyError:
        return None, None

    fix = _get_fixed_parameters(table, parameters, name_map)
    ses = ses[~fix]
    sdcorr = table.omega_sigma_se_stdcorr[~fix]
    ses = ses.rename(index=name_map)
    sdcorr = sdcorr.rename(index=name_map)
    sdcorr_ses = ses.copy()
    sdcorr_ses.update(sdcorr)
    sdcorr_ses = sdcorr_ses.rename(index=name_map)
    return ses, sdcorr_ses


def _parse_evaluation(estimation_steps: EstimationSteps):
    index = list(range(1, len(estimation_steps) + 1))
    evaluation = [est.evaluation for est in estimation_steps]
    return pd.Series(evaluation, index=index, name='evaluation', dtype='float64')


def _get_fixed_parameters(table: ExtTable, parameters: Parameters, pe_translation: Dict):
    try:
        return table.fixed
    except KeyError:
        # NM 7.2 does not have row -1000000006 indicating FIXED status
        ests = table.final_parameter_estimates
        fixed = pd.Series(parameters.fix)
        # NOTE parameters in result file haven't been renamed yet
        fixed = fixed.rename({value: key for key, value in pe_translation.items()})
        return pd.concat([fixed, pd.Series(True, index=ests.index.difference(fixed.index))])


def simfit_results(model, model_path):
    """Read in modelfit results from a simulation/estimation model"""
    nsubs = model.internals.control_stream.get_records('SIMULATION')[0].nsubs
    results = []
    for i in range(1, nsubs + 1):
        res = parse_modelfit_results(model, model_path, subproblem=i)
        results.append(res)
    return results


# def parse_ext(model, path, subproblem):
#     try:
#         ext_tables = NONMEMTableFile(path.with_suffix('.ext'))
#     except ValueError:
#         failed_pe = _create_failed_parameter_estimates(model.parameters)
#         n = len(model.estimation_steps)
#         df = pd.concat([failed_pe] * n, axis=1).T
#         df['step'] = range(1, n + 1)
#         df['iteration'] = 0
#         df = df.set_index(['step', 'iteration'])
#         return (
#             [],
#             np.nan,
#             _create_failed_ofv_iterations(len(model.estimation_steps)),
#             failed_pe,
#             failed_pe,
#             df,
#             None,
#             None,
#         )
#     return _parse_ext(model.internals.control_stream, ext_tables, subproblem, model.parameters)


def parse_modelfit_results(
    model, path: Optional[Union[str, Path]], subproblem: Optional[int] = None
):
    name_map = create_name_map(model)
    name_map = {value: key for key, value in name_map.items()}
    res = _parse_modelfit_results(
        path,
        model.internals.control_stream,
        name_map,
        model,
        subproblem=subproblem,
    )
    return res
