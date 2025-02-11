from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional, Union

import pharmpy.modeling as modeling
from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.math import nearest_positive_semidefinite
from pharmpy.model import ExecutionSteps, Model, Parameters, RandomVariables
from pharmpy.model.external.nonmem.nmtran_parser import NMTranControlStream
from pharmpy.model.external.nonmem.parsing import extract_verbatim_derivatives, parse_table_columns
from pharmpy.model.external.nonmem.table import ExtTable, NONMEMTableFile, PhiTable
from pharmpy.model.external.nonmem.update import create_name_map
from pharmpy.workflows.log import Log
from pharmpy.workflows.results import ModelfitResults, SimulationResults

from .results_file import NONMEMResultsFile


def _parse_modelfit_results(
    path: Optional[Union[str, Path]],
    control_stream: NMTranControlStream,
    name_map,
    model: Model,
    strict: bool = False,
    subproblem: Optional[int] = None,
):
    # Path to model file or results file
    if path is None:
        return None

    path = Path(path)

    execution_steps = model.execution_steps
    parameters = model.parameters
    etas = model.random_variables.etas

    log = Log()
    ext_path = path.with_suffix('.ext')
    if not ext_path.is_file():
        msg = f"Couldn't find NONMEM .ext-file at {ext_path}"
        log = log.log_error(msg)
        if strict:
            raise FileNotFoundError(msg)
        return create_failed_results(model, log)

    try:
        try:
            ext_tables = NONMEMTableFile(ext_path)
        except ValueError:
            log = log.log_error(f"Broken ext-file {path.with_suffix('.ext')}")
            return create_failed_results(model, log)

        for table in ext_tables:
            try:
                table.data_frame
            except ValueError:
                log = log.log_error(
                    f"Broken table in ext-file {path.with_suffix('.ext')}, "
                    f"table no. {table.number}"
                )
    except (FileNotFoundError, OSError):
        # FIXME: Can this still happen?
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
        cov_abort,
    ) = _parse_ext(control_stream, name_map, ext_tables, subproblem, parameters)

    table_df = _parse_tables(
        model, path, control_stream, netas=len(model.random_variables.etas.names)
    )  # $TABLEs
    residuals = _parse_residuals(table_df)
    predictions = _parse_predictions(table_df)
    derivatives = _parse_derivatives(table_df, model)
    iofv, ie, iec = _parse_phi(path, control_stream, name_map, etas, model, final_pe, subproblem)
    gradients_iterations, final_zero_gradient, gradients = _parse_grd(
        path, control_stream, name_map, parameters, subproblem
    )
    rse = _calculate_relative_standard_errors(final_pe, ses)
    (
        runtime_total,
        log_likelihood,
        covstatus,
        covstatus_table_number,
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
        estimate_near_boundary,
        log,
        est_table_numbers,
    ) = _parse_lst(len(execution_steps), path, table_numbers, log)

    if est_table_numbers:
        eststeps = est_table_numbers
    else:
        eststeps = list(range(1, len(execution_steps) + 1))
    last_est_ind = _get_last_est(execution_steps)

    minimization_successful = override_minimization_successful(
        minimization_successful, pe_iterations
    )

    minsucc_iters = pd.Series(
        minimization_successful, index=eststeps, name='minimization_successful'
    )
    esttime_iters = pd.Series(estimation_runtime, index=eststeps, name='estimation_runtime')
    funcevals_iters = pd.Series(function_evaluations, index=eststeps, name='function_evaluations')
    termcause_iters = pd.Series(termination_cause, index=eststeps, name='termination_cause')
    sigdigs_iters = pd.Series(significant_digits, index=eststeps, name='significant_digits')

    if covstatus and ses is not None and not cov_abort:
        cov = _parse_matrix(path.with_suffix(".cov"), control_stream, name_map, table_numbers)
        cor = _parse_matrix(path.with_suffix(".cor"), control_stream, name_map, table_numbers)
        if cor is not None:
            np.fill_diagonal(cor.values, 1)
        coi = _parse_matrix(path.with_suffix(".coi"), control_stream, name_map, table_numbers)
    else:
        cov, cor, coi = None, None, None

    cov, cor, coi, ses = calculate_cov_cor_coi_ses(cov, cor, coi, ses)
    if cov is not None:
        cov = nearest_positive_semidefinite(cov)

    evaluation = _parse_evaluation(execution_steps)

    if not model.execution_steps or model.execution_steps[-1].parameter_uncertainty_method is None:
        covstep_successful = None
    elif covstatus:
        covstep_successful = True
    else:
        covstep_successful = False

    warnings = []
    if any(estimate_near_boundary):
        warnings.append('estimate_near_boundary')
    if final_zero_gradient:
        warnings.append('final_zero_gradient')

    indetas = _parse_ets(path, etas, subproblem)

    res = ModelfitResults(
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
        derivatives=derivatives,
        evaluation=evaluation,
        log=log,
        covstep_successful=covstep_successful,
        gradients=gradients,
        gradients_iterations=gradients_iterations,
        warnings=warnings,
        individual_eta_samples=indetas,
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


def _empty_lst_results(n: int, log):
    false_vec = [False] * n
    nan_vec = [np.nan] * n
    none_vec = [None] * n
    return (
        None,
        np.nan,
        False,
        None,
        false_vec,
        nan_vec,
        nan_vec,
        none_vec,
        nan_vec,
        false_vec,
        log,
        nan_vec,
    )


def _parse_lst(n: int, path: Path, table_numbers, log: Log):
    try:
        rfile = NONMEMResultsFile(path.with_suffix('.lst'), log=log)
    except OSError:
        return _empty_lst_results(n, log)

    if not table_numbers:
        return _empty_lst_results(n, log)

    runtime_total = rfile.runtime_total

    covstatus_table_number = table_numbers[-1]

    try:
        for table_number in reversed(table_numbers):
            if "OPTIMALITY" not in rfile.table[table_number]['METH']:
                log_likelihood_table_number = table_number
                break
    except (KeyError, FileNotFoundError):
        log_likelihood_table_number = None

    try:
        log_likelihood = rfile.table[log_likelihood_table_number]['ofv_with_constant']
    except (KeyError, FileNotFoundError):
        log_likelihood = np.nan

    covstatus = rfile.covariance_status(covstatus_table_number)['covariance_step_ok']

    if log_likelihood_table_number is not None:
        est_table_numbers = [
            table_number
            for table_number in table_numbers
            if table_number <= log_likelihood_table_number
        ]
    else:
        est_table_numbers = [table_number for table_number in table_numbers]

    (
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
        estimate_near_boundary,
    ) = parse_estimation_status(rfile, est_table_numbers)

    return (
        runtime_total,
        log_likelihood,
        covstatus,
        covstatus_table_number,
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
        estimate_near_boundary,
        rfile.log,
        est_table_numbers,
    )


def parse_estimation_status(results_file, table_numbers):
    minimization_successful = []
    function_evaluations = []
    significant_digits = []
    termination_cause = []
    estimation_runtime = []
    estimate_near_boundary = []
    for tabno in table_numbers:
        if results_file is not None:
            estimation_status = results_file.estimation_status(tabno)
        else:
            estimation_status = NONMEMResultsFile.unknown_termination()
        minimization_successful.append(estimation_status['minimization_successful'])
        function_evaluations.append(estimation_status['function_evaluations'])
        significant_digits.append(estimation_status['significant_digits'])
        estimate_near_boundary.append(estimation_status['estimate_near_boundary'])
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
        estimate_near_boundary,
    )


def _get_last_est(execution_steps: ExecutionSteps):
    # Find last estimation
    for i in range(len(execution_steps) - 1, -1, -1):
        step = execution_steps[i]
        if not step.evaluation:
            return i
    # If all steps were evaluation the last evaluation step is relevant
    return len(execution_steps) - 1


def _parse_phi(
    path: Path,
    control_stream: NMTranControlStream,
    name_map,
    etas: RandomVariables,
    model,
    pe,
    subproblem=None,
):
    try:
        phi_tables = NONMEMTableFile(path.with_suffix('.phi'))
    except FileNotFoundError:
        return None, None, None
    if subproblem is None:
        table = None

        for t in reversed(phi_tables.tables):
            if t.design_optimality is None:
                table = t
                break
    else:
        table = phi_tables.tables[subproblem - 1]

    if table is None:
        return None, None, None

    assert isinstance(table, PhiTable)

    eta_names = set(name_map.values())
    rv_names = list(filter(eta_names.__contains__, etas.names))
    try:
        individual_ofv = table.iofv
        prefix, individual_estimates = _parse_individual_estimates(model, pe, table, rv_names)
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


def _parse_individual_estimates(model, pe, table, rv_names):
    df = table.etas
    prefix = df.columns[0][0:3]  # PHI or ETA
    d = {f"{prefix}({i})": name for i, name in enumerate(rv_names, start=1)}
    df = df.rename(columns=d)  # [rv_names] needed?
    if prefix == "PHI":
        for i in range(1, len(rv_names) + 1):
            mu = Expr.symbol(f"MU_{i}")
            expr = model.statements.before_odes.full_expression(mu)
            if expr != mu:  # MU is defined in model code
                value = expr.subs(dict(pe)).subs(model.parameters.inits)
                columns = value.free_symbols
                if columns:
                    # MU depends on covariates
                    # NONMEM do not support time varying covariates so we can use baselines
                    colnames = [str(name) for name in columns]
                    baseline_data = modeling.get_baselines(model)[colnames]
                    for j, (_, row) in enumerate(baseline_data.iterrows()):
                        rowcovs = row.to_dict()
                        rowvalue = value.subs(rowcovs)
                        df.iloc[j, i - 1] -= float(rowvalue)
                else:
                    df.iloc[:, i - 1] -= float(value)
    return prefix, df


def _parse_grd(
    path: Path,
    control_stream: NMTranControlStream,
    name_map,
    parameters: Parameters,
    subproblem=None,
):
    try:
        grd_tables = NONMEMTableFile(path.with_suffix('.grd'))
    except FileNotFoundError:
        return None, None, None
    if subproblem is None:
        table = grd_tables.tables[-1]
    else:
        table = grd_tables.tables[subproblem - 1]

    if table is None:
        return None, None, None

    gradients_table = table.data_frame
    old_col_names = table.data_frame.columns.to_list()[1::]
    param_names = [name for name in list(name_map.values()) if name in parameters.names]
    new_col_names = {old_col_names[i]: param_names[i] for i in range(len(old_col_names))}
    gradients_table = gradients_table.rename(columns=new_col_names)
    last_row = gradients_table.tail(1)
    last_row = last_row.drop(columns=['ITERATION'])
    last_row = last_row.squeeze(axis=0).rename('gradients')
    final_zero_gradient = (last_row == 0).any() or last_row.isnull().any()
    return gradients_table, final_zero_gradient, last_row


def _parse_ets(path, etas, subproblem):
    try:
        ets_tables = NONMEMTableFile(path.with_suffix('.ets'))
    except FileNotFoundError:
        return None
    if subproblem is None:
        table = ets_tables.tables[-1]
    else:
        table = ets_tables.tables[subproblem - 1]

    if table is None:
        return None

    df = table.data_frame
    df = df.drop(columns=['SUBJECT_NO'])
    d = {f'ETA({i})': name for i, name in enumerate(etas.names, start=1)}
    d['SAMPLE'] = 'sample'
    df = df.rename(columns=d).set_index(['ID', 'sample'])
    return df


def _parse_tables(
    model: Model, path: Path, control_stream: NMTranControlStream, netas
) -> pd.DataFrame:
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

    derivative_regex = r'[HG]\d+'
    verbatim_derivatives = extract_verbatim_derivatives(control_stream, model.random_variables)
    verbatim_derivativates_names = set(verbatim_derivatives.keys())
    interesting_columns = interesting_columns.union(verbatim_derivativates_names)

    table_recs = control_stream.get_records('TABLE')
    colnames_list = parse_table_columns(control_stream, netas)
    found = set()
    df = pd.DataFrame()
    for table_rec, colnames in zip(table_recs, colnames_list):
        columns_in_table = []
        colnames_in_table = []
        for i, name in enumerate(colnames):
            if (
                name in interesting_columns or re.match(derivative_regex, name)
            ) and name not in found:
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


def _extract_at_least_one_column(df: pd.DataFrame, cols):
    # Extract at least one column from df
    found_cols = [col for col in cols if col in df.columns]
    if not found_cols:
        return None
    return df[found_cols]


def _parse_residuals(df: pd.DataFrame):
    cols = ['RES', 'WRES', 'CWRES']
    df = _extract_at_least_one_column(df, cols)
    if df is not None:
        df = df.loc[(df != 0).any(axis=1)]  # Simple way of removing non-observations
    return df


def _parse_predictions(df: pd.DataFrame):
    cols = ['PRED', 'CIPREDI', 'CPRED', 'IPRED']
    df = _extract_at_least_one_column(df, cols)
    return df


def _parse_derivatives(df: pd.DataFrame, model: Model):
    cols = []
    derivative_regex = r'[HG]\d+'
    verbatim_derivatives = extract_verbatim_derivatives(
        model.internals.control_stream, model.random_variables
    )
    verbatim_derivatives_names = list(verbatim_derivatives.keys())
    cols += verbatim_derivatives_names

    rename_derivative_dict = {}  # Name is ";".join of names as stored in EstimationStep
    for col in df.columns:
        if re.match(derivative_regex, col):
            if match := re.match(r'H(\d+)1', col):
                param_name = model.random_variables.epsilons.names[int(match.group(1)) - 1]
                rename_derivative_dict[col] = param_name
            elif match := re.match(r'G(\d+)1', col):
                param_name = model.random_variables.etas.names[int(match.group(1)) - 1]
                rename_derivative_dict[col] = param_name
            cols.append(col)
        elif col in verbatim_derivatives_names:
            param_name = verbatim_derivatives[col]
            param_name = map(str, param_name)
            rename_derivative_dict[col] = ";".join(param_name)

    df = _extract_at_least_one_column(df, cols)
    if df is not None:
        df = df.rename(rename_derivative_dict, axis=1)
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
    with warnings.catch_warnings():
        # Needed because pandas 2.1.1 uses the _data attribute that it
        # also has deprecated
        warnings.simplefilter("ignore")
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
    ses, ses_sdcorr, cov_abort = _parse_standard_errors(
        control_stream, name_map, ext_tables, parameters, final_pe
    )
    return (
        table_numbers,
        final_ofv,
        ofv_iterations,
        final_pe,
        sdcorr,
        pe_iterations,
        ses,
        ses_sdcorr,
        cov_abort,
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
    for table_number, table in enumerate(ext_tables.tables, start=1):
        if (subproblem and table.subproblem != subproblem) or table.design_optimality is not None:
            continue
        df = _get_iter_df(table.data_frame)
        n = len(df)
        step += [table_number] * n
        iteration += list(df['ITERATION'])
        ofv += list(df['OBJ'])
        final_table = table

    assert isinstance(final_table, ExtTable)
    if np.isnan(ofv[-1]):
        final_ofv = ofv[-1]
    else:
        final_ofv = final_table.final_ofv
    ofv_iterations = create_ofv_iterations_series(ofv, step, iteration)
    return float(final_ofv), ofv_iterations


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
    for table_number, table in enumerate(ext_tables.tables, start=1):
        if (subproblem and table.subproblem != subproblem) or table.design_optimality is not None:
            continue
        df = _get_iter_df(table.data_frame)
        assert isinstance(table, ExtTable)
        fix = _get_fixed_parameters(table, parameters, name_map)
        fixed_param_names = [name for name in list(df.columns)[1:-1] if fix[name]]
        df = df.drop(fixed_param_names + ['OBJ'], axis=1)
        df['step'] = table_number
        df = df.rename(columns=name_map)
        pe = pd.concat([pe, df])
        final_table = table

    assert fix is not None
    assert final_table is not None
    if pe.iloc[-1].drop(['ITERATION', 'step']).isnull().all():
        final_pe = pe.iloc[-1].drop(['ITERATION', 'step']).rename('estimates')
    else:
        final_pe = final_table.final_parameter_estimates
        final_pe = final_pe.drop(fixed_param_names)
        final_pe = final_pe.rename(index=name_map)
    pe = pe.rename(columns={'ITERATION': 'iteration'}).set_index(['step', 'iteration'])

    try:
        sdcorr = final_table.omega_sigma_stdcorr[~fix]
    except KeyError:
        sdcorr_ests = pd.Series(np.nan, index=pe.index)
    else:
        sdcorr = sdcorr.rename(index=name_map)
        sdcorr_ests = final_pe.copy()
        sdcorr_ests.update(sdcorr)
    return final_pe, sdcorr_ests, pe


def _parse_standard_errors(
    control_stream: NMTranControlStream,
    name_map,
    ext_tables: NONMEMTableFile,
    parameters: Parameters,
    pe: pd.Series,
):
    cov_abort = False
    table = ext_tables.tables[-1]
    assert isinstance(table, ExtTable)
    try:
        ses = table.standard_errors
    except KeyError:
        ses = pd.Series(np.nan, index=pe.index, name="SE")
        sdcorr_ses = pd.Series(np.nan, index=pe.index, name="SE_sdcorr")
        return ses, sdcorr_ses, cov_abort

    fix = _get_fixed_parameters(table, parameters, name_map)
    ses = ses[~fix]
    try:  # if line -1000000001 and -1000000005 exist in .ext file
        sdcorr = table.omega_sigma_se_stdcorr[~fix]
    except KeyError:  # if only line -1000000001 exists but not -1000000005
        ses = pd.Series(np.nan, index=pe.index, name="SE")
        sdcorr_ses = pd.Series(np.nan, index=pe.index, name="SE_sdcorr")
        cov_abort = True
        return ses, sdcorr_ses, cov_abort
    else:
        ses = ses.rename(index=name_map)
        sdcorr = sdcorr.rename(index=name_map)
        sdcorr_ses = ses.copy()
        sdcorr_ses.update(sdcorr)
        sdcorr_ses = sdcorr_ses.rename(index=name_map)
    return ses, sdcorr_ses, cov_abort


def _parse_evaluation(execution_steps: ExecutionSteps):
    index = list(range(1, len(execution_steps) + 1))
    evaluation = [est.evaluation for est in execution_steps]
    return pd.Series(evaluation, index=index, name='evaluation', dtype='float64')


def _get_fixed_parameters(table: ExtTable, parameters: Parameters, pe_translation: dict):
    try:
        return table.fixed
    except KeyError:
        # NM 7.2 does not have row -1000000006 indicating FIXED status
        ests = table.final_parameter_estimates
        fixed = pd.Series(parameters.fix)
        # NOTE: Parameters in result file haven't been renamed yet
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
#         n = len(model.execution_steps)
#         df = pd.concat([failed_pe] * n, axis=1).T
#         df['step'] = range(1, n + 1)
#         df['iteration'] = 0
#         df = df.set_index(['step', 'iteration'])
#         return (
#             [],
#             np.nan,
#             _create_failed_ofv_iterations(len(model.execution_steps)),
#             failed_pe,
#             failed_pe,
#             df,
#             None,
#             None,
#         )
#     return _parse_ext(model.internals.control_stream, ext_tables, subproblem, model.parameters)


def parse_modelfit_results(
    model, path: Optional[Union[str, Path]], strict=False, subproblem: Optional[int] = None
):
    name_map = create_name_map(model)
    name_map = {value: key for key, value in name_map.items()}
    res = _parse_modelfit_results(
        path,
        model.internals.control_stream,
        name_map,
        model,
        strict=strict,
        subproblem=subproblem,
    )
    return res


def _parse_table_file(model, path: Optional[Union[str, Path]], subproblem: Optional[int] = None):
    table_recs = model.internals.control_stream.get_records('TABLE')
    df = pd.DataFrame()
    for table_rec in table_recs:
        noheader = table_rec.has_option("NOHEADER")
        notitle = table_rec.has_option("NOTITLE") or noheader
        nolabel = table_rec.has_option("NOLABEL") or noheader
        table_path = path.parent / table_rec.path
        try:
            table_file = NONMEMTableFile(table_path, notitle=notitle, nolabel=nolabel)
        except IOError:
            continue
        for i in range(len(table_file)):
            sim = table_file.tables[i].data_frame[['DV']].copy()
            assert 'DV' in sim.columns
            sim['SIM'] = i + 1
            sim['index'] = np.arange(len(model.dataset))
            df = pd.concat([df, sim], ignore_index=True)

    if not df.empty:
        df = df.set_index(['SIM', 'index'])
    return df


def parse_simulation_results(
    model, path: Optional[Union[str, Path]], subproblem: Optional[int] = None
):
    table = _parse_table_file(model, path=path, subproblem=subproblem)
    res = SimulationResults(table=table)
    return res


def override_minimization_successful(minimization_successful, pe_iterations):
    # NONMEM could return infinity as parameter estimate even if minimization
    # was successful. We set minimization successful to False in these cases.
    # This reduces the need for special cases further downstream.

    new_minsucc = []
    for i, minsucc in enumerate(minimization_successful):
        try:
            ests_for_iteration = pe_iterations.loc[i + 1].iloc[-1]
        except KeyError:
            new_minsucc.append(minsucc)
            continue
        have_inf = np.isinf(ests_for_iteration).any()
        if have_inf:
            new_minsucc.append(False)
        else:
            new_minsucc.append(minsucc)
    return new_minsucc


def create_failed_results(model, log):
    return ModelfitResults(
        minimization_successful=False,
        ofv=float("NaN"),
        parameter_estimates=_create_failed_parameter_estimates(model.parameters),
        log=log,
    )
