from pathlib import Path

import pharmpy.modeling as modeling
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.results import ModelfitResults
from pharmpy.workflows.log import Log

from .parameters import parameter_translation
from .random_variables import rv_translation
from .results_file import NONMEMResultsFile
from .table import NONMEMTableFile


def parse_modelfit_results(model, path, subproblem=None):
    # Path to model file or results file
    if path is None:
        return None
    else:
        path = Path(path)

    try:
        res = NONMEMChainedModelfitResults(path, model=model, subproblem=subproblem)
    except (FileNotFoundError, OSError):
        return None

    log = res.log
    table_df = parse_tables(model, path)
    residuals = parse_residuals(table_df)
    predictions = parse_predictions(table_df)
    iofv, ie, iec = parse_phi(model, path)
    (
        table_numbers,
        final_ofv,
        ofv_iterations,
        final_pe,
        sdcorr,
        pe_iterations,
        ses,
        ses_sdcorr,
    ) = parse_ext(model, path, subproblem)
    rse = calculate_relative_standard_errors(final_pe, ses)
    (
        runtime_total,
        log_likelihood,
        covstatus,
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
    ) = parse_lst(model, path, table_numbers, log)

    if table_numbers:
        eststeps = table_numbers
    else:
        eststeps = list(range(1, len(model.estimation_steps) + 1))
    last_est_ind = get_last_est(model)
    minsucc_iters = pd.Series(
        minimization_successful, index=eststeps, name='minimization_successful'
    )
    esttime_iters = pd.Series(estimation_runtime, index=eststeps, name='estimation_runtime')
    funcevals_iters = pd.Series(function_evaluations, index=eststeps, name='function_evaluations')
    termcause_iters = pd.Series(termination_cause, index=eststeps, name='termination_cause')
    sigdigs_iters = pd.Series(significant_digits, index=eststeps, name='significant_digits')

    if covstatus and ses is not None:
        cov = parse_matrix(path.with_suffix(".cov"), model, table_numbers)
        cor = parse_matrix(path.with_suffix(".cor"), model, table_numbers)
        if cor is not None:
            np.fill_diagonal(cor.values, 1)
        coi = parse_matrix(path.with_suffix(".coi"), model, table_numbers)
    else:
        cov, cor, coi = None, None, None

    cov, cor, coi, ses = calculate_cov_cor_coi_ses(cov, cor, coi, ses)

    res.minimization_successful = minimization_successful[last_est_ind]
    res.minimization_successful_iterations = minsucc_iters
    res.estimation_runtime = estimation_runtime[last_est_ind]
    res.estimation_runtime_iterations = esttime_iters
    res.function_evaluations = function_evaluations[last_est_ind]
    res.function_evaluations_iterations = funcevals_iters
    res.termination_cause = termination_cause[last_est_ind]
    res.termination_cause_iterations = termcause_iters
    res.significant_digits = significant_digits[-1]
    res.significant_digits_iterations = sigdigs_iters
    res.relative_standard_errors = rse
    res.individual_estimates = ie
    res.individual_estimates_covariance = iec
    res.runtime_total = runtime_total
    res.log_likelihood = log_likelihood
    res.covariance_matrix = cov
    res.correlation_matrix = cor
    res.information_matrix = coi
    res.standard_errors = ses
    res.standard_errors_sdcorr = ses_sdcorr
    res.individual_ofv = iofv
    res.parameter_estimates = final_pe
    res.parameter_estimates_sdcorr = sdcorr
    res.parameter_estimates_iterations = pe_iterations
    res.ofv = final_ofv
    res.ofv_iterations = ofv_iterations
    res.predictions = predictions
    res.residuals = residuals
    return res


class NONMEMChainedModelfitResults(ChainedModelfitResults):
    def __init__(self, path, model=None, subproblem=None):
        # Path is path to any result file
        self.log = Log()
        path = Path(path)
        self._path = path
        self._read_ext_table()

    def _read_ext_table(self):
        try:
            ext_tables = NONMEMTableFile(self._path.with_suffix('.ext'))
        except ValueError:
            # The ext-file is illegal
            self.log.log_error(f"Broken ext-file {self._path.with_suffix('.ext')}")
            return

        for table in ext_tables:
            try:
                table.data_frame
            except ValueError:
                self.log.log_error(
                    f"Broken table in ext-file {self._path.with_suffix('.ext')}, "
                    f"table no. {table.number}"
                )


def calculate_cov_cor_coi_ses(cov, cor, coi, ses):
    if cov is None:
        if cor is not None:
            cov = modeling.calculate_cov_from_corrse(cor, ses)
        elif coi is not None:
            cov = modeling.calculate_cov_from_inf(coi)
    if cor is None:
        if cov is not None:
            cor = modeling.calculate_corr_from_cov(cov)
        elif coi is not None:
            cor = modeling.calculate_corr_from_inf(coi)
    if coi is None:
        if cov is not None:
            coi = modeling.calculate_inf_from_cov(cov)
        elif cor is not None:
            coi = modeling.calculate_inf_from_corrse(cor, ses)
    if ses is None:
        if cov is not None:
            ses = modeling.calculate_se_from_cov(cov)
        elif coi is not None:
            ses = modeling.calculate_se_from_inf(coi)
    return cov, cor, coi, ses


def parse_matrix(path, model, table_numbers):
    try:
        table = NONMEMTableFile(path)
    except OSError:
        return None

    df = table.table_no(table_numbers[-1]).data_frame
    if df is not None:
        if model:
            df = df.rename(index=parameter_translation(model.internals.control_stream))
            df.columns = df.index
    return df


def empty_lst_results(model):
    n = len(model.estimation_steps)
    false_vec = [False] * n
    nan_vec = [np.nan] * n
    none_vec = [None] * n
    return None, np.nan, False, false_vec, nan_vec, nan_vec, none_vec, nan_vec


def parse_lst(model, path, table_numbers, log):
    try:
        rfile = NONMEMResultsFile(path.with_suffix('.lst'), log=log)
    except OSError:
        return empty_lst_results(model)

    if not table_numbers:
        return empty_lst_results(model)

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


def get_last_est(model):
    est_steps = model.estimation_steps
    # Find last estimation
    for i in range(len(est_steps) - 1, -1, -1):
        step = est_steps[i]
        if not step.evaluation:
            return i
    # If all steps were evaluation the last evaluation step is relevant
    return len(est_steps) - 1


def parse_phi(model, path):
    try:
        phi_tables = NONMEMTableFile(path.with_suffix('.phi'))
    except FileNotFoundError:
        return None, None, None
    table = phi_tables.tables[-1]

    if table is not None:
        trans = rv_translation(model.internals.control_stream, reverse=True)
        rv_names = [name for name in model.random_variables.etas.names if name in trans]
        try:
            individual_ofv = table.iofv
            individual_estimates = table.etas.rename(
                columns=rv_translation(model.internals.control_stream)
            )[rv_names]
            covs = table.etcs
            covs = covs.transform(
                lambda cov: cov.rename(
                    columns=rv_translation(model.internals.control_stream),
                    index=rv_translation(model.internals.control_stream),
                )
            )
            covs = covs.transform(lambda cov: cov[rv_names].loc[rv_names])
            return individual_ofv, individual_estimates, covs
        except KeyError:
            pass
    return None, None, None


def parse_tables(model, path):
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

    table_recs = model.internals.control_stream.get_records('TABLE')
    found = set()
    df = pd.DataFrame()
    for table_rec in table_recs:
        columns_in_table = []
        for key, value in table_rec.all_options:
            if key in interesting_columns and key not in found and value is None:
                # FIXME: Cannot handle synonyms here
                colname = key
            elif value in interesting_columns and value not in found:
                colname = value
            else:
                continue

            found.add(colname)
            columns_in_table.append(colname)

            noheader = table_rec.has_option("NOHEADER")
            notitle = table_rec.has_option("NOTITLE") or noheader
            nolabel = table_rec.has_option("NOLABEL") or noheader
            path = path.parent / table_rec.path
            try:
                table_file = NONMEMTableFile(path, notitle=notitle, nolabel=nolabel)
            except IOError:
                continue
            table = table_file.tables[0]
            df[columns_in_table] = table.data_frame[columns_in_table]

    if 'ID' in df.columns:
        df['ID'] = df['ID'].convert_dtypes()
    return df


def _extract_from_df(df, mandatory, optional):
    # Extract all mandatory and at least one optional column from df
    columns = set(df.columns)
    if not (set(mandatory) <= columns):
        return None

    found_optionals = [col for col in optional if col in columns]
    if not found_optionals:
        return None
    return df[mandatory + found_optionals]


def parse_residuals(df):
    index_cols = ['ID', 'TIME']
    cols = ['RES', 'WRES', 'CWRES']
    df = _extract_from_df(df, index_cols, cols)
    if df is not None:
        df.set_index(['ID', 'TIME'], inplace=True)
        df = df.loc[(df != 0).any(axis=1)]  # Simple way of removing non-observations
    return df


def parse_predictions(df):
    index_cols = ['ID', 'TIME']
    cols = ['PRED', 'CIPREDI', 'CPRED', 'IPRED']
    df = _extract_from_df(df, index_cols, cols)
    if df is not None:
        df.set_index(['ID', 'TIME'], inplace=True)
    return df


def create_failed_ofv_iterations(model):
    steps = list(range(1, len(model.estimation_steps) + 1))
    iterations = [0] * len(steps)
    ofvs = [np.nan] * len(steps)
    ofv_iterations = create_ofv_iterations_series(ofvs, steps, iterations)
    return ofv_iterations


def create_ofv_iterations_series(ofv, steps, iterations):
    step_series = pd.Series(steps, dtype='int32', name='steps')
    iteration_series = pd.Series(iterations, dtype='int32', name='iteration')
    ofv_iterations = pd.Series(
        ofv, name='OFV', dtype='float64', index=[step_series, iteration_series]
    )
    return ofv_iterations


def create_failed_parameter_estimates(model):
    pe = pd.Series(np.nan, name='estimates', index=list(model.parameters.nonfixed.inits.keys()))
    return pe


def parse_ext(model, path, subproblem):
    try:
        ext_tables = NONMEMTableFile(path.with_suffix('.ext'))
    except ValueError:
        failed_pe = create_failed_parameter_estimates(model)
        n = len(model.estimation_steps)
        df = pd.concat([failed_pe] * n, axis=1).T
        df['step'] = range(1, n + 1)
        df['iteration'] = 0
        df = df.set_index(['step', 'iteration'])
        return (
            [],
            np.nan,
            create_failed_ofv_iterations(model),
            failed_pe,
            failed_pe,
            df,
            None,
            None,
        )

    table_numbers = parse_table_numbers(ext_tables, subproblem)

    final_ofv, ofv_iterations = parse_ofv(model, ext_tables, subproblem)
    final_pe, sdcorr, pe_iterations = parse_parameter_estimates(model, ext_tables, subproblem)
    ses, ses_sdcorr = parse_standard_errors(model, ext_tables)
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


def parse_table_numbers(ext_tables, subproblem):
    table_numbers = []
    for table in ext_tables.tables:
        if subproblem and table.subproblem != subproblem:
            continue
        table_numbers.append(table.number)
    return table_numbers


def parse_ofv(model, ext_tables, subproblem):
    step = []
    iteration = []
    ofv = []
    for i, table in enumerate(ext_tables.tables, start=1):
        if subproblem and table.subproblem != subproblem:
            continue
        df = table.data_frame
        df = df[df['ITERATION'] >= 0]
        n = len(df)
        step += [i] * n
        iteration += list(df['ITERATION'])
        ofv += list(df['OBJ'])
        final_table = table
    final_ofv = final_table.final_ofv
    ofv_iterations = create_ofv_iterations_series(ofv, step, iteration)
    return final_ofv, ofv_iterations


def calculate_relative_standard_errors(pe, se):
    if pe is None or se is None:
        ser = None
    else:
        ser = se / pe
        ser.name = 'RSE'
    return ser


def parse_parameter_estimates(model, ext_tables, subproblem):
    pe = pd.DataFrame()
    for i, table in enumerate(ext_tables.tables, start=1):
        if subproblem and table.subproblem != subproblem:
            continue
        df = table.data_frame
        df = df[df['ITERATION'] >= 0]

        fix = get_fixed_parameters(table, model)
        fixed_param_names = [name for name in list(df.columns)[1:-1] if fix[name]]
        df = df.drop(fixed_param_names + ['OBJ'], axis=1)
        df['step'] = i
        if model:
            df = df.rename(columns=parameter_translation(model.internals.control_stream))
        pe = pd.concat([pe, df])

    final = table.final_parameter_estimates
    final = final.drop(fixed_param_names)
    if model:
        final = final.rename(index=parameter_translation(model.internals.control_stream))
    pe = pe.rename(columns={'ITERATION': 'iteration'}).set_index(['step', 'iteration'])

    try:
        sdcorr = table.omega_sigma_stdcorr[~fix]
    except KeyError:
        sdcorr_ests = pd.Series(np.nan, index=pe.index)
    else:
        if model:
            sdcorr = sdcorr.rename(index=parameter_translation(model.internals.control_stream))
        sdcorr_ests = final.copy()
        sdcorr_ests.update(sdcorr)
    return final, sdcorr_ests, pe


def parse_standard_errors(model, ext_tables):
    table = ext_tables.tables[-1]
    try:
        ses = table.standard_errors
    except Exception:
        return None, None

    fix = get_fixed_parameters(table, model)
    ses = ses[~fix]
    sdcorr = table.omega_sigma_se_stdcorr[~fix]
    if model:
        ses = ses.rename(index=parameter_translation(model.internals.control_stream))
        sdcorr = sdcorr.rename(index=parameter_translation(model.internals.control_stream))
    sdcorr_ses = ses.copy()
    sdcorr_ses.update(sdcorr)
    if model:
        sdcorr_ses = sdcorr_ses.rename(index=parameter_translation(model.internals.control_stream))
    return ses, sdcorr_ses


def get_fixed_parameters(table, model):
    try:
        fix = table.fixed
    except KeyError:
        # NM 7.2 does not have row -1000000006 indicating FIXED status
        ests = table.final_parameter_estimates
        if model:
            fixed = pd.Series(model.parameters.fix)
            fix = pd.concat([fixed, pd.Series(True, index=ests.index.difference(fixed.index))])
    return fix


def simfit_results(model, model_path):
    """Read in modelfit results from a simulation/estimation model"""
    nsubs = model.internals.control_stream.get_records('SIMULATION')[0].nsubs
    results = []
    for i in range(1, nsubs + 1):
        res = parse_modelfit_results(model, model_path, subproblem=i)
        results.append(res)
    return results
