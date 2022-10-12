from pathlib import Path

import pharmpy.modeling as modeling
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.plugins.nonmem.parsing import parameter_translation
from pharmpy.plugins.nonmem.results_file import NONMEMResultsFile
from pharmpy.plugins.nonmem.table import NONMEMTableFile
from pharmpy.plugins.nonmem.update import rv_translation
from pharmpy.results import ChainedModelfitResults, ModelfitResults
from pharmpy.workflows.log import Log


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

    cov = res[-1].covariance_matrix
    cor = res[-1].correlation_matrix
    coi = res[-1].information_matrix
    ses = res[-1].standard_errors

    table_df = parse_tables(model, path)
    residuals = parse_residuals(table_df)
    predictions = parse_predictions(table_df)
    iofv, ie, iec = parse_phi(model, path)
    table_numbers, final_ofv, ofv_iterations, final_pe, sdcorr, pe_iterations = parse_ext(
        model, path, subproblem
    )
    rse = calculate_relative_standard_errors(final_pe, ses)
    runtime_total, log_likelihood = parse_lst(path, table_numbers)

    cov, cor, coi, ses = calculate_cov_cor_coi_ses(cov, cor, coi, ses)

    res.ofv = final_ofv
    res.ofv_iterations = ofv_iterations
    res.parameter_estimates = final_pe
    res.parameter_estimates_sdcorr = sdcorr
    res.parameter_estimates_iterations = pe_iterations
    res.relative_standard_errors = rse
    res.individual_ofv = iofv
    res.individual_estimates = ie
    res.individual_estimates_covariance = iec
    res.predictions = predictions
    res.residuals = residuals
    res.runtime_total = runtime_total
    res.log_likelihood = log_likelihood
    res.covariance_matrix = cov
    res.correlation_matrix = cor
    res.information_matrix = coi
    res.standard_errors = ses
    return res


class NONMEMModelfitResults(ModelfitResults):
    def __init__(self, chain):
        self._chain = chain
        super().__init__()

    def _set_covariance_status(self, results_file, table_with_cov=None):
        covariance_status = {
            'requested': True
            if self.standard_errors is not None
            else (table_with_cov == self.table_number),
            'completed': (self.standard_errors is not None),
            'warnings': None,
        }
        if self.standard_errors is not None and results_file is not None:
            status = results_file.covariance_status(self.table_number)
            if status['covariance_step_ok'] is not None:
                covariance_status['warnings'] = not status['covariance_step_ok']

        self._covariance_status = covariance_status

    def _set_estimation_status(self, results_file, requested):
        estimation_status = {'requested': requested}
        status = NONMEMResultsFile.unknown_termination()
        if results_file is not None:
            status = results_file.estimation_status(self.table_number)
        for k, v in status.items():
            estimation_status[k] = v
        self._estimation_status = estimation_status
        self.minimization_successful = estimation_status['minimization_successful']
        self.function_evaluations = estimation_status['function_evaluations']
        self.significant_digits = estimation_status['significant_digits']
        if estimation_status['maxevals_exceeded'] is True:
            self.termination_cause = 'maxevals_exceeded'
        elif estimation_status['rounding_errors'] is True:
            self.termination_cause = 'rounding_errors'
        else:
            self.termination_cause = None


class NONMEMChainedModelfitResults(ChainedModelfitResults):
    def __init__(self, path, model=None, subproblem=None):
        # Path is path to any result file
        self.log = Log()
        path = Path(path)
        self._path = path
        self._subproblem = subproblem
        self.model = model
        super().__init__()
        self._read_ext_table()
        self._read_lst_file()
        self._read_cov_table()
        self._read_cor_table()
        self._read_coi_table()

    def __getattr__(self, item):
        # Avoid infinite recursion when deepcopying
        # See https://stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy
        if item.startswith('__'):
            raise AttributeError('')
        return super().__getattribute__(item)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def _read_ext_table(self):
        try:
            ext_tables = NONMEMTableFile(self._path.with_suffix('.ext'))
        except ValueError:
            # The ext-file is illegal
            self.log.log_error(f"Broken ext-file {self._path.with_suffix('.ext')}")
            result_obj = NONMEMModelfitResults(self)
            result_obj.model = self.model
            is_covariance_step = self.model.estimation_steps[0].cov
            result_obj = self._fill_empty_results(result_obj, is_covariance_step)
            result_obj.table_number = 1
            self.append(result_obj)
            return

        for table in ext_tables:
            if self._subproblem and table.subproblem != self._subproblem:
                continue
            result_obj = NONMEMModelfitResults(self)
            result_obj.model = self.model
            result_obj.table_number = table.number

            try:
                table.data_frame
            except ValueError:
                self.log.log_error(
                    f"Broken table in ext-file {self._path.with_suffix('.ext')}, "
                    f"table no. {table.number}"
                )
                is_covariance_step = self.model.estimation_steps[table.number - 1].cov
                result_obj = self._fill_empty_results(result_obj, is_covariance_step)
                self.append(result_obj)
                continue

            ests = table.final_parameter_estimates
            try:
                fix = table.fixed
            except KeyError:
                # NM 7.2 does not have row -1000000006 indicating FIXED status
                if self.model:
                    fixed = pd.Series(self.model.parameters.fix)
                    fix = pd.concat(
                        [fixed, pd.Series(True, index=ests.index.difference(fixed.index))]
                    )

            try:
                ses = table.standard_errors
                result_obj._set_covariance_status(table)
            except Exception:
                # If there are no standard errors in ext-file it means
                # there can be no cov, cor or coi either
                result_obj.standard_errors = None
                result_obj.covariance_matrix = None
                result_obj.correlation_matrix = None
                result_obj.information_matrix = None
                result_obj._set_covariance_status(None)
            else:
                ses = ses[~fix]
                sdcorr = table.omega_sigma_se_stdcorr[~fix]
                if self.model:
                    ses = ses.rename(
                        index=parameter_translation(self.model.internals.control_stream)
                    )
                    sdcorr = sdcorr.rename(
                        index=parameter_translation(self.model.internals.control_stream)
                    )
                result_obj.standard_errors = ses
                sdcorr_ses = ses.copy()
                sdcorr_ses.update(sdcorr)
                if self.model:
                    sdcorr_ses = sdcorr_ses.rename(
                        index=parameter_translation(self.model.internals.control_stream)
                    )
                result_obj.standard_errors_sdcorr = sdcorr_ses
            self.append(result_obj)

    def _fill_empty_results(self, result_obj, is_covariance_step):
        # Parameter estimates NaN for all parameters that should be estimated
        pe = pd.Series(
            np.nan, name='estimates', index=list(self.model.parameters.nonfixed.inits.keys())
        )
        result_obj.parameter_estimates = pe
        result_obj.ofv = np.nan
        if is_covariance_step:
            se = pd.Series(
                np.nan, name='SE', index=list(self.model.parameters.nonfixed.inits.keys())
            )
            result_obj.standard_errors = se
        else:
            result_obj.standard_errors = None
        return result_obj

    def _read_lst_file(self):
        try:
            rfile = NONMEMResultsFile(self._path.with_suffix('.lst'), self.log)
        except OSError:
            return
        table_with_cov = -99
        if self.model is not None:
            if len(self.model.internals.control_stream.get_records('COVARIANCE')) > 0:
                table_with_cov = self[-1].table_number  # correct unless interrupted
        for table_no, result_obj in enumerate(self, 1):
            result_obj._set_estimation_status(rfile, requested=True)
            # _covariance_status already set to None if ext table did not have standard errors
            result_obj._set_covariance_status(rfile, table_with_cov=table_with_cov)
            try:
                result_obj.estimation_runtime = rfile.table[table_no]['estimation_runtime']
            except (KeyError, FileNotFoundError):
                result_obj.estimation_runtime = np.nan

    def _read_cov_table(self):
        try:
            cov_table = NONMEMTableFile(self._path.with_suffix('.cov'))
        except OSError:
            for result_obj in self:
                if not hasattr(result_obj, 'covariance_matrix'):
                    result_obj.covariance_matrix = None
            return
        for result_obj in self:
            if _check_covariance_status(result_obj):
                df = cov_table.table_no(result_obj.table_number).data_frame
                if df is not None:
                    if self.model:
                        df = df.rename(
                            index=parameter_translation(self.model.internals.control_stream)
                        )
                        df.columns = df.index
                result_obj.covariance_matrix = df
            else:
                result_obj.covariance_matrix = None

    def _read_coi_table(self):
        try:
            coi_table = NONMEMTableFile(self._path.with_suffix('.coi'))
        except OSError:
            for result_obj in self:
                if not hasattr(result_obj, 'information_matrix'):
                    result_obj.information_matrix = None
            return
        for result_obj in self:
            if _check_covariance_status(result_obj):
                df = coi_table.table_no(result_obj.table_number).data_frame
                if df is not None:
                    if self.model:
                        df = df.rename(
                            index=parameter_translation(self.model.internals.control_stream)
                        )
                        df.columns = df.index
                result_obj.information_matrix = df
            else:
                result_obj.information_matrix = None

    def _read_cor_table(self):
        try:
            cor_table = NONMEMTableFile(self._path.with_suffix('.cor'))
        except OSError:
            for result_obj in self:
                if not hasattr(result_obj, 'correlation_matrix'):
                    result_obj.correlation_matrix = None
            return
        for result_obj in self:
            if _check_covariance_status(result_obj):
                cor = cor_table.table_no(result_obj.table_number).data_frame
                if cor is not None:
                    if self.model:
                        cor = cor.rename(
                            index=parameter_translation(self.model.internals.control_stream)
                        )
                        cor.columns = cor.index
                    np.fill_diagonal(cor.values, 1)
                result_obj.correlation_matrix = cor
            else:
                result_obj.correlation_matrix = None


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


def parse_lst(path, table_numbers):
    try:
        rfile = NONMEMResultsFile(path.with_suffix('.lst'), log=None)
    except OSError:
        return None, np.nan

    runtime_total = rfile.runtime_total

    try:
        log_likelihood = rfile.table[table_numbers[-1]]['ofv_with_constant']
    except (KeyError, FileNotFoundError):
        log_likelihood = np.nan

    return runtime_total, log_likelihood


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
    steps = list(range(len(model.estimation_steps)))
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
        return np.nan, create_failed_ofv_iterations(model), failed_pe, failed_pe, None

    table_numbers = parse_table_numbers(ext_tables, subproblem)

    final_ofv, ofv_iterations = parse_ofv(model, ext_tables, subproblem)
    final_pe, sdcorr, pe_iterations = parse_parameter_estimates(model, ext_tables, subproblem)
    return table_numbers, final_ofv, ofv_iterations, final_pe, sdcorr, pe_iterations


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


def _check_covariance_status(result):
    return (
        isinstance(result, NONMEMModelfitResults) and result._covariance_status['warnings'] is False
    )
