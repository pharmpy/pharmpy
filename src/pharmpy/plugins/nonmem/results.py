from pathlib import Path

import numpy as np
import pandas as pd

from pharmpy.plugins.nonmem.results_file import NONMEMResultsFile
from pharmpy.plugins.nonmem.table import NONMEMTableFile
from pharmpy.results import ChainedModelfitResults, ModelfitResults


class NONMEMModelfitResults(ModelfitResults):
    # objects here need to know what chain they are in to be able to do lazy reading
    def __init__(self, chain):
        self._repara = False
        self._chain = chain

    def reparameterize(self, parameterizations):
        names = {p.name for p in parameterizations}
        if len(names) == 2 and names == {'sdcorr', 'sd'}:
            self._repara = True
        else:
            raise NotImplementedError("Only support reparametrization to sdcorr and sd")

    @property
    def ofv(self):
        return self._ofv  # no try here, if object exists then ext has been read

    @property
    def evaluation_ofv(self):
        return self._evaluation_ofv  # no try here, if object exists then ext has been read

    @property
    def parameter_estimates(self):
        if not self._repara:
            return self._parameter_estimates
        else:
            return self._parameter_estimates_sdcorr

    @property
    def standard_errors(self):
        if not self._repara:
            return self._standard_errors
        else:
            return self._standard_errors_sdcorr

    @property
    def minimization_successful(self):
        return self.estimation_step['minimization_successful']

    @property
    def estimation_step(self):
        try:
            return self._estimation_status
        except AttributeError:
            try:
                self._chain._read_lst_file()
            except FileNotFoundError:
                self._set_estimation_status(None)
            return self._estimation_status

    @property
    def covariance_step(self):
        try:
            return self._covariance_status
        except AttributeError:
            try:
                self._chain._read_lst_file()
            except FileNotFoundError:
                self._set_covariance_status(None)
            return self._covariance_status

    @property
    def condition_number(self):
        try:
            return self._condition_number
        except AttributeError:
            try:
                self._condition_number = np.linalg.cond(self.correlation_matrix)
            except Exception:
                self._condition_number = None
            return self._condition_number

    @property
    def covariance_matrix(self):
        """The covariance matrix of the population parameter estimates"""
        try:
            return self._covariance_matrix
        except AttributeError:
            # if object exists then ext has been read, and _covariance_matrix
            # has already been set to None for all table numbers without standard errors
            try:
                self._chain._read_cov_table()
            except OSError:
                pass
            else:
                return self._covariance_matrix
            try:
                self._chain._read_cor_table()
            except OSError:
                pass
            else:
                self._covariance_matrix = self._cov_from_corrse()
                return self._covariance_matrix
            try:
                self._chain._read_coi_table()
            except OSError:
                pass
            else:
                self._covariance_matrix = self._cov_from_inf()
                return self._covariance_matrix
            raise FileNotFoundError("Could not find any of the cov/cor/coi files")

    @property
    def information_matrix(self):
        """The Fischer information matrix of the population parameter estimates"""
        try:
            return self._information_matrix
        except AttributeError:
            try:
                self._chain._read_coi_table()
            except OSError:
                pass
            else:
                return self._information_matrix
            try:
                self._chain._read_cor_table()
            except OSError:
                pass
            else:
                self._information_matrix = self._inf_from_cov()
                return self._information_matrix
            try:
                self._chain._read_cor_table()
            except OSError:
                pass
            else:
                self._information_matrix = self._inf_from_corrse()
            raise FileNotFoundError("Could not find any of the cov/cor/coi files")

    @property
    def correlation_matrix(self):
        try:
            return self._correlation_matrix
        except AttributeError:
            try:
                self._chain._read_cor_table()
            except OSError:
                pass
            else:
                return self._correlation_matrix
            try:
                self._chain._read_cov_table()
            except OSError:
                pass
            else:
                self._correlation_matrix = self._corr_from_cov()
                return self._correlation_matrix
            try:
                self._chain._read_coi_table()
            except OSError:
                pass
            else:
                self._correlation_matrix = self._corr_from_coi()
                return self._correlation_matrix
            raise FileNotFoundError("Could not find any of the cov/cor/coi files")

    @property
    def individual_ofv(self):
        """A Series with individual estimates indexed over ID"""
        try:
            return self._individual_ofv
        except AttributeError:
            self._chain._read_phi_table()
            return self._individual_ofv

    @property
    def individual_estimates(self):
        """ETA values from phi-file"""
        try:
            return self._individual_estimates
        except AttributeError:
            self._chain._read_phi_table()
            return self._individual_estimates

    @property
    def individual_estimates_covariance(self):
        """ETCs from phi-file as Series of DataFrames"""
        try:
            return self._individual_estimates_covariance
        except AttributeError:
            self._chain._read_phi_table()
            return self._individual_estimates_covariance

    @property
    def residuals(self):
        df = self._chain._read_from_tables(['ID', 'TIME', 'RES', 'WRES', 'CWRES'], self)
        df.set_index(['ID', 'TIME'], inplace=True)
        df = df.loc[(df != 0).any(axis=1)]  # Simple way of removing non-observations
        return df

    def _set_covariance_status(self, results_file, table_with_cov=None):
        covariance_status = {
            'requested': True
            if self._standard_errors is not None
            else (table_with_cov == self.table_number),
            'completed': (self._standard_errors is not None),
            'warnings': None,
        }
        if self._standard_errors is not None and results_file is not None:
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

    def high_correlations(self, limit=0.9):
        df = self.correlation_matrix
        if df is not None:
            high_and_below_diagonal = df.abs().ge(limit) & np.triu(np.ones(df.shape), k=1).astype(
                np.bool
            )
            return df.where(high_and_below_diagonal).stack()

    def covariance_step_summary(self, condition_number_limit=1000, correlation_limit=0.9):
        result = dict()
        if self.covariance_step['requested'] is False:
            result['Covariance step not run'] = ''
        else:
            if self.covariance_step['completed'] is False:
                result['Covariance step not completed'] = 'ERROR'
            else:
                if self.covariance_step['warnings'] is None:
                    result['Covariance step completed'] = ''
                elif self.covariance_step['warnings'] is True:
                    result['Covariance step completed with warnings'] = 'WARNING'
                else:
                    result['Covariance step successful'] = 'OK'
                try:
                    if self.condition_number >= condition_number_limit:
                        result['Large condition number'] = 'WARNING'
                    else:
                        result[f'Condition number < {condition_number_limit}'] = 'OK'
                    result[str.format('Condition number: {:.1f}', self.condition_number)] = ''
                except Exception:
                    result['Condition number not available'] = ''
                try:
                    high = self.high_correlations(correlation_limit)
                    if high.empty:
                        result[f'No correlations larger than {correlation_limit}'] = 'OK'
                    else:
                        result['Large correlations found'] = 'WARNING'
                        for line in high.to_string().split('\n'):
                            result[line] = ''
                except FileNotFoundError:
                    result['Correlation matrix not available'] = ''

        return pd.DataFrame.from_dict(result, orient='index', columns=[''])

    def estimation_step_summary(self):
        result = dict()
        step = self.estimation_step
        if step['requested'] is False:
            result['Estimation step not run'] = ''
        else:
            if step['minimization_successful'] is None:
                result['Termination status not available'] = ''
            else:
                if step['minimization_successful'] is True:
                    result['Minimization successful'] = 'OK'
                else:
                    result['Termination problems'] = 'ERROR'
                if step['rounding_errors'] is False:
                    result['No rounding errors'] = 'OK'
                else:
                    result['Rounding errors'] = 'ERROR'
                if step['maxevals_exceeded'] is True:
                    result['Max number of evaluations exceeded'] = 'ERROR'
                if np.isnan(step['function_evaluations']) is False:
                    result[
                        str.format(
                            'Number of function evaluations: {:d}', step['function_evaluations']
                        )
                    ] = ''
                if step['estimate_near_boundary'] is True:
                    # message issued from NONMEM independent of ModelfitResults near_bounds method
                    result['Parameter near boundary'] = 'WARNING'
                if step['warning'] is True:
                    result['NONMEM estimation warnings'] = 'WARNING'
            result[str.format('Objective function value: {:.1f}', self.ofv)] = ''
            if np.isnan(step['significant_digits']) is False:
                result[str.format('Significant digits: {:.1f}', result['significant_digits'])] = ''
        return pd.DataFrame.from_dict(result, orient='index', columns=[''])

    def sumo(
        self,
        condition_number_limit=1000,
        correlation_limit=0.9,
        zero_limit=0.001,
        significant_digits=2,
        to_string=True,
    ):
        messages = pd.concat(
            [
                self.estimation_step_summary(),
                pd.DataFrame.from_dict({'': ''}, orient='index', columns=['']),
                self.covariance_step_summary(condition_number_limit, correlation_limit),
            ]
        )
        summary = self.parameter_summary()
        near_bounds = self.near_bounds(zero_limit, significant_digits)
        if to_string:
            summary[''] = near_bounds.transform(lambda x: 'Near boundary' if x else '')
            return str(messages) + '\n\n' + str(summary)
        else:
            summary['Near boundary'] = near_bounds
            return {'Messages': messages, 'Parameter summary': summary}


class NONMEMChainedModelfitResults(ChainedModelfitResults):
    def __init__(self, path, model=None, subproblem=None):
        # Path is path to any result file
        self._path = Path(path)
        self._read_phi = False
        self._read_ext = False
        self._read_cov = False
        self._read_coi = False
        self._read_cor = False
        self._read_lst = False
        self._subproblem = subproblem
        self.model = model
        extensions = ['.lst', '.ext', '.cov', '.cor', '.coi', '.phi']
        self.tool_files = [self._path.with_suffix(ext) for ext in extensions]

    def __getattr__(self, item):
        # Avoid infinite recursion when deepcopying
        # See https://stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy
        if item.startswith('__'):
            raise AttributeError('')
        self._load()
        return super().__getattribute__(item)

    def __getitem__(self, key):
        self._load()
        return super().__getitem__(key)

    def _load(self):
        self._read_ext_table()

    def __bool__(self):
        # without this, an existing but 'unloaded' object will evaluate to False
        self._load()
        return len(self) > 0

    def _read_ext_table(self):
        if not self._read_ext:
            ext_tables = NONMEMTableFile(self._path.with_suffix('.ext'))
            for table in ext_tables:
                if self._subproblem and table.subproblem != self._subproblem:
                    continue
                result_obj = NONMEMModelfitResults(self)
                result_obj.model_name = self._path.stem
                result_obj.model = self.model
                result_obj.table_number = table.number
                result_obj._ofv = table.final_ofv
                result_obj._evaluation_ofv = table.initial_ofv
                if table.is_evaluation:
                    result_obj._set_estimation_status(results_file=None, requested=False)
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
                ests = ests[~fix]
                if self.model:
                    ests = ests.rename(index=self.model.parameter_translation())
                result_obj._parameter_estimates = ests
                sdcorr = table.omega_sigma_stdcorr[~fix]
                sdcorr_ests = ests.copy()
                sdcorr_ests.update(sdcorr)
                result_obj._parameter_estimates_sdcorr = sdcorr_ests
                try:
                    ses = table.standard_errors
                except Exception:
                    # If there are no standard errors in ext-file it means
                    # there can be no cov, cor or coi either
                    result_obj._standard_errors = None
                    result_obj._covariance_matrix = None
                    result_obj._correlation_matrix = None
                    result_obj._information_matrix = None
                    result_obj._condition_number = None
                    result_obj._set_covariance_status(None)
                else:
                    ses = ses[~fix]
                    if self.model:
                        ses = ses.rename(index=self.model.parameter_translation())
                    result_obj._standard_errors = ses
                    sdcorr = table.omega_sigma_se_stdcorr[~fix]
                    sdcorr_ses = ses.copy()
                    sdcorr_ses.update(sdcorr)
                    if self.model:
                        sdcorr_ses = sdcorr_ses.rename(index=self.model.parameter_translation())
                    result_obj._standard_errors_sdcorr = sdcorr_ses
                    try:
                        condition_number = table.condition_number
                    except Exception:
                        pass  # PRINT=E not set in $COV, but could compute from correlation matrix
                    else:
                        result_obj._condition_number = condition_number
                self.append(result_obj)
            self._read_ext = True

    def _read_lst_file(self):
        if not self._read_lst:
            self._load()
            rfile = NONMEMResultsFile(self._path.with_suffix('.lst'))
            table_with_cov = -99
            if self.model is not None:
                if len(self.model.control_stream.get_records('COVARIANCE')) > 0:
                    table_with_cov = self[-1].table_number  # correct unless interrupted
            for result_obj in self:
                # _estimation_status is already set to None if ext table has (Evaluation)
                if hasattr(result_obj, '_estimation_status') is False:
                    result_obj._set_estimation_status(rfile, requested=True)
                # _covariance_status already set to None if ext table did not have standard errors
                if hasattr(result_obj, '_covariance_status') is False:
                    result_obj._set_covariance_status(rfile, table_with_cov=table_with_cov)
        self._read_lst = True

    @property
    def covariance_step(self):
        return self[-1].covariance_step

    @property
    def estimation_step(self):
        return self[-1].estimation_step

    @property
    def condition_number(self):
        return self[-1].condition_number

    def sumo(self, **kwargs):
        return self[-1].sumo(**kwargs)

    def _read_cov_table(self):
        if not self._read_cov:
            self._load()
            cov_table = NONMEMTableFile(self._path.with_suffix('.cov'))
            for result_obj in self:
                df = cov_table.table_no(result_obj.table_number).data_frame
                if df is not None:
                    if self.model:
                        df = df.rename(index=self.model.parameter_translation())
                        df.columns = df.index
                result_obj._covariance_matrix = df
            self._read_cov = True

    def _read_coi_table(self):
        if not self._read_coi:
            self._load()
            coi_table = NONMEMTableFile(self._path.with_suffix('.coi'))
            for result_obj in self:
                df = coi_table.table_no(result_obj.table_number).data_frame
                if df is not None:
                    if self.model:
                        df = df.rename(index=self.model.parameter_translation())
                        df.columns = df.index
                result_obj._information_matrix = df
            self._read_coi = True

    def _read_cor_table(self):
        if not self._read_cor:
            self._load()
            cor_table = NONMEMTableFile(self._path.with_suffix('.cor'))
            for result_obj in self:
                cor = cor_table.table_no(result_obj.table_number).data_frame
                if cor is not None:
                    if self.model:
                        cor = cor.rename(index=self.model.parameter_translation())
                        cor.columns = cor.index
                    np.fill_diagonal(cor.values, 1)
                result_obj._correlation_matrix = cor
            self._read_cor = True

    def _read_phi_table(self):
        if not self._read_phi:
            self._load()
            for result_obj in self:
                result_obj._individual_ofv = None
                result_obj._individual_estimates = None
                result_obj._individual_estimates_covariance = None

            rv_names = [rv.name for rv in self.model.random_variables if rv.name.startswith('ETA')]
            try:
                phi_tables = NONMEMTableFile(self._path.with_suffix('.phi'))
            except FileNotFoundError:
                return
            for result_obj in self:
                table = phi_tables.table_no(result_obj.table_number)
                if table is not None:
                    result_obj._individual_ofv = table.iofv
                    result_obj._individual_estimates = table.etas[rv_names]
                    covs = table.etcs
                    covs = covs.transform(lambda cov: cov[rv_names].loc[rv_names])
                    result_obj._individual_estimates_covariance = covs
            self._read_phi = True

    def _read_from_tables(self, columns, result_obj):
        self._load()
        table_recs = self.model.control_stream.get_records('TABLE')
        found = []
        df = pd.DataFrame()
        for table_rec in table_recs:
            columns_in_table = []
            for key, value in table_rec.all_options:
                if key in columns and key not in found:
                    colname = key
                elif value in columns and value not in found:
                    colname = value
                else:
                    continue
                found.append(colname)
                columns_in_table.append(colname)
            if columns_in_table:
                table_file = NONMEMTableFile(self.model.source.path.parent / table_rec.path)
                table = table_file.table_no(result_obj.table_number)
                df[columns_in_table] = table.data_frame[columns_in_table]
        return df


def simfit_results(model):
    """Read in modelfit results from a simulation/estimation model"""
    nsubs = model.control_stream.get_records('SIMULATION')[0].nsubs
    results = []
    for i in range(1, nsubs + 1):
        res = NONMEMChainedModelfitResults(model.source.path, model=model, subproblem=i)
        results.append(res)
    return results
