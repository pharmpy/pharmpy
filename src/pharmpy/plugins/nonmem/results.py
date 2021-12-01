from pathlib import Path

import numpy as np
import pandas as pd

from pharmpy.plugins.nonmem.results_file import NONMEMResultsFile
from pharmpy.plugins.nonmem.table import NONMEMTableFile
from pharmpy.results import ChainedModelfitResults, ModelfitResults
from pharmpy.workflows.log import Log


class NONMEMModelfitResults(ModelfitResults):
    # objects here need to know what chain they are in to be able to do lazy reading
    def __init__(self, chain):
        self._chain = chain

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

    def predictions_for_observations(self):
        """predictions only for observation data records"""
        df = self._chain._read_from_tables(['ID', 'TIME', 'MDV', 'PRED', 'CIPREDI', 'CPRED'], self)
        df.set_index(['ID', 'TIME'], inplace=True)
        df = df[df['MDV'] == 0]
        df = df.drop(columns=['MDV'])
        return df

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

    def high_correlations(self, limit=0.9):
        df = self.correlation_matrix
        if df is not None:
            high_and_below_diagonal = df.abs().ge(limit) & np.triu(np.ones(df.shape), k=1).astype(
                bool
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
        self.log = Log()
        self._path = Path(path)
        self._subproblem = subproblem
        self.model = model
        extensions = ['.lst', '.ext', '.cov', '.cor', '.coi', '.phi']
        self.tool_files = [self._path.with_suffix(ext) for ext in extensions]
        super().__init__()
        self._read_ext_table()
        self._read_lst_file()
        self._read_cov_table()
        self._read_cor_table()
        self._read_coi_table()
        self._read_phi_table()
        self._read_residuals()
        self._read_predictions()
        self._calculate_cov_cor_coi()

    def __getattr__(self, item):
        # Avoid infinite recursion when deepcopying
        # See https://stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy
        if item.startswith('__'):
            raise AttributeError('')
        return super().__getattribute__(item)

    def __getitem__(self, key):
        return super().__getitem__(key)

    def __bool__(self):
        # without this, an existing but 'unloaded' object will evaluate to False
        return len(self) > 0

    def _read_ext_table(self):
        try:
            ext_tables = NONMEMTableFile(self._path.with_suffix('.ext'))
        except ValueError:
            # The ext-file is illegal
            self.log.log_error(f"Broken ext-file {self._path.with_suffix('.ext')}")
            result_obj = NONMEMModelfitResults(self)
            result_obj.model_name = self._path.stem
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
            result_obj.model_name = self._path.stem
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

            result_obj.ofv = table.final_ofv
            result_obj.evaluation_ofv = table.initial_ofv
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
            result_obj.parameter_estimates = ests
            try:
                sdcorr = table.omega_sigma_stdcorr[~fix]
            except KeyError:
                pass
            else:
                if self.model:
                    sdcorr = sdcorr.rename(index=self.model.parameter_translation())
                sdcorr_ests = ests.copy()
                sdcorr_ests.update(sdcorr)
                result_obj.parameter_estimates_sdcorr = sdcorr_ests
            try:
                ses = table.standard_errors
            except Exception:
                # If there are no standard errors in ext-file it means
                # there can be no cov, cor or coi either
                result_obj.standard_errors = None
                result_obj.covariance_matrix = None
                result_obj.correlation_matrix = None
                result_obj.information_matrix = None
                result_obj._condition_number = None
                result_obj._set_covariance_status(None)
            else:
                ses = ses[~fix]
                sdcorr = table.omega_sigma_se_stdcorr[~fix]
                if self.model:
                    ses = ses.rename(index=self.model.parameter_translation())
                    sdcorr = sdcorr.rename(index=self.model.parameter_translation())
                result_obj.standard_errors = ses
                sdcorr_ses = ses.copy()
                sdcorr_ses.update(sdcorr)
                if self.model:
                    sdcorr_ses = sdcorr_ses.rename(index=self.model.parameter_translation())
                result_obj.standard_errors_sdcorr = sdcorr_ses
                try:
                    condition_number = table.condition_number
                except Exception:
                    pass  # PRINT=E not set in $COV, but could compute from correlation matrix
                else:
                    result_obj._condition_number = condition_number
            self.append(result_obj)

    def _fill_empty_results(self, result_obj, is_covariance_step):
        # Parameter estimates NaN for all parameters that should be estimated
        pe = pd.Series(
            np.nan, name='estimates', index=list(self.model.parameters.nonfixed_inits.keys())
        )
        result_obj.parameter_estimates = pe
        result_obj.ofv = np.nan
        if is_covariance_step:
            se = pd.Series(
                np.nan, name='SE', index=list(self.model.parameters.nonfixed_inits.keys())
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
            if len(self.model.control_stream.get_records('COVARIANCE')) > 0:
                table_with_cov = self[-1].table_number  # correct unless interrupted
        for table_no, result_obj in enumerate(self, 1):
            result_obj._set_estimation_status(rfile, requested=True)
            # _covariance_status already set to None if ext table did not have standard errors
            if hasattr(result_obj, '_covariance_status') is False:
                result_obj._set_covariance_status(rfile, table_with_cov=table_with_cov)
            result_obj.runtime_total = rfile.runtime_total

    @property
    def covariance_step(self):
        return self[-1].covariance_step

    @property
    def estimation_step(self):
        return self[-1].estimation_step

    @property
    def condition_number(self):
        return self[-1].condition_number

    @property
    def predictions(self):
        return self[-1].predictions

    @property
    def predictions_for_observations(self):
        return self[-1].predictions_for_observations

    def sumo(self, **kwargs):
        return self[-1].sumo(**kwargs)

    def _read_cov_table(self):
        try:
            cov_table = NONMEMTableFile(self._path.with_suffix('.cov'))
        except OSError:
            for result_obj in self:
                if not hasattr(result_obj, 'covariance_matrix'):
                    result_obj.covariance_matrix = None
            return
        for result_obj in self:
            df = cov_table.table_no(result_obj.table_number).data_frame
            if df is not None:
                if self.model:
                    df = df.rename(index=self.model.parameter_translation())
                    df.columns = df.index
            result_obj.covariance_matrix = df

    def _read_coi_table(self):
        try:
            coi_table = NONMEMTableFile(self._path.with_suffix('.coi'))
        except OSError:
            for result_obj in self:
                if not hasattr(result_obj, 'information_matrix'):
                    result_obj.information_matrix = None
            return
        for result_obj in self:
            df = coi_table.table_no(result_obj.table_number).data_frame
            if df is not None:
                if self.model:
                    df = df.rename(index=self.model.parameter_translation())
                    df.columns = df.index
            result_obj.information_matrix = df

    def _read_cor_table(self):
        try:
            cor_table = NONMEMTableFile(self._path.with_suffix('.cor'))
        except OSError:
            for result_obj in self:
                if not hasattr(result_obj, 'correlation_matrix'):
                    result_obj.correlation_matrix = None
            return
        for result_obj in self:
            cor = cor_table.table_no(result_obj.table_number).data_frame
            if cor is not None:
                if self.model:
                    cor = cor.rename(index=self.model.parameter_translation())
                    cor.columns = cor.index
                np.fill_diagonal(cor.values, 1)
            result_obj.correlation_matrix = cor

    def _calculate_cov_cor_coi(self):
        for obj in self:
            if obj.covariance_matrix is None:
                if obj.correlation_matrix is not None:
                    obj.covariance_matrix = self._cov_from_corrse()
                elif obj.information_matrix is not None:
                    obj.covariance_matrix = obj._cov_from_inf()
            if obj.correlation_matrix is None:
                if obj.covariance_matrix is not None:
                    obj.correlation_matrix = obj._corr_from_cov()
                elif obj.information_matrix is not None:
                    obj.correlation_matrix = obj._corr_from_coi()
            if obj.information_matrix is None:
                if obj.covariance_matrix is not None:
                    obj.information_matrix = obj._inf_from_cov()
                elif obj.correlation_matrix is not None:
                    obj.information_matrix = obj._inf_from_corrse()
            if obj.standard_errors is None:
                if obj.covariance_matrix is not None:
                    obj.standard_errors = obj._se_from_cov()
                elif obj.information_matrix is not None:
                    obj.standard_errors = obj._se_from_inf()

    def _read_phi_table(self):
        for result_obj in self:
            result_obj.individual_ofv = None
            result_obj.individual_estimates = None
            result_obj.individual_estimates_covariance = None

        trans = self.model.rv_translation(reverse=True)
        rv_names = [name for name in self.model.random_variables.etas.names if name in trans]
        try:
            phi_tables = NONMEMTableFile(self._path.with_suffix('.phi'))
        except FileNotFoundError:
            return
        for result_obj in self:
            table = phi_tables.table_no(result_obj.table_number)
            if table is not None:
                try:
                    result_obj.individual_ofv = table.iofv
                    result_obj.individual_estimates = table.etas.rename(
                        columns=self.model.rv_translation()
                    )[rv_names]
                    covs = table.etcs
                    covs = covs.transform(
                        lambda cov: cov.rename(
                            columns=self.model.rv_translation(), index=self.model.rv_translation()
                        )
                    )
                    covs = covs.transform(lambda cov: cov[rv_names].loc[rv_names])
                    result_obj.individual_estimates_covariance = covs
                except KeyError:
                    result_obj.individual_ofv = None
                    result_obj.inividual_estimates = None
                    result_obj.individual_estimates_covariance = None

    def _read_residuals(self):
        for obj in self:
            try:
                df = self._read_from_tables(['ID', 'TIME', 'RES', 'WRES', 'CWRES'], obj)
                df['ID'] = df['ID'].convert_dtypes()
                df.set_index(['ID', 'TIME'], inplace=True)
            except (KeyError, OSError):
                obj.residuals = None
            else:
                df = df.loc[(df != 0).any(axis=1)]  # Simple way of removing non-observations
                obj.residuals = df

    def _read_predictions(self):
        for obj in self:
            try:

                df = self._read_from_tables(
                    ['ID', 'TIME', 'PRED', 'CIPREDI', 'CPRED', 'IPRED'], obj
                )
                df['ID'] = df['ID'].convert_dtypes()
                df.set_index(['ID', 'TIME'], inplace=True)
            except (KeyError, OSError):
                obj.predictions = None
            else:
                obj.predictions = df

    def _read_from_tables(self, columns, result_obj):
        table_recs = self.model.control_stream.get_records('TABLE')
        found = []
        df = pd.DataFrame()
        for table_rec in table_recs:
            columns_in_table = []
            for key, value in table_rec.all_options:
                if key in columns and key not in found and value is None:
                    # FIXME: Cannot handle synonyms here
                    colname = key
                elif value in columns and value not in found:
                    colname = value
                else:
                    continue
                found.append(colname)
                columns_in_table.append(colname)
            if columns_in_table:
                noheader = table_rec.has_option("NOHEADER")
                notitle = table_rec.has_option("NOTITLE") or noheader
                nolabel = table_rec.has_option("NOLABEL") or noheader
                path = table_rec.path
                if not path.is_absolute():
                    model_path = self.model.database.retrieve_file(self.model.name, path)
                    path = model_path.parent / path  # Relative model source file.
                table_file = NONMEMTableFile(path, notitle=notitle, nolabel=nolabel)
                table = table_file.tables[0]
                df[columns_in_table] = table.data_frame[columns_in_table]
        return df


def simfit_results(model):
    """Read in modelfit results from a simulation/estimation model"""
    nsubs = model.control_stream.get_records('SIMULATION')[0].nsubs
    results = []
    for i in range(1, nsubs + 1):
        model_path = model.database.retrieve_file(model.name, model.name + model.filename_extension)
        res = NONMEMChainedModelfitResults(model_path, model=model, subproblem=i)
        results.append(res)
    return results
