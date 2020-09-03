from pathlib import Path

import numpy as np
import pandas as pd

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
        try:
            return self._ofv
        except AttributeError:
            self._chain._read_ext_table()
            return self._ofv

    @property
    def parameter_estimates(self):
        try:
            if not self._repara:
                return self._parameter_estimates
            else:
                return self._parameter_estimates_sdcorr
        except AttributeError:
            self._chain._read_ext_table()
            if not self._repara:
                return self._parameter_estimates
            else:
                return self._parameter_estimates_sdcorr

    @property
    def covariance_matrix(self):
        """The covariance matrix of the population parameter estimates
        """
        try:
            return self._covariance_matrix
        except AttributeError:
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
        """The Fischer information matrix of the population parameter estimates
        """
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
    def standard_errors(self):
        try:
            if not self._repara:
                return self._standard_errors
            else:
                return self._standard_errors_sdcorr
        except AttributeError:
            try:
                self._chain._read_ext_table()
            except OSError:
                pass
            else:
                if not self._repara:
                    return self._standard_errors
                else:
                    return self._standard_errors_sdcorr
            if self._repara:
                raise NotImplementedError("Cannot find ext-file to get standard errors on sdcorr "
                                          "form. Not supported to try getting SEs from other "
                                          "NONMEM output files")
            try:
                self._chain._read_cor_table()
            except OSError:
                pass
            else:
                return self._standard_errors
            try:
                self._chain._read_cov_table()
            except OSError:
                pass
            else:
                self._standard_errors = self._se_from_cov()
                return self._standard_errors
            try:
                self._chain._read_coi_table()
            except OSError:
                pass
            else:
                self._standard_errors = self._se_from_inf()
                return self._standard_errors
            raise FileNotFoundError("Could not find any of the ext/cov/cor/coi files")

    @property
    def individual_ofv(self):
        """A Series with individual estimates indexed over ID
        """
        try:
            return self._individual_ofv
        except AttributeError:
            self._chain._read_phi_table()
            return self._individual_ofv

    @property
    def individual_estimates(self):
        """ETA values from phi-file
        """
        try:
            return self._individual_estimates
        except AttributeError:
            self._chain._read_phi_table()
            return self._individual_estimates

    @property
    def individual_estimates_covariance(self):
        """ETCs from phi-file as Series of DataFrames
        """
        try:
            return self._individual_estimates_covariance
        except AttributeError:
            self._chain._read_phi_table()
            return self._individual_estimates_covariance


class NONMEMChainedModelfitResults(ChainedModelfitResults):
    def __init__(self, path, n, model=None):
        # Path is path to any result file. lst-file would make most sense probably
        # FIXME: n (number of $EST) could be removed as it could be inferred
        self._path = Path(path)
        self._read_phi = False
        self._read_ext = False
        self._read_cov = False
        self._read_coi = False
        self._read_cor = False
        self.model = model
        for i in range(n):
            res = NONMEMModelfitResults(self)
            res.model_name = self._path.stem
            res.model = model
            if i != n - 1:    # Only the final estimation step carries covariance information
                res._covariance_matrix = None
                res._correlation_matrix = None
                res._information_matrix = None
            self.append(res)
        extensions = ['.lst', '.ext', '.cov', '.cor', '.coi', '.phi']
        self.tool_files = [self._path.with_suffix(ext) for ext in extensions]

    def _read_ext_table(self):
        if not self._read_ext:
            ext_tables = NONMEMTableFile(self._path.with_suffix('.ext'))
            for table, result_obj in zip(ext_tables, self):
                ests = table.final_parameter_estimates
                fix = table.fixed
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
                    result_obj._standard_errors = None
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
                result_obj._ofv = table.final_ofv
            self._read_ext = True

    def _read_cov_table(self):
        if not self._read_cov:
            cov_table = NONMEMTableFile(self._path.with_suffix('.cov'))
            df = next(cov_table).data_frame
            if self.model:
                df = df.rename(index=self.model.parameter_translation())
                df.columns = df.index
            self[-1]._covariance_matrix = df
            self._read_cov = True

    def _read_coi_table(self):
        if not self._read_coi:
            coi_table = NONMEMTableFile(self._path.with_suffix('.coi'))
            df = next(coi_table).data_frame
            if self.model:
                df = df.rename(index=self.model.parameter_translation())
                df.columns = df.index
            self[-1]._information_matrix = df
            self._read_coi = True

    def _read_cor_table(self):
        if not self._read_cor:
            cor_table = NONMEMTableFile(self._path.with_suffix('.cor'))
            cor = next(cor_table).data_frame
            if self.model:
                cor = cor.rename(index=self.model.parameter_translation())
                cor.columns = cor.index
            if not hasattr(self[-1], '_standard_errors'):   # In case read from the ext-file
                self[-1]._standard_errors = pd.Series(np.diag(cor), index=cor.index)
            np.fill_diagonal(cor.values, 1)
            self[-1]._correlation_matrix = cor
            self._read_cor = True

    def _read_phi_table(self):
        if not self._read_phi:
            rv_names = [rv.name for rv in self.model.random_variables if rv.name.startswith('ETA')]
            phi_tables = NONMEMTableFile(self._path.with_suffix('.phi'))
            for table, result_obj in zip(phi_tables, self):
                result_obj._individual_ofv = table.iofv
                result_obj._individual_estimates = table.etas[rv_names]
                covs = table.etcs
                for index, cov in covs.iteritems():     # Remove ETCs for 0 FIX OMEGAs
                    covs[index] = cov[rv_names].loc[rv_names]
                result_obj._individual_estimates_covariance = covs

            self._read_phi = True
