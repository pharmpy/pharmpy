from pathlib import Path

from pharmpy.plugins.nonmem.table import NONMEMTableFile
from pharmpy.results import ChainedModelfitResults, ModelfitResults


class NONMEMModelfitResults(ModelfitResults):
    # objects here need to know what chain they are in to be able to do lazy reading
    def __init__(self, chain):
        self._chain = chain

    @property
    def ofv(self):
        if not self._chain._read_ext:
            self._chain._read_ext_table()
        return self._ofv

    @property
    def parameter_estimates(self):
        if not self._chain._read_ext:
            self._chain._read_ext_table()
        return self._parameter_estimates

    @property
    def covariance_matrix(self):
        """The covariance matrix of the population parameter estimates
        """
        # FIXME: Should only return if last in chain. Raise otherwise? Would standard errors work? If not $COV will ext be different?
        if self._covariance_matrix is None:     # FIXME: Move the if is None to the read function.
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
                return self.cov_from_corrse()
            try:
                self._chain._read_coi_table()
            except OSError:
                pass
            else:
                return self.cov_from_inf()
            raise FileNotFoundError("Could not find any of the cov/cor and coi files")
 

        return self._covariance_matrix

        if not self._chain._read_cov:
            self._chain._read_cov_table()
        return self._covariance_matrix
        # FIXME: New idea here: check if _covariance_matrix is not None. Read from cov or coi or cor/ext

    @property
    def information_matrix(self):
        """The Fischer information matrix of the population parameter estimates
        """
        if not self._chain._read_coi:
            try:
                self._chain._read_coi_table()
            except FileNotFoundError:
                return super().information_matrix       # Fallback to inverting the cov matrix
        return self._information_matrix

    @property
    def correlation_matrix(self):
    	# FIXME: HERE: could read cov and set diags to 1
	# Also have standard errors as method
        if not self._chain._read_cor:
            self._chain._read_cor_table()
        return self._cor_matrix

    @property
    def standard_errors(self):
        # Also for FIX here? should be 0 for these

    @property
    def individual_OFV(self):
        """A Series with individual estimates indexed over ID
        """
        if not self._chain._read_phi:
            self._chain._read_phi_table()
        return self._individual_OFV


class NONMEMChainedModelfitResults(ChainedModelfitResults):
    def __init__(self, path, n):
        # Path is path to any result file. lst-file would make most sense probably
        # FIXME: n (number of $EST) could be removed as it could be inferred
        self._path = Path(path)
        self._read_phi = False
        self._read_ext = False
        for _ in range(n):
            res = NONMEMModelfitResults(self)
            res.model_name = self._path.stem
            self.append(res)

    def _read_ext_table(self):
        ext_tables = NONMEMTableFile(self._path.with_suffix('.ext'))
        for table, result_obj in zip(ext_tables, self):
            ests = table.final_parameter_estimates
            fix = table.fixed
            ests = ests[~fix]
            result_obj._parameter_estimates = ests
            result_obj._ofv = table.final_ofv
        self._read_ext = True

    def _read_cov_table(self):
        cov_table = NONMEMTableFile(self._path.with_suffix('.cov'))
        self[-1]._covariance_matrix = next(cov_table).data_frame

    def _read_coi_table(self):
        coi_table = NONMEMTableFile(self._path.with_suffix('.coi'))
        self[-1]._information_matrix = next(coi_table).data_frame
    
    def _read_cor_table(self):
        cor_table = NONMEMTableFile(self._path.with_suffix('.cor'))
        cor = next(cor_table).data_frame
        self[-1]._sderr = pd.Series(np.diag(cor), index=cor.index)
        np.fill_diagonal(cor.values, 1)
        self[-1]._cor_matrix = cor

    def _read_phi_table(self):
        phi_tables = NONMEMTableFile(self._path.with_suffix('.phi'))
        for table, result_obj in zip(phi_tables, self):
            df = table.data_frame[['ID', 'OBJ']]
            df.columns = ['ID', 'iOFV']
            df.set_index('ID', inplace=True)
            result_obj._individual_OFV = df['iOFV']
            self._read_phi = True
