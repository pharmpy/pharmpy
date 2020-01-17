from pathlib import Path

from pharmpy.plugins.nonmem.table import NONMEMTableFile
from pharmpy.results import ChainedModelfitResults, ModelfitResults


class NONMEMModelfitResults(ModelfitResults):
    # objects here need to know what chain they are in to be able to do lazy reading
    def __init__(self, chain):
        self._chain = chain

    @property
    def covariance_matrix(self):
        """The covariance matrix of the population parameter estimates
        """
        if not self._chain._read_cov:
            self._chain._read_cov_table()
        return self._covariance_matrix

    @property
    def information_matrix(self):
        """The Fischer information matrix of the population parameter estimates
        """
        if not self._chain._read_coi:
            self._chain_read_coi_table()
        return self._information_table

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
        self._read_cov = False
        self._read_coi = False
        for _ in range(n):
            res = NONMEMModelfitResults(self)
            res.model_name = self._path.stem
            self.append(res)

    def _read_cov_table(self):
        cov_table = NONMEMTableFile(self._path.with_suffix('.cov'))
        self[-1]._covariance_matrix = next(cov_table).data_frame
        self._read_cov = True

    def _read_coi_table(self):
        coi_table = NONMEMTableFile(self._path.with_suffix('.coi'))
        self[-1]._information_matrix = next(coi_table).data_frame
        self._read_coi = True

    def _read_phi_table(self):
        phi_tables = NONMEMTableFile(self._path.with_suffix('.phi'))
        for table, result_obj in zip(phi_tables, self):
            df = table.data_frame[['ID', 'OBJ']]
            df.columns = ['ID', 'iOFV']
            df.set_index('ID', inplace=True)
            result_obj._individual_OFV = df['iOFV']
            self._read_phi = True
