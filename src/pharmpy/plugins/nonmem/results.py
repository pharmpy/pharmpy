from pathlib import Path

from pharmpy.plugins.nonmem.table import NONMEMTableFile
from pharmpy.results import ChainedModelfitResults, ModelfitResults


class NONMEMModelfitResults(ModelfitResults):
    # objects here need to know what chain they are in to be able to do lazy reading
    def __init__(self, chain):
        self._chain = chain

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
        for _ in range(n):
            self.append(NONMEMModelfitResults(self))

    def _read_phi_table(self):
        phi_tables = NONMEMTableFile(self._path.with_suffix('.phi'))
        for table, result_obj in zip(phi_tables, self):
            df = table.data_frame[['ID', 'OBJ']]
            df.columns = ['ID', 'iOFV']
            df.set_index('ID', inplace=True)
            result_obj._individual_OFV = df['iOFV']
            self._read_phi = True
