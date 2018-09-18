import re
from io import StringIO
from pathlib import Path
import pandas as pd



class NONMEMTableIO(StringIO):
    """ An IO class for NONMEM table files that is a prefilter for pandas.read_table.
        Things that cannot be handled directly by pandas will be taken care of here and the
        rest will be taken care of by pandas.
    """
    def __init__(self, filename):
        with open(str(filename), 'r') as tablefile:
            contents = tablefile.read()      # All variations of newlines are converted into \n

        # We might want to read out the data from the ext header here
        # and also add replicate (Table) column if applicable
        contents = re.sub(r'TABLE NO.\s+\d+.*\n', '', contents, flags=re.MULTILINE)

        super().__init__(contents)


class NONMEMTable:
    '''A NONMEM output table.
    '''
    def __init__(self, filename):
        file_io = NONMEMTableIO(filename)
        self._df = pd.read_table(file_io, sep='\s+', engine='python')

    @property
    def data_frame(self):
        return self._df


class ExtTable(NONMEMTable):
    def _get_parameters(self, iteration, include_thetas=True):
        df = self.data_frame
        row = df.loc[df['ITERATION'] == iteration]
        del row['ITERATION']
        del row['OBJ']
        if not include_thetas:
            row = row[row.columns.drop(list(row.filter(regex='THETA')))]
        return row.squeeze()

    def _get_ofv(self, iteration):
        df = self.data_frame
        row = df.loc[df['ITERATION'] == iteration]
        row.squeeze()
        return row['OBJ'].squeeze()

    @property
    def final_parameter_estimates(self):
        '''Get the final parameter estimates as a Series
           A specific parameter estimate can be retreived as
           final_parameter_estimates['THETA1']
        '''
        return self._get_parameters(-1000000000)

    @property
    def standard_errors(self):
        '''Get the standard errors of the parameter estimates as a Series
        '''
        return self._get_parameters(-1000000001)

    @property
    def omega_sigma_stdcorr(self):
        '''Get the omegas and sigmas in standard deviation/correlation form
        '''
        return self._get_parameters(-1000000004, include_thetas=False)

    @property
    def omega_sigma_se_stdcorr(self):
        '''The standard error of the omegas and sigmas in stdcorr form
        '''
        return self._get_parameters(-1000000005, include_thetas=False)

    @property
    def fixed(self):
        '''What parameters were fixed?
        '''
        fix = self._get_parameters(-1000000006)
        return fix.apply(bool)

    @property
    def final_OFV(self):
        return self._get_ofv(-1000000000)

    @property
    def initial_OFV(self):
        return self._get_ofv(0)
