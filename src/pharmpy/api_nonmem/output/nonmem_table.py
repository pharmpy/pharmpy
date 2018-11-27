import re
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

import pharmpy.math



class NONMEMTableFile:
    '''A NONMEM table file that can contain multiple tables
    '''
    def __init__(self, filename):
        path = Path(filename)
        suffix = path.suffix
        self.tables = []
        with open(str(filename), 'r') as tablefile:
            current = []
            for line in tablefile:
                if line.startswith("TABLE NO."):
                    if current:
                        self._add_table(current, suffix)
                    current = [line]
                else:
                    current.append(line)
            self._add_table(current, suffix)

    def _add_table(self, content, suffix):
        table_line = content.pop(0)
        if suffix == '.ext':
            table = ExtTable(''.join(content))
        elif suffix == '.phi':
            table = PhiTable(''.join(content))
        else:
            table = NONMEMTable(''.join(content))       # Fallback to non-specific type of NONMEM table
        m = re.match(r'TABLE NO.\s+(\d+)', table_line)   # This is guaranteed to match
        table.number = m.group(1)
        m = re.match(r'TABLE NO.\s+\d+: (.*?): (?:Goal Function=(.*): )?Problem=(\d+) Subproblem=(\d+) Superproblem1=(\d+) Iteration1=(\d+) Superproblem2=(\d+) Iteration2=(\d+)', table_line)
        if m:
            table.method = m.group(1)
            table.goal_function = m.group(2)
            table.problem = int(m.group(3))
            table.subproblem = int(m.group(4))
            table.superproblem1 = int(m.group(5))
            table.iteration1 = int(m.group(6))
            table.superproblem2 = int(m.group(7))
            table.iteration2 = int(m.group(8))
        self.tables.append(table)

    def number_of_tables(self):
        return len(tables)

    @property
    def table(self, problem=1, subproblem=0, superproblem1=0, iteration1=0, superproblem2=0, iteration2=0):
        for t in self.tables:
            if t.problem == problem and t.subproblem == subproblem and t.superproblem1 == superproblem1 \
                    and t.iteration1 == iteration1 and t.superproblem2 == superproblem2 and t.iteration2 == iteration2:
                return t 

    def table_no(self, table_number=1):
        for table in self.tables:
            if table.number == table_number:
                return table


class NONMEMTable:
    '''A NONMEM output table.
    '''
    def __init__(self, content):
        self._df = pd.read_table(StringIO(content), sep='\s+', engine='python')

    @property
    def data_frame(self):
        return self._df


class PhiTable(NONMEMTable):
    @property
    def data_frame(self):
        df = self._df.copy(deep=True)
        df.drop(columns=['SUBJECT_NO'], inplace=True)
        eta_col_names = [col for col in df if col.startswith('ETA')]
        df['ETA'] = df[eta_col_names].values.tolist()       # ETA column to be list of eta values for each ID
        df.drop(columns=eta_col_names, inplace=True)
        etc_col_names = [col for col in df if col.startswith('ETC')]
        vals = df[etc_col_names].values
        matrix_array = [pharmpy.math.flattened_to_symmetric(x) for x in vals]
        df['ETC'] = matrix_array                            # ETC column to be symmetric matrices for each ID
        df.drop(columns=etc_col_names, inplace=True)
        return df


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
