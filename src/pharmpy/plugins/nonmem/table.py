import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import pharmpy.math


class NONMEMTableFile:
    """A NONMEM table file that can contain multiple tables"""

    def __init__(self, path=None, tables=None):
        if path is not None:
            path = Path(path)
            suffix = path.suffix
            self.tables = []
            with open(str(path), 'r') as tablefile:
                current = []
                for line in tablefile:
                    if line.startswith("TABLE NO."):
                        if current:
                            self._add_table(current, suffix)
                        current = [line]
                    else:
                        current.append(line)
                self._add_table(current, suffix)
            self._count = 0
        else:
            self.tables = tables

    def __iter__(self):
        return self

    def _add_table(self, content, suffix):
        table_line = content.pop(0)
        if suffix == '.ext':
            table = ExtTable(''.join(content))
        elif suffix == '.phi':
            table = PhiTable(''.join(content))
        elif suffix == '.cov' or suffix == '.cor' or suffix == '.coi':
            table = CovTable(''.join(content))
        else:
            table = NONMEMTable(''.join(content))  # Fallback to non-specific table type
        m = re.match(r'TABLE NO.\s+(\d+)', table_line)  # This is guaranteed to match
        table.number = int(m.group(1))
        table.is_evaluation = False
        if re.search(r'(Evaluation)', table_line):
            table.is_evaluation = True  # No estimation step was run
        m = re.match(
            r'TABLE NO.\s+\d+: (.*?): (?:Goal Function=(.*): )?Problem=(\d+) '
            r'Subproblem=(\d+) Superproblem1=(\d+) Iteration1=(\d+) Superproblem2=(\d+) '
            r'Iteration2=(\d+)',
            table_line,
        )
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

    def __len__(self):
        return len(self.tables)

    @property
    def table(
        self, problem=1, subproblem=0, superproblem1=0, iteration1=0, superproblem2=0, iteration2=0
    ):
        for t in self.tables:
            if (
                t.problem == problem
                and t.subproblem == subproblem
                and t.superproblem1 == superproblem1
                and t.iteration1 == iteration1
                and t.superproblem2 == superproblem2
                and t.iteration2 == iteration2
            ):
                return t

    def table_no(self, table_number=1):
        for table in self.tables:
            if table.number == table_number:
                return table

    def write(self, path):
        with open(path, 'w') as df:
            for table in self.tables:
                print('TABLE NO.     1', file=df)
                table.create_content()
                print(table.content, file=df, end='')

    def __next__(self):
        if self._count >= len(self):
            raise StopIteration
        else:
            self._count += 1
            return self.tables[self._count - 1]


class NONMEMTable:
    """A NONMEM output table."""

    def __init__(self, content=None, df=None):
        if content is not None:
            self._df = pd.read_table(StringIO(content), sep=r'\s+', engine='python')
        else:
            self._df = df

    @property
    def data_frame(self):
        return self._df

    @staticmethod
    def rename_index(df, ext=True):
        """If columns rename also the row index"""
        theta_labels = [x for x in df.columns if x.startswith('THETA')]
        omega_labels = [x for x in df.columns if x.startswith('OMEGA')]
        sigma_labels = [x for x in df.columns if x.startswith('SIGMA')]
        labels = theta_labels + omega_labels + sigma_labels
        if ext:
            labels = ['ITERATION'] + labels + ['OBJ']
        else:
            df = df.reindex(labels)
        df = df.reindex(labels, axis=1)
        df.columns = df.columns.str.replace(r'THETA(\d+)', r'THETA(\1)')
        if not ext:
            df.index = df.index.str.replace(r'THETA(\d+)', r'THETA(\1)')
        return df


class CovTable(NONMEMTable):
    @property
    def data_frame(self):
        df = self._df.copy(deep=True)
        df.set_index('NAME', inplace=True)
        df.index.name = None
        df = NONMEMTable.rename_index(df, ext=False)
        df = df.loc[(df != 0).any(axis=1), (df != 0).any(axis=0)]  # Remove FIX
        return df


class PhiTable(NONMEMTable):
    @property
    def iofv(self):
        df = self._df.copy(deep=True)
        iofv = df[['ID', 'OBJ']]
        iofv.set_index('ID', inplace=True)
        iofv.columns = ['iOFV']
        return iofv['iOFV']

    @property
    def etas(self):
        df = self._df.copy(deep=True)
        eta_col_names = [col for col in df if col.startswith('ETA')]
        etas = df[eta_col_names]
        etas.index = df['ID']
        return etas

    @property
    def etcs(self):
        df = self._df.copy(deep=True)
        eta_col_names = [col for col in df if col.startswith('ETA')]
        etc_col_names = [col for col in df if col.startswith('ETC')]
        vals = df[etc_col_names].values
        matrix_array = [pharmpy.math.flattened_to_symmetric(x) for x in vals]
        etc_frames = [
            pd.DataFrame(matrix, columns=eta_col_names, index=eta_col_names)
            for matrix in matrix_array
        ]
        etcs = pd.Series(etc_frames, index=df['ID'])
        return etcs

    def create_content(self):
        df = self._df.copy(deep=True)
        df.reset_index(inplace=True)
        df.insert(loc=0, column='SUBJECT_NO', value=np.arange(len(df)) + 1)
        fmt = '%13d%13d' + '%13.5E' * (len(df.columns) - 2)
        with StringIO() as s:
            np.savetxt(s, df.values, fmt=fmt)
            self.content = s.getvalue()
        with StringIO() as s:
            header_fmt = ' %-12s' * len(df.columns) + '\n'
            header = header_fmt % tuple(df.columns)
            self.content = header + self.content


class ExtTable(NONMEMTable):
    @property
    def data_frame(self):
        df = self._df.copy(deep=True)
        df = NONMEMTable.rename_index(df)
        return df

    def _get_parameters(self, iteration, include_thetas=True):
        df = self.data_frame
        row = df.loc[df['ITERATION'] == iteration]
        if row.empty:
            raise KeyError(f'Row {iteration} not available in ext-file')
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
        """Get the final parameter estimates as a Series
        A specific parameter estimate can be retreived as
        final_parameter_estimates['THETA1']
        """
        ser = self._get_parameters(-1000000000)
        ser.name = 'estimates'
        return ser

    @property
    def standard_errors(self):
        """Get the standard errors of the parameter estimates as a Series"""
        ser = self._get_parameters(-1000000001)
        ser.name = 'SE'
        return ser

    @property
    def condition_number(self):
        """Get the condition number of the correlation matrix"""
        ser = self._get_parameters(-1000000003)
        return ser.values[0]

    @property
    def omega_sigma_stdcorr(self):
        """Get the omegas and sigmas in standard deviation/correlation form"""
        return self._get_parameters(-1000000004, include_thetas=False)

    @property
    def omega_sigma_se_stdcorr(self):
        """The standard error of the omegas and sigmas in stdcorr form"""
        return self._get_parameters(-1000000005, include_thetas=False)

    @property
    def fixed(self):
        """What parameters were fixed?"""
        fix = self._get_parameters(-1000000006)
        return fix.apply(bool)

    @property
    def final_ofv(self):
        return self._get_ofv(-1000000000)

    @property
    def initial_ofv(self):
        return self._get_ofv(0)
