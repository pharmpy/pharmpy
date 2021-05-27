from pathlib import Path

import pandas as pd

from pharmpy.results import Results


class ResmodResults(Results):
    """Resmod results class"""

    def __init__(self, models=None):
        self.models = models


def psn_resmod_results(path):
    path = Path(path)
    res = ResmodResults()
    respath = path / 'resmod_results.csv'
    if respath.is_file():
        df = pd.read_csv(respath, names=range(40), skiprows=[0], engine='python')
    df[0].fillna(1, inplace=True)
    df[1].fillna(1, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df2 = df[[0, 1, 2, 3]].copy()
    df2 = df2.astype({0: int})
    df2.columns = ['iteration', 'DVID', 'model', 'dOFV']
    df2.set_index(['iteration', 'DVID', 'model'], inplace=True)
    parameters = pd.Series(name='parameters', index=df.index, dtype=object)
    for rowind, row in df.iterrows():
        d = dict()
        for i in range(4, len(row)):
            if row[i] is not None:
                a = row[i].split('=')
                d[a[0]] = float(a[1])
        parameters[rowind] = d
    parameters.index = df2.index
    df2['parameters'] = parameters
    res.models = df2
    return res
