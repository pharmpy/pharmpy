from pathlib import Path

import numpy as np
import pandas as pd

from pharmpy.results import Results


class ResmodResults(Results):
    """Resmod results class"""

    def __init__(self, models=None):
        self.models = models


def calculate_results(base_model, iiv_on_ruv, power):
    base_ofv = base_model.modelfit_results.ofv
    dofv_iiv_on_ruv = iiv_on_ruv.modelfit_results.ofv - base_ofv
    dofv_power = power.modelfit_results.ofv - base_ofv
    df = pd.DataFrame(
        {
            'model': ['IIV_on_RUV', 'power'],
            'dvid': 1,
            'iteration': 1,
            'dofv': [dofv_iiv_on_ruv, dofv_power],
            'parameters': np.nan,
        }
    )
    df.set_index(['model', 'dvid', 'iteration'], inplace=True)
    res = ResmodResults(models=df)
    return res


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
