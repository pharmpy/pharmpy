from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.tools.common import ToolResults


@dataclass(frozen=True)
class RUVSearchResults(ToolResults):
    """RUVSearch results class"""

    cwres_models: Optional[Any] = None


def calculate_results(models):
    names = [model.name for model in models]
    iterations = [int(name.split('_')[-1]) for name in names]
    iter_dfs = []
    for iteration in range(min(iterations), max(iterations) + 1):
        base_index = names.index(f'base_{iteration}')
        base_model = models[base_index]
        if base_model.modelfit_results is not None and base_model.modelfit_results.ofv is not None:
            base_ofv = base_model.modelfit_results.ofv
        else:
            warnings.warn(f'base CWRES model of iteration {iteration} failed.')
            res = RUVSearchResults()
            return res

        iteration_models = [
            model
            for model in models
            if model.name.endswith(f'_{iteration}')
            and not model.name.startswith('best_ruvsearch')
            and not model.name.startswith('base')
        ]

        model_name = []
        model_dofv = []
        model_params = []
        for model in iteration_models:
            name = model.name
            if model.modelfit_results is not None and model.modelfit_results.ofv is not None:
                dofv = base_ofv - model.modelfit_results.ofv
                if name.startswith('IIV_on_RUV'):
                    param = {
                        'omega': round(model.modelfit_results.parameter_estimates["IIV_RUV1"], 6)
                    }
                elif name.startswith('power'):
                    param = {
                        'theta': round(model.modelfit_results.parameter_estimates["power1"], 6)
                    }
                elif name.startswith('time_varying'):
                    param = {
                        'theta': round(
                            model.modelfit_results.parameter_estimates["time_varying"], 6
                        )
                    }
                else:
                    param = {
                        'sigma_add': round(
                            model.modelfit_results.parameter_estimates["sigma_add"], 6
                        ),
                        'sigma_prop': round(
                            model.modelfit_results.parameter_estimates["sigma_prop"], 6
                        ),
                    }
                a = name.split('_')
                name = '_'.join(a[0:-1])

                model_name.append(name)
                model_dofv.append(dofv)
                model_params.append(param)
            else:
                warnings.warn(
                    f"{name} model has no ofv and will be skipped in {iteration} iteration."
                )
        df = pd.DataFrame(
            {
                'model': model_name,
                'dvid': 1,
                'iteration': iteration,
                'dofv': model_dofv,
                'parameters': model_params,
            }
        )
        if not df.empty:
            iter_dfs.append(df)

    df_final = pd.concat(iter_dfs)
    df_final.set_index(['model', 'dvid', 'iteration'], inplace=True)
    df_final.sort_index(inplace=True)

    return RUVSearchResults(cwres_models=df_final)


def psn_resmod_results(path):
    path = Path(path)
    respath = path / 'resmod_results.csv'
    if not respath.is_file():
        return RUVSearchResults()
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
        d = {}
        for i in range(4, len(row)):
            if row[i] is not None and row[i] is not np.nan:
                print(row[i])
                a = row[i].split('=')
                d[a[0]] = float(a[1])
        parameters[rowind] = d
    parameters.index = df2.index
    df2['parameters'] = parameters
    return RUVSearchResults(cwres_models=df2)
