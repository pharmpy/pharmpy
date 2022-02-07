from pathlib import Path

import pandas as pd

from pharmpy.results import Results


class ResmodResults(Results):
    """Resmod results class"""

    def __init__(self, models=None):
        self.models = models


def calculate_results(base_model, tvar_models, other_models):
    base_ofv = base_model.modelfit_results.ofv

    model_name = []
    model_dofv = []
    model_params = []
    for model in other_models:
        name = model.name
        dofv = base_ofv - model.modelfit_results.ofv
        if name == 'IIV_on_RUV':
            param = {'omega': round(model.modelfit_results.parameter_estimates["IIV_RUV1"], 6)}
        elif name == 'power':
            param = {'theta': round(model.modelfit_results.parameter_estimates["power1"], 6)}
        else:
            param = {
                'sigma_add': round(model.modelfit_results.parameter_estimates["sigma_add"], 6),
                'sigma_prop': round(model.modelfit_results.parameter_estimates["sigma_prop"], 6),
            }
        model_name.append(name)
        model_dofv.append(dofv)
        model_params.append(param)

    df = pd.DataFrame(
        {
            'model': model_name,
            'dvid': 1,
            'iteration': 1,
            'dofv': model_dofv,
            'parameters': model_params,
        }
    )

    tvar_name = []
    dofv_tvar = []
    theta_tvar = []
    for model in tvar_models:
        name = model.name
        dofv = base_ofv - model.modelfit_results.ofv
        theta = round(model.modelfit_results.parameter_estimates["time_varying"], 6)
        tvar_name.append(name)
        dofv_tvar.append(dofv)
        theta_tvar.append(theta)

    params_tvar = []
    for i in range(1, len(tvar_models) + 1):
        param = {f"theta_tvar{i}": theta_tvar[i - 1]}
        params_tvar.append(param)

    df_tvar = pd.DataFrame(
        {
            'model': tvar_name,
            'dvid': 1,
            'iteration': 1,
            'dofv': dofv_tvar,
            'parameters': params_tvar,
        }
    )

    df_final = pd.concat([df, df_tvar])
    df_final.set_index(['model', 'dvid', 'iteration'], inplace=True)

    res = ResmodResults(models=df_final)
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
