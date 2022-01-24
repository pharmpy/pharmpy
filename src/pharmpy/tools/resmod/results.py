from pathlib import Path

import pandas as pd

from pharmpy.results import Results


class ResmodResults(Results):
    """Resmod results class"""

    def __init__(self, models=None):
        self.models = models


def calculate_results(base_model, iiv_on_ruv, power, combined, tvar_models):
    base_ofv = base_model.modelfit_results.ofv
    dofv_iiv_on_ruv = base_ofv - iiv_on_ruv.modelfit_results.ofv
    dofv_power = base_ofv - power.modelfit_results.ofv
    dofv_combined = base_ofv - combined.modelfit_results.ofv
    omega_iiv_on_ruv = round(iiv_on_ruv.modelfit_results.parameter_estimates["IIV_RUV1"], 6)
    theta_power = round(power.modelfit_results.parameter_estimates["power1"], 6)
    sigma_add = round(combined.modelfit_results.parameter_estimates["sigma_add"], 6)
    sigma_prop = round(combined.modelfit_results.parameter_estimates["sigma_prop"], 6)

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

    df = pd.DataFrame(
        {
            'model': [
                'IIV_on_RUV',
                'power',
                'combined',
            ],
            'dvid': 1,
            'iteration': 1,
            'dofv': [
                dofv_iiv_on_ruv,
                dofv_power,
                dofv_combined,
            ],
            'parameters': [
                {'omega': omega_iiv_on_ruv},
                {'theta': theta_power},
                dict(sigma_add=sigma_add, sigma_prop=sigma_prop),
            ],
        }
    )

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
