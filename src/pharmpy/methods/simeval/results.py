import pandas as pd

from pharmpy import Model
from pharmpy.methods.simfit.results import psn_simfit_results
from pharmpy.results import Results


class SimevalResults(Results):
    """Simeval results class"""

    def __init__(self, iofv=None, iofv_residuals=None, residual_outliers=None, data_mask=None):
        self.iofv = iofv
        self.iofv_residuals = iofv_residuals
        self.residual_outliers = residual_outliers
        self.data_mask = data_mask


def calculate_results(original_model, simfit_results):
    """Calculate simeval results"""
    origiofv = original_model.modelfit_results.individual_ofv
    iofv = pd.DataFrame({'original': origiofv})
    for i, res in enumerate(simfit_results.modelfit_results):
        iofv[f'sample_{i + 1}'] = res.individual_ofv
    df = iofv.T
    first = df.iloc[[0]].values[0]
    df = df.apply(lambda row: (first - row) / row.std(), axis=1)
    df.drop('original', inplace=True)
    df.index = range(1, len(df) + 1)
    df.index.name = 'sample'
    iofv_residuals = df
    iofv_medians = iofv_residuals.median(axis=0)
    outser = abs(iofv_medians) >= 3
    df = original_model.dataset[['ID']]  # FIXME!
    df = outser[df['ID']].astype(int)
    df.reset_index(drop=True, inplace=True)
    data_mask = df
    residual_outliers = list(iofv_medians[outser].index)
    res = SimevalResults(
        iofv=iofv,
        iofv_residuals=iofv_residuals,
        residual_outliers=residual_outliers,
        data_mask=data_mask,
    )
    return res


def psn_simeval_results(path):
    simfit_paths = (path / 'm1').glob('sim-*.mod')
    simfit_results = psn_simfit_results(simfit_paths)
    original = Model(path / 'm1' / 'original.mod')
    res = calculate_results(original, simfit_results)
    return res
