import pandas as pd

from pharmpy import Model
from pharmpy.results import Results
from pharmpy.tools.simfit.results import psn_simfit_results


class SimevalResults(Results):
    """Simeval results class"""

    def __init__(self, sampled_iofv=None, iofv_summary=None, individual_predictions_plot=None):
        self.sampled_iofv = sampled_iofv
        self.iofv_summary = iofv_summary
        self.individual_predictions_plot = individual_predictions_plot


def calculate_results(original_model, simfit_results):
    """Calculate simeval results"""
    sampled_iofv = pd.DataFrame()
    for i, res in enumerate(simfit_results.modelfit_results):
        sampled_iofv[f'{i + 1}'] = res.individual_ofv
    origiofv = original_model.modelfit_results.individual_ofv
    iofv_summary = pd.DataFrame(
        {
            'original': origiofv,
            'sampled_median': sampled_iofv.T.median(),
            'sampled_stdev': sampled_iofv.T.std(),
        }
    )
    iofv_summary['residual'] = (
        abs(iofv_summary['original'] - iofv_summary['sampled_median'])
        / iofv_summary['sampled_stdev']
    )
    iofv_summary['residual_outlier'] = iofv_summary['residual'] >= 3

    ids = iofv_summary.index[iofv_summary['residual_outlier']].tolist()
    id_plot = None
    if ids:
        try:
            id_plot = original_model.modelfit_results.plot_individual_predictions(
                predictions=['CIPREDI', 'PRED'], individuals=ids
            )
        except Exception:
            pass

    res = SimevalResults(
        sampled_iofv=sampled_iofv,
        iofv_summary=iofv_summary,
        individual_predictions_plot=id_plot,
    )
    return res


def psn_simeval_results(path):
    simfit_paths = (path / 'm1').glob('sim-*.mod')
    simfit_results = psn_simfit_results(simfit_paths)
    original = Model(path / 'm1' / 'original.mod')
    res = calculate_results(original, simfit_results)

    # Add CWRES outliers as 2 in data_flag
    # Reading PsN results for now
    # df = pd.read_csv(path / 'summary_cwres.csv')
    # outliers = df['OUTLIER'].fillna(0).astype(int)
    # outliers.replace({1.0: 2}, inplace=True)
    # outliers = outliers + res.data_flag
    # outliers.replace({3: 1}, inplace=True)
    # res.data_flag = outliers
    return res
