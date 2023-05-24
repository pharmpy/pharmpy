from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pharmpy.deps import pandas as pd
from pharmpy.model import Model, Results
from pharmpy.modeling import plot_individual_predictions
from pharmpy.tools import read_modelfit_results
from pharmpy.tools.simfit.results import psn_simfit_results


@dataclass(frozen=True)
class SimevalResults(Results):
    """Simeval results class"""

    sampled_iofv: Optional[Any] = None
    iofv_summary: Optional[Any] = None
    individual_predictions_plot: Optional[Any] = None


def calculate_results(original_model, original_results, simfit_results):
    """Calculate simeval results"""
    sampled_iofv = pd.concat(
        [res.individual_ofv for res in simfit_results.modelfit_results],
        axis=1,
        keys=range(1, len(simfit_results.modelfit_results) + 1),
    )
    origiofv = original_results.individual_ofv
    iofv_summary = pd.DataFrame(
        {
            'original': origiofv,
            'sampled_mean': sampled_iofv.T.mean(),
            'sampled_stdev': sampled_iofv.T.std(),
        }
    )
    iofv_summary['residual'] = (
        iofv_summary['original'] - iofv_summary['sampled_mean']
    ) / iofv_summary['sampled_stdev']
    iofv_summary['residual_q1'] = (
        iofv_summary['original'] - sampled_iofv.T.quantile(0.25)
    ) / iofv_summary['sampled_stdev']
    iofv_summary['residual_q3'] = (
        iofv_summary['original'] - sampled_iofv.T.quantile(0.75)
    ) / iofv_summary['sampled_stdev']
    iofv_summary['residual_outlier'] = iofv_summary['residual'] >= 3

    ids = iofv_summary.index[iofv_summary['residual_outlier']].tolist()
    id_plot = None
    if ids:
        try:
            id_plot = plot_individual_predictions(
                original_model,
                original_results.predictions[['IPRED', 'PRED']],
                individuals=ids,
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
    path = Path(path)
    simfit_paths = (path / 'm1').glob('sim-*.mod')
    simfit_results = psn_simfit_results(simfit_paths)
    original = Model.parse_model(path / 'm1' / 'original.mod')
    original_results = read_modelfit_results(path / 'm1' / 'original.mod')
    res = calculate_results(original, original_results, simfit_results)

    # Add CWRES outliers as 2 in data_flag
    # Reading PsN results for now
    # df = pd.read_csv(path / 'summary_cwres.csv')
    # outliers = df['OUTLIER'].fillna(0).astype(int)
    # outliers.replace({1.0: 2}, inplace=True)
    # outliers = outliers + res.data_flag
    # outliers.replace({3: 1}, inplace=True)
    # res.data_flag = outliers
    return res
