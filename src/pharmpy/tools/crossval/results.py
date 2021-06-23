import re

import pandas as pd

from pharmpy import Model
from pharmpy.results import Results


class CrossvalResults(Results):
    """Crossval results class"""

    def __init__(self, runs=None, prediction_ofv_sum=None):
        self.runs = runs
        self.prediction_ofv_sum = prediction_ofv_sum


def calculate_results(estimation_models, prediction_models):
    """Calculate crossval results"""
    est_ofvs = [model.modelfit_results.ofv for model in estimation_models]
    pred_ofvs = [model.modelfit_results.ofv for model in prediction_models]
    runs = pd.DataFrame(
        {'estimation_ofv': est_ofvs, 'prediction_ofv': pred_ofvs},
        index=pd.RangeIndex(start=1, stop=len(est_ofvs) + 1),
    )

    res = CrossvalResults(runs=runs, prediction_ofv_sum=sum(pred_ofvs))
    return res


def psn_crossval_results(path):
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    est_paths = [str(p) for p in (path / 'm1').glob('est_model*.mod')]
    est_paths.sort(key=natural_keys)
    pred_paths = [str(p) for p in (path / 'm1').glob('pred_model*.mod')]
    pred_paths.sort(key=natural_keys)
    ests = [Model(path) for path in est_paths]
    preds = [Model(path) for path in pred_paths]

    res = calculate_results(ests, preds)
    return res
