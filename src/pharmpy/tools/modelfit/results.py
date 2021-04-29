import pandas as pd

from pharmpy.results import Results


class AggregatedModelfitResults(Results):
    def __init__(self, ofv=None, parameter_estimates=None):
        self.ofv = ofv
        self.parameter_estimates = parameter_estimates


def calculate_results(models):
    names = [model.name for model in models]

    ofvs = [model.modelfit_results.ofv for model in models]
    ofv = pd.Series(ofvs, index=names)
    ofv.name = 'OFV'

    params = pd.DataFrame(
        [model.modelfit_results.parameter_estimates for model in models], index=names
    )

    res = AggregatedModelfitResults(ofv=ofv, parameter_estimates=params)
    return res
