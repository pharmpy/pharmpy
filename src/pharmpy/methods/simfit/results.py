from pharmpy import Model
from pharmpy.plugins.nonmem.results import simfit_results
from pharmpy.results import Results


class SimfitResults(Results):
    """Simfit results class"""

    def __init__(self):
        pass


def calculate_results(modelfit_results):
    """Calculate simfit results"""
    res = SimfitResults()
    res.modelfit_results = modelfit_results
    return res


def psn_simfit_results(paths):
    modelfit_results = []
    for path in paths:
        model = Model(path)
        modelfit_results.extend(simfit_results(model))
    res = calculate_results(modelfit_results)
    return res
