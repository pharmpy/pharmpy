from pharmpy.model import Model, Results
from pharmpy.plugins.nonmem.results import simfit_results


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
        model = Model.create_model(path)
        modelfit_results.extend(simfit_results(model, path))
    res = calculate_results(modelfit_results)
    return res
